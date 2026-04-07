"""
    SafeTensors reader for loading model weights.

The safetensors format:
- 8 bytes: little-endian UInt64 header length
- N bytes: JSON header mapping tensor names to {dtype, shape, data_offsets}
- Remaining bytes: raw tensor data (concatenated, aligned)

Reference: https://huggingface.co/docs/safetensors/
"""

# Map safetensors dtype strings to Julia types
const DTYPE_MAP = Dict{String, DataType}(
    "F16"  => Float16,
    "F32"  => Float32,
    "F64"  => Float64,
    "BF16" => UInt16,   # BFloat16 stored as UInt16, convert later if needed
    "I8"   => Int8,
    "I16"  => Int16,
    "I32"  => Int32,
    "I64"  => Int64,
    "U8"   => UInt8,
    "U16"  => UInt16,
    "U32"  => UInt32,
    "U64"  => UInt64,
    "BOOL" => Bool,
)

"""
Metadata about a single tensor in a safetensors file.
"""
struct SafeTensorInfo
    name::String
    dtype::DataType
    shape::Vector{Int}
    data_offset_start::Int  # byte offset from start of data section
    data_offset_end::Int    # byte offset end (exclusive)
end

"""
    parse_safetensors_header(path::String) -> (Vector{SafeTensorInfo}, Dict, Int)

Parse the header of a safetensors file. Returns:
- Vector of SafeTensorInfo for each tensor
- The raw parsed header dict (for metadata access)
- The byte offset where the data section begins
"""
function parse_safetensors_header(path::String)
    open(path, "r") do io
        # Read header length (8 bytes, little-endian UInt64)
        header_len_bytes = read(io, 8)
        header_len = reinterpret(UInt64, header_len_bytes)[1]

        # Read and parse JSON header
        header_json = read(io, header_len)
        header = JSON3.read(String(header_json))

        data_offset = 8 + header_len  # where tensor data starts in the file

        tensors = SafeTensorInfo[]
        metadata = Dict{String, Any}()

        for (key, val) in pairs(header)
            name = String(key)
            if name == "__metadata__"
                for (mk, mv) in pairs(val)
                    metadata[String(mk)] = String(mv)
                end
                continue
            end

            dtype_str = String(val.dtype)
            dtype = get(DTYPE_MAP, dtype_str, nothing)
            if dtype === nothing
                @warn "Unknown dtype $dtype_str for tensor $name, skipping"
                continue
            end

            # Shape comes as JSON array — safetensors uses row-major (C) order,
            # but we keep the shape as-is and handle layout at load time
            shape = Int[val.shape[i] for i in 1:length(val.shape)]

            offsets = val.data_offsets
            push!(tensors, SafeTensorInfo(name, dtype, shape, Int(offsets[1]), Int(offsets[2])))
        end

        return tensors, metadata, Int(data_offset)
    end
end

"""
    load_safetensors(path::String; mmap_data::Bool=true) -> Dict{String, Array}

Load all tensors from a safetensors file. Returns a Dict mapping tensor names
to Julia arrays.

If `mmap_data=true` (default), the file is memory-mapped for zero-copy access.
This is ideal for Apple Silicon's unified memory — the data stays in place and
can be wrapped into MtlArrays without copying.

Tensors are returned in their original (row-major) memory layout. For safetensors,
shape [O, I] means O rows, I columns in row-major. Julia is column-major, so the
returned array has reversed dimensions: Array of size (I, O) in Julia's convention,
giving the same memory layout.
"""
function load_safetensors(path::String; mmap_data::Bool=true)
    tensors_info, metadata, data_offset = parse_safetensors_header(path)

    result = Dict{String, Array}()

    if mmap_data
        # Memory-map the entire file
        io = open(path, "r")
        file_data = Mmap.mmap(io, Vector{UInt8})

        for info in tensors_info
            nbytes = info.data_offset_end - info.data_offset_start
            elem_size = sizeof(info.dtype)

            if nbytes == 0
                # Empty tensor
                julia_shape = length(info.shape) == 0 ? () : tuple(reverse(info.shape)...)
                result[info.name] = Array{info.dtype}(undef, julia_shape...)
                continue
            end

            # View into mmap'd data
            byte_start = data_offset + info.data_offset_start + 1  # 1-indexed
            byte_end = data_offset + info.data_offset_end
            raw = @view file_data[byte_start:byte_end]

            # Reinterpret as the target dtype
            typed = reinterpret(info.dtype, raw)

            # Reverse shape for Julia's column-major layout
            # safetensors shape [O, I] -> Julia size (I, O) so memory layout matches
            if length(info.shape) == 0
                result[info.name] = typed[1:1]  # scalar
            elseif length(info.shape) == 1
                result[info.name] = typed
            else
                julia_shape = tuple(reverse(info.shape)...)
                result[info.name] = reshape(typed, julia_shape)
            end
        end

        # Keep a reference to prevent GC of mmap'd data
        # Store it under a special key
        result["__mmap_handle__"] = file_data
        result["__mmap_io__"] = [io]  # wrapped in array to fit Dict type

        return result
    else
        # Read into memory (copy)
        file_data = read(path)

        for info in tensors_info
            nbytes = info.data_offset_end - info.data_offset_start
            elem_size = sizeof(info.dtype)

            byte_start = data_offset + info.data_offset_start + 1
            byte_end = data_offset + info.data_offset_end

            raw = file_data[byte_start:byte_end]
            typed = reinterpret(info.dtype, raw)

            if length(info.shape) == 0
                result[info.name] = copy(typed[1:1])
            elseif length(info.shape) == 1
                result[info.name] = copy(typed)
            else
                julia_shape = tuple(reverse(info.shape)...)
                result[info.name] = reshape(copy(typed), julia_shape)
            end
        end

        return result
    end
end

"""
    load_safetensors_lazy(path::String) -> (Vector{SafeTensorInfo}, Function)

Return tensor metadata and a loader function that loads individual tensors on demand.
Useful for inspecting weight layouts without loading everything into memory.

Usage:
    infos, loader = load_safetensors_lazy("model.safetensors")
    for info in infos
        println(info.name, " ", info.dtype, " ", info.shape)
    end
    tensor = loader(infos[1])  # load just one tensor
"""
function load_safetensors_lazy(path::String)
    tensors_info, metadata, data_offset = parse_safetensors_header(path)

    function load_tensor(info::SafeTensorInfo)
        open(path, "r") do io
            nbytes = info.data_offset_end - info.data_offset_start
            seek(io, data_offset + info.data_offset_start)
            raw = read(io, nbytes)
            typed = reinterpret(info.dtype, raw)
            if length(info.shape) <= 1
                return copy(typed)
            else
                return reshape(copy(typed), tuple(reverse(info.shape)...))
            end
        end
    end

    return tensors_info, load_tensor
end
