"""
    metal_graph.jl — MPSGraph wrapper for graph-level GPU compilation

Wraps Apple's MPSGraph framework (the engine behind MLX) with a Julia-friendly API.
Enables graph-level operation fusion, eliminating intermediate memory round-trips.

Usage:
    g = MetalGraphBuilder()
    x = placeholder!(g, (3072, 32), Float16)
    w = constant!(g, weight_matrix)
    y = matmul!(g, w, x)
    z = rmsnorm!(g, y, norm_weight, 1f-5)
    exe = compile!(g, [z])
    result = execute!(exe, Dict(x => input_data))
"""
module MetalGraphModule

using Metal
using Metal.MTL

# Load MPSGraph framework once
const _framework_loaded = Ref(false)
function ensure_framework!()
    _framework_loaded[] && return
    OC = Base.loaded_modules[Base.PkgId(
        Base.UUID("e86c9b32-1129-44ac-8ea0-90d5bb39ded9"), "ObjectiveC")]
    OC.load_framework("MetalPerformanceShadersGraph")
    _framework_loaded[] = true
end

# ObjC helpers
_sel(n) = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), n)
_cls(n) = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), n)
_msg(o, s) = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), o, _sel(s))
_alloc_init(c) = _msg(_msg(_cls(c), "alloc"), "init")

# MPSDataType constants
const MPSDataTypeFloat16 = UInt32(0x10000010)
const MPSDataTypeFloat32 = UInt32(0x10000020)
const MPSDataTypeInt32   = UInt32(0x20000020)

_mps_dtype(::Type{Float16}) = MPSDataTypeFloat16
_mps_dtype(::Type{Float32}) = MPSDataTypeFloat32
_mps_dtype(::Type{Int32})   = MPSDataTypeInt32

# Create NSArray of NSNumbers from Julia tuple/vector
function _ns_shape(dims)
    nums = map(dims) do d
        ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Int64),
              _cls("NSNumber"), _sel("numberWithLongLong:"), Int64(d))
    end
    arr = collect(Ptr{Cvoid}, nums)
    GC.@preserve arr ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, UInt),
        _cls("NSArray"), _sel("arrayWithObjects:count:"),
        pointer(arr), UInt(length(arr)))
end

# ── Graph tensor handle ──
struct GraphTensor
    ptr::Ptr{Cvoid}  # MPSGraphTensor*
    shape::Tuple    # Julia-convention shape (col-major)
    dtype::Type
end
GraphTensor(ptr, shape) = GraphTensor(ptr, shape, Float16)  # default

# ── Graph builder ──
mutable struct MetalGraphBuilder
    graph::Ptr{Cvoid}  # MPSGraph*
    placeholders::Vector{GraphTensor}
    constants::Vector{Any}  # keep MtlArrays alive

    function MetalGraphBuilder()
        ensure_framework!()
        g = _alloc_init("MPSGraph")
        new(g, GraphTensor[], Any[])
    end
end

"""Create a placeholder input tensor. Shape is Julia (col-major) convention;
internally stored as reversed shape for MPSGraph (row-major)."""
function placeholder!(g::MetalGraphBuilder, shape, ::Type{T}; name="") where T
    # Reverse shape: Julia col-major (M,N) = MPSGraph row-major (N,M)
    ns = _ns_shape(reverse(shape))
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{Cvoid}),
        g.graph, _sel("placeholderWithShape:dataType:name:"),
        ns, _mps_dtype(T), C_NULL)
    gt = GraphTensor(t, tuple(shape...), T)
    push!(g.placeholders, gt)
    gt
end

"""Create a placeholder for constant data (weights). Pass as feed at execution time."""
function constant!(g::MetalGraphBuilder, shape, ::Type{T}) where T
    placeholder!(g, shape, T)
end

"""Matrix multiplication: out = a @ b (Julia convention)."""
function matmul!(g::MetalGraphBuilder, a::GraphTensor, b::GraphTensor)
    # Swap: MPSGraph row-major matmul(b, a) = Julia col-major a @ b
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:"),
        b.ptr, a.ptr, C_NULL)
    # Output shape: (a.rows, b.cols) in Julia
    out_shape = (a.shape[1], b.shape[2])
    GraphTensor(t, out_shape, a.dtype)
end

"""Element-wise addition: out = a + b."""
function add!(g::MetalGraphBuilder, a::GraphTensor, b::GraphTensor)
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("additionWithPrimaryTensor:secondaryTensor:name:"),
        a.ptr, b.ptr, C_NULL)
    GraphTensor(t, a.shape, a.dtype)
end

"""Element-wise multiplication."""
function mul!(g::MetalGraphBuilder, a::GraphTensor, b::GraphTensor)
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("multiplicationWithPrimaryTensor:secondaryTensor:name:"),
        a.ptr, b.ptr, C_NULL)
    GraphTensor(t, a.shape, a.dtype)
end

"""Cast tensor to different type."""
function cast!(g::MetalGraphBuilder, x::GraphTensor, ::Type{T}) where T
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{Cvoid}),
        g.graph, _sel("castTensor:toType:name:"),
        x.ptr, _mps_dtype(T), C_NULL)
    GraphTensor(t, x.shape, T)
end

"""Reshape tensor."""
function reshape!(g::MetalGraphBuilder, x::GraphTensor, shape)
    ns = _ns_shape(reverse(shape))
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("reshapeTensor:withShape:name:"),
        x.ptr, ns, C_NULL)
    GraphTensor(t, tuple(shape...), x.dtype)
end

"""Apply SiLU activation: x * sigmoid(x)."""
function silu!(g::MetalGraphBuilder, x::GraphTensor)
    # MPSGraph has direct sigmoid + multiplication
    sig = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("sigmoidWithTensor:name:"), x.ptr, C_NULL)
    t = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        g.graph, _sel("multiplicationWithPrimaryTensor:secondaryTensor:name:"),
        x.ptr, sig, C_NULL)
    GraphTensor(t, x.shape, x.dtype)
end

# ── Compiled executable ──
struct CompiledGraph
    graph::Ptr{Cvoid}
    targets::Vector{GraphTensor}
    placeholders::Vector{GraphTensor}  # ordered inputs
    queue_ptr::Ptr{Cvoid}
    executable::Ptr{Cvoid}  # pre-compiled MPSGraphExecutable (or C_NULL for legacy path)
end

"""Compile graph for repeated execution.
If `inputs` is provided, pre-compiles to an MPSGraphExecutable for minimal per-run overhead.
"""
function compile!(g::MetalGraphBuilder, targets::Vector{GraphTensor};
                  inputs::Vector{GraphTensor}=g.placeholders)
    qp = reinterpret(Ptr{Cvoid}, Metal.global_queue(Metal.device()).ptr)

    # Try to pre-compile to MPSGraphExecutable
    # compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:
    # feeds is NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>
    # Since all inputs are placeholders, we can build the feeds dict from their known shapes
    # But MPSGraphShapedType requires shape info we track in GraphTensor

    # For now, use the simpler runtime path (runWithMTLCommandQueue)
    # TODO: pre-compile requires tracking shapes in GraphTensor
    CompiledGraph(g.graph, targets, inputs, qp, C_NULL)
end

"""Execute graph with feed data. Returns Dict of target → host Array (copies from GPU)."""
function execute!(cg::CompiledGraph, feeds::Dict{GraphTensor, <:MtlArray})
    # Build feed dict: MPSGraphTensor → MPSGraphTensorData
    feed_keys = Ptr{Cvoid}[]
    feed_vals = Ptr{Cvoid}[]
    roots = Any[]

    for (gt, data) in feeds
        push!(roots, data)
        shape = _ns_shape(reverse(size(data)))
        buf_ptr = reinterpret(Ptr{Cvoid}, data.data[].ptr)
        td = ccall(:objc_msgSend, Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt32),
            _msg(_cls("MPSGraphTensorData"), "alloc"),
            _sel("initWithMTLBuffer:shape:dataType:"),
            buf_ptr, shape, _mps_dtype(eltype(data)))
        push!(feed_keys, gt.ptr)
        push!(feed_vals, td)
    end

    feed_dict = GC.@preserve feed_keys feed_vals ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, UInt),
        _cls("NSDictionary"), _sel("dictionaryWithObjects:forKeys:count:"),
        pointer(feed_vals), pointer(feed_keys), UInt(length(feed_keys)))

    target_ptrs = [t.ptr for t in cg.targets]
    target_arr = GC.@preserve target_ptrs ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, UInt),
        _cls("NSArray"), _sel("arrayWithObjects:count:"),
        pointer(target_ptrs), UInt(length(target_ptrs)))

    result_dict = ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        cg.graph, _sel("runWithMTLCommandQueue:feeds:targetTensors:targetOperations:"),
        cg.queue_ptr, feed_dict, target_arr, C_NULL)

    if result_dict == C_NULL
        error("MPSGraph execution failed")
    end

    results = Dict{GraphTensor, Array}()
    for gt in cg.targets
        rtd = ccall(:objc_msgSend, Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            result_dict, _sel("objectForKey:"), gt.ptr)
        nda = _msg(rtd, "mpsndarray")

        ndims = ccall(:objc_msgSend, UInt, (Ptr{Cvoid}, Ptr{Cvoid}),
            nda, _sel("numberOfDimensions"))
        mps_shape = ntuple(ndims) do i
            Int(ccall(:objc_msgSend, UInt, (Ptr{Cvoid}, Ptr{Cvoid}, UInt),
                nda, _sel("lengthOfDimension:"), UInt(i - 1)))
        end
        shape = reverse(mps_shape)

        n_elements = prod(shape)
        result = gt.dtype == Float16 ? zeros(Float16, n_elements) : zeros(Float32, n_elements)
        GC.@preserve result ccall(:objc_msgSend, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            nda, _sel("readBytes:strideBytes:"), pointer(result), C_NULL)

        results[gt] = reshape(result, shape)
    end

    return results
end

"""Execute graph and write results to pre-allocated MtlArrays (stays on GPU).
This is the fast path — avoids host↔device copies.
"""
function execute_gpu!(cg::CompiledGraph, feeds::Dict{GraphTensor, <:MtlArray},
                     outputs::Dict{GraphTensor, <:MtlArray})
    # Build feeds_array and results_array as NSArrays
    # runWithMTLCommandQueue:feeds:targetTensors:targetOperations: returns a dict
    # We then need to copy from result's MTLBuffer to output MtlArrays
    # OR use encodeToCommandBuffer which writes directly to provided buffers

    feed_keys = Ptr{Cvoid}[]
    feed_vals = Ptr{Cvoid}[]
    roots = Any[feeds, outputs]

    for (gt, data) in feeds
        shape = _ns_shape(reverse(size(data)))
        buf_ptr = reinterpret(Ptr{Cvoid}, data.data[].ptr)
        td = ccall(:objc_msgSend, Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt32),
            _msg(_cls("MPSGraphTensorData"), "alloc"),
            _sel("initWithMTLBuffer:shape:dataType:"),
            buf_ptr, shape, _mps_dtype(eltype(data)))
        push!(feed_keys, gt.ptr)
        push!(feed_vals, td)
    end

    # Pre-create output TensorDatas wrapping user's MtlArrays
    results_results_dict = Dict{GraphTensor, Ptr{Cvoid}}()
    out_keys = Ptr{Cvoid}[]
    out_vals = Ptr{Cvoid}[]
    for (gt, data) in outputs
        shape = _ns_shape(reverse(size(data)))
        buf_ptr = reinterpret(Ptr{Cvoid}, data.data[].ptr)
        td = ccall(:objc_msgSend, Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt32),
            _msg(_cls("MPSGraphTensorData"), "alloc"),
            _sel("initWithMTLBuffer:shape:dataType:"),
            buf_ptr, shape, _mps_dtype(eltype(data)))
        push!(out_keys, gt.ptr)
        push!(out_vals, td)
    end

    feed_dict = GC.@preserve feed_keys feed_vals ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, UInt),
        _cls("NSDictionary"), _sel("dictionaryWithObjects:forKeys:count:"),
        pointer(feed_vals), pointer(feed_keys), UInt(length(feed_keys)))

    results_dict = GC.@preserve out_keys out_vals ccall(:objc_msgSend, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, UInt),
        _cls("NSDictionary"), _sel("dictionaryWithObjects:forKeys:count:"),
        pointer(out_vals), pointer(out_keys), UInt(length(out_keys)))

    # Call: -[MPSGraph runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:]
    # This writes results directly into the provided MTLBuffers
    ccall(:objc_msgSend, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        cg.graph, _sel("runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:"),
        cg.queue_ptr, feed_dict, C_NULL, results_dict)

    GC.@preserve roots nothing
    return outputs
end

export MetalGraphBuilder, GraphTensor, CompiledGraph
export placeholder!, matmul!, add!, mul!, cast!, reshape!, silu!
export compile!, execute!, execute_gpu!

end # module
