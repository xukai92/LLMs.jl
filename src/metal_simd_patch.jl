"""
    metal_simd_patch.jl — Patch for Metal.jl simdgroup ops with runtime origins

This file patches `simdgroup_load` and `simdgroup_store` to work with
runtime-dependent matrix_origin arguments (e.g., computed from
simdgroup_index_in_threadgroup()).

## Problem
Metal.jl's `convert_origin` constructs `NTuple{2, VecElement{Int64}}`
from the origin tuple. When the origin is runtime-computed, Julia's
compiler emits `jl_apply_generic` for the tuple construction, which
GPUCompiler rejects as "dynamic function invocation".

## Fix
Replace the tuple construction with direct `VecElement` wrapping using
`@inline` and ensure the entire call is inlineable. The key change is
making `convert_origin` use `@inline` and constructing VecElements
individually rather than as a tuple literal.

## How to apply to Metal.jl upstream
In `src/device/intrinsics/simd.jl`, change:
```julia
function convert_origin(origin::NTuple{2, Int64})
    return (VecElement{Int64}(origin[1]-1), VecElement{Int64}(origin[2]-1))
end
```
to:
```julia
@inline function convert_origin(origin::NTuple{2, Int64})
    o1 = VecElement{Int64}(origin[1] - Int64(1))
    o2 = VecElement{Int64}(origin[2] - Int64(1))
    return (o1, o2)
end
```

This file monkey-patches the existing methods to test the fix.
"""
module MetalSimdPatch

using Metal
using Metal: MtlDeviceArray
using Core: LLVMPtr, VecElement
using Metal.LLVM.Interop: @typed_ccall

# The fixed convert_origin — @inline ensures the compiler inlines
# the VecElement construction even with runtime arguments
@inline function convert_origin_fixed(origin::NTuple{2, Int64})
    o1 = VecElement{Int64}(origin[1] - Int64(1))
    o2 = VecElement{Int64}(origin[2] - Int64(1))
    return (o1, o2)
end

# Also try with explicit Int64 arithmetic to avoid any ambiguity
@inline function convert_origin_v2(col::Int64, row::Int64)
    return (VecElement{Int64}(col - Int64(1)), VecElement{Int64}(row - Int64(1)))
end

# Re-define simdgroup_load/store with the fixed convert_origin
# for Float32 on ThreadGroup memory (most common case for tiled matmul)

const AS_ThreadGroup = Metal.AS.ThreadGroup
const AS_Device = Metal.AS.Device

for (jltype, suffix) in ((:Float16, "f16"), (:Float32, "f32"))
    for as in (AS_Device, AS_ThreadGroup)
        # Patched simdgroup_load that works with runtime origins
        @eval begin
            @inline function simdgroup_load_patched(
                data::MtlDeviceArray{$jltype, <:Any, $as},
                matrix_origin::NTuple{2, Int64},
            )
                origin = convert_origin_fixed(matrix_origin)
                return @typed_ccall($"air.simdgroup_matrix_8x8_load.v64$suffix.p$as$suffix",
                    llvmcall, NTuple{64, VecElement{$jltype}},
                    (LLVMPtr{$jltype, $as}, Int64, NTuple{2, VecElement{Int64}}, Bool),
                    pointer(data), size(data)[1], origin, Val(true))
            end

            @inline function simdgroup_store_patched(
                src::NTuple{64, VecElement{$jltype}},
                dest::MtlDeviceArray{$jltype, <:Any, $as},
                matrix_origin::NTuple{2, Int64},
            )
                origin = convert_origin_fixed(matrix_origin)
                return @typed_ccall($"air.simdgroup_matrix_8x8_store.v64$suffix.p$as$suffix",
                    llvmcall, Cvoid,
                    (NTuple{64, VecElement{$jltype}}, LLVMPtr{$jltype, $as}, Int64, NTuple{2, VecElement{Int64}}, Bool),
                    src, pointer(dest), size(dest)[1], origin, Val(true))
            end
        end
    end
end

export simdgroup_load_patched, simdgroup_store_patched

end # module
