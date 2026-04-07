"""
MetalBatch: batched dispatch with pre-allocated argument buffers.

Two key optimizations over vanilla @metal:
1. One command buffer + encoder for all dispatches (saves ~3.8μs × N)
2. Pre-allocated argument buffers, reused via unsafe_store! (saves ~6μs × args × N)

Combined savings: ~10-12μs per dispatch → ~4ms for 340-dispatch forward pass.
"""

using Metal.MTL: MTLCommandBuffer, MTLComputeCommandEncoder, MTLSize, MTLBuffer,
    set_function!, set_buffer!, dispatchThreadgroups!, commit!

"""
Pre-allocated argument buffer pool for one kernel signature.
Stores MTLBuffers sized for each non-buffer argument position.
"""
struct ArgPool
    buffers::Vector{MTLBuffer}   # one per argument position
    sizes::Vector{Int}           # sizeof each arg type
end

"""
MetalBatch with argument buffer caching.
"""
mutable struct MetalBatch
    queue::Any
    cmdbuf::Union{Nothing, MTLCommandBuffer}
    cce::Union{Nothing, MTLComputeCommandEncoder}
    dispatch_count::Int
    # Cache: kernel function hash → ArgPool
    arg_cache::Dict{UInt64, ArgPool}
    # GC roots
    roots::Vector{Any}
    # Kernel state buffer (reused across all dispatches)
    ks_buf::MTLBuffer
end

function MetalBatch()
    dev = Metal.device()
    ks_buf = Metal.MTL.alloc(dev, sizeof(Metal.KernelState); storage=Metal.SharedStorage)
    MetalBatch(Metal.global_queue(dev), nothing, nothing, 0,
               Dict{UInt64, ArgPool}(), Any[], ks_buf)
end

function open!(batch::MetalBatch)
    batch.cmdbuf = MTLCommandBuffer(batch.queue)
    batch.cce = MTLComputeCommandEncoder(batch.cmdbuf)
    empty!(batch.roots)
    batch.dispatch_count = 0
end

"""
Get or create pre-allocated argument buffers for a kernel signature.
"""
function get_arg_pool!(batch::MetalBatch, converted_args::Tuple)::ArgPool
    # Hash based on types
    h = hash(map(typeof, converted_args))
    pool = get(batch.arg_cache, h, nothing)
    if pool !== nothing
        return pool
    end

    # Create new pool
    dev = Metal.device()
    buffers = MTLBuffer[]
    sizes = Int[]
    for carg in converted_args
        T = typeof(carg)
        if T <: MTLBuffer || T <: Metal.MtlPtr ||
           Metal.isghosttype(T) || Core.Compiler.isconstType(T)
            # These don't need arg buffers
            push!(buffers, MTLBuffer())  # placeholder
            push!(sizes, 0)
        else
            buf = Metal.MTL.alloc(dev, sizeof(T); storage=Metal.SharedStorage)
            push!(buffers, buf)
            push!(sizes, sizeof(T))
        end
    end

    pool = ArgPool(buffers, sizes)
    batch.arg_cache[h] = pool
    return pool
end

"""
Dispatch a kernel using pre-allocated argument buffers.
Only unsafe_store! (0.17μs) instead of alloc (6μs) per arg.
"""
function dispatch!(batch::MetalBatch, f, args...; threads=1, groups=1)
    converted_args = map(a -> Metal.mtlconvert(a, batch.cce), args)
    tt = Tuple{map(Core.Typeof, converted_args)...}
    hk = Metal.mtlfunction(f, tt)

    set_function!(batch.cce, hk.pipeline)

    # Kernel state (reuse pre-allocated buffer)
    ks = Metal.KernelState(rand(UInt32))
    unsafe_store!(convert(Ptr{Metal.KernelState}, batch.ks_buf), ks)
    set_buffer!(batch.cce, batch.ks_buf, 0, 1)

    # Get pre-allocated arg buffers
    pool = get_arg_pool!(batch, converted_args)

    # Encode arguments using pre-allocated buffers
    idx = 2
    for (i, carg) in enumerate(converted_args)
        T = typeof(carg)
        if T <: MTLBuffer
            set_buffer!(batch.cce, carg, 0, idx)
        elseif T <: Metal.MtlPtr
            set_buffer!(batch.cce, carg.buffer, carg.offset, idx)
        elseif Metal.isghosttype(T) || Core.Compiler.isconstType(T)
            continue
        else
            # Reuse pre-allocated buffer — just store new value
            unsafe_store!(convert(Ptr{T}, pool.buffers[i]), carg)
            set_buffer!(batch.cce, pool.buffers[i], 0, idx)
        end
        idx += 1
    end

    dispatchThreadgroups!(batch.cce, MTLSize(groups), MTLSize(threads))
    push!(batch.roots, args)  # keep MtlArrays alive
    batch.dispatch_count += 1
end

function close_and_commit!(batch::MetalBatch)
    if batch.cce !== nothing
        close(batch.cce)
        batch.cce = nothing
    end
    if batch.cmdbuf !== nothing
        roots = batch.roots
        Metal.MTL.on_completed(batch.cmdbuf) do buf
            empty!(roots)
        end
        commit!(batch.cmdbuf)
        batch.cmdbuf = nothing
    end
end
