"""
    metal_dispatch_ext.jl — Proposed extensions to Metal.jl for reduced dispatch overhead

Self-contained module that can be extracted for a Metal.jl PR.

# Problem
Each `@metal` creates a command buffer + allocates MTLBuffers for args.
For 340 dispatches: ~10ms overhead (28μs × 340).

# Solution: MetalCommandBatch
Encodes multiple dispatches into one command buffer. Uses the same
`@generated encode_arguments!` as @metal for zero-overhead arg encoding,
but shares the command buffer/encoder across dispatches.

# What this would look like as a Metal.jl change:
In execution.jl, add a `batch_call` method to HostKernel that takes
an existing (cmdbuf, cce) pair instead of creating new ones.
"""
module MetalDispatchExt

using Metal
using Metal.MTL: MTLCommandBuffer, MTLComputeCommandEncoder, MTLSize,
    MTLBuffer, set_function!, set_buffer!, dispatchThreadgroups!, commit!

export MetalCommandBatch, batch_dispatch!, batch_submit!, batch_wait!

# ════════════════════════════════════════════════════════════════
# MetalCommandBatch — the clean API for batched dispatch
# ════════════════════════════════════════════════════════════════

"""
    MetalCommandBatch()

Accumulates kernel dispatches into a single MTLCommandBuffer.

    batch = MetalCommandBatch()
    # Use @metal launch=false to get compiled kernels, then dispatch:
    k1 = @metal launch=false kernel1!(args...)
    k2 = @metal launch=false kernel2!(args...)

    batch_dispatch!(batch, k1, args1...; threads=T1, groups=G1)
    batch_dispatch!(batch, k2, args2...; threads=T2, groups=G2)
    batch_submit!(batch)
    batch_wait!(batch)

Saves ~4μs per dispatch (command buffer + encoder creation) plus
reduces GPU scheduling overhead by submitting all work at once.
"""
mutable struct MetalCommandBatch
    queue::Any
    cmdbuf::MTLCommandBuffer
    cce::MTLComputeCommandEncoder
    roots::Vector{Any}
    arg_buffers::Vector{Any}
    count::Int
end

function MetalCommandBatch(; queue=Metal.global_queue(Metal.device()))
    cmdbuf = MTLCommandBuffer(queue)
    cce = MTLComputeCommandEncoder(cmdbuf)
    MetalCommandBatch(queue, cmdbuf, cce, Any[], Any[], 0)
end

"""
    batch_dispatch!(batch, kernel::HostKernel, args...; threads, groups)

Encode a kernel dispatch into the batch's command buffer.
`kernel` should be obtained via `@metal launch=false f(args...)`.

This is the efficient path: it uses the same compiled pipeline as @metal,
and the argument encoding is done by the same @generated function.
"""
function batch_dispatch!(batch::MetalCommandBatch, kernel, args...; threads=1, groups=1)
    groups_mtl = MTLSize(groups)
    threads_mtl = MTLSize(threads)

    ks = Metal.KernelState(rand(UInt32))

    set_function!(batch.cce, kernel.pipeline)
    bufs = Metal.encode_arguments!(batch.cce, kernel, ks, kernel.f, args...)
    Metal.MTL.append_current_function!(batch.cce, groups_mtl, threads_mtl)

    push!(batch.roots, (kernel.f, args))
    for b in bufs
        push!(batch.arg_buffers, b)
    end
    batch.count += 1
end

"""Submit all encoded dispatches."""
function batch_submit!(batch::MetalCommandBatch)
    close(batch.cce)
    roots = batch.roots
    arg_bufs = batch.arg_buffers
    Metal.MTL.on_completed(batch.cmdbuf) do _
        foreach(Metal.MTL.free, arg_bufs)
        empty!(roots)
    end
    commit!(batch.cmdbuf)
end

"""Wait for the batch to complete (alias for Metal.synchronize)."""
function batch_wait!(batch::MetalCommandBatch)
    Metal.synchronize()
end

end # module
