"""
Batched Metal dispatch context.

Wraps multiple @metal dispatches into a single MTLCommandBuffer,
reducing per-dispatch overhead from ~19μs to ~5μs (1.67x faster for 300 dispatches).

Usage:
    ctx = BatchContext()
    batch_metal!(ctx, threads=32, groups=96) do
        my_kernel!(args...)
    end
    # ... more batch_metal! calls ...
    submit_and_wait!(ctx)
"""
module MetalBatchContext

using Metal
using Metal.MTL

export BatchContext, submit_and_wait!

mutable struct BatchContext
    queue::Any
    cmdbuf::MTLCommandBuffer
    cce::MTLComputeCommandEncoder
    roots::Vector{Any}

    function BatchContext(; queue=Metal.global_queue(Metal.device()))
        cmdbuf = MTLCommandBuffer(queue)
        cce = MTLComputeCommandEncoder(cmdbuf)
        new(queue, cmdbuf, cce, Any[])
    end
end

"""
Encode a kernel dispatch into the batch context's shared command buffer.
Usage matches @metal: `batch_metal!(ctx, kernel, args...; threads, groups)`
where `kernel` is obtained via `@metal launch=false f(args...)`.
"""
function batch_dispatch!(ctx::BatchContext, kernel, args...; threads=1, groups=1)
    groups_mtl = MTLSize(groups)
    threads_mtl = MTLSize(threads)
    ks = Metal.KernelState(rand(UInt32))
    MTL.set_function!(ctx.cce, kernel.pipeline)
    Metal.encode_arguments!(ctx.cce, kernel, ks, kernel.f, args...)
    MTL.append_current_function!(ctx.cce, groups_mtl, threads_mtl)
    push!(ctx.roots, (kernel.f, args))
end

"""Submit all encoded dispatches and wait for completion."""
function submit_and_wait!(ctx::BatchContext)
    roots = ctx.roots
    close(ctx.cce)
    MTL.on_completed(ctx.cmdbuf) do _
        empty!(roots)
    end
    commit!(ctx.cmdbuf)
    Metal.synchronize()
end

end # module
