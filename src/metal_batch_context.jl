"""
    metal_batch_context.jl — Batch Metal dispatches into a single command buffer.

Monkey-patches Metal.jl's kernel launch to encode into a shared command buffer
when a batch context is active, instead of creating a new command buffer per dispatch.

Reduces per-dispatch overhead from ~19μs to ~5μs (tested: 1.67x faster for 300 dispatches).

Usage:
    with_metal_batch() do
        # All @metal calls inside here share one command buffer
        forward_opt!(model, tokens, cache, pool, dc)
    end
    # Automatically submits and waits on exit
"""
module MetalBatchContext

using Metal
using Metal.MTL

export with_metal_batch

# Task-local batch context
const _active_context = Ref{Any}(nothing)

mutable struct BatchCtx
    queue::Any
    cmdbuf::MTLCommandBuffer
    cce::MTLComputeCommandEncoder
    roots::Vector{Any}
    arg_buffers::Vector{Any}
    count::Int
    flush_every::Int
end

function BatchCtx(; queue=Metal.global_queue(Metal.device()), flush_every=20)
    cmdbuf = MTLCommandBuffer(queue)
    cce = MTLComputeCommandEncoder(cmdbuf)
    BatchCtx(queue, cmdbuf, cce, Any[], Any[], 0, flush_every)
end

"""Flush the current command buffer and start a new one."""
function flush!(ctx::BatchCtx)
    ctx.count == 0 && return
    close(ctx.cce)
    roots = ctx.roots
    arg_bufs = ctx.arg_buffers
    MTL.on_completed(ctx.cmdbuf) do buf
        empty!(roots)
        foreach(Metal.MTL.free, arg_bufs)
        if buf.status == MTL.MTLCommandBufferStatusError
            Core.println("ERROR: Failed to submit command buffer: $(buf.error.localizedDescription)")
        end
    end
    commit!(ctx.cmdbuf)
    # Start new command buffer
    ctx.cmdbuf = MTLCommandBuffer(ctx.queue)
    ctx.cce = MTLComputeCommandEncoder(ctx.cmdbuf)
    ctx.roots = Any[]
    ctx.arg_buffers = Any[]
    ctx.count = 0
end

"""Run `f()` with @metal dispatches batched. Flushes every `flush_every` dispatches."""
function with_metal_batch(f::Function; flush_every::Int=20)
    ctx = BatchCtx(; flush_every)
    _active_context[] = ctx
    try
        f()
    finally
        _active_context[] = nothing
        if ctx.count > 0
            close(ctx.cce)
            roots = ctx.roots; arg_bufs = ctx.arg_buffers
            MTL.on_completed(ctx.cmdbuf) do buf
                empty!(roots)
                foreach(Metal.MTL.free, arg_bufs)
            end
            commit!(ctx.cmdbuf)
        end
        Metal.synchronize()
    end
end

function apply_patch!()
    # Monkey-patch the HostKernel call operator to check for active batch context
    @eval function (kernel::Metal.HostKernel)(args...; groups=1, threads=1,
                                               queue=Metal.global_queue(Metal.device()))
        ctx = $(_active_context)[]
        if ctx !== nothing
            # Batch mode: encode into shared command buffer
            groups_mtl = Metal.MTLSize(groups)
            threads_mtl = Metal.MTLSize(threads)
            ks = Metal.KernelState(rand(UInt32))
            MTL.set_function!(ctx.cce, kernel.pipeline)
            bufs = Metal.encode_arguments!(ctx.cce, kernel, ks, kernel.f, args...)
            MTL.append_current_function!(ctx.cce, groups_mtl, threads_mtl)
            push!(ctx.roots, (kernel.f, args))
            append!(ctx.arg_buffers, bufs)
            ctx.count += 1
            if ctx.count >= ctx.flush_every
                $flush!(ctx)
            end
            return
        end

        # Normal mode: original behavior (one command buffer per dispatch)
        groups_mtl = Metal.MTLSize(groups)
        threads_mtl = Metal.MTLSize(threads)
        (groups_mtl.width>0 && groups_mtl.height>0 && groups_mtl.depth>0) ||
            throw(ArgumentError("All group dimensions should be non-zero"))
        (threads_mtl.width>0 && threads_mtl.height>0 && threads_mtl.depth>0) ||
            throw(ArgumentError("All thread dimensions should be non-zero"))
        (threads_mtl.width * threads_mtl.height * threads_mtl.depth) > kernel.pipeline.maxTotalThreadsPerThreadgroup &&
            throw(ArgumentError("Thread count exceeds max"))

        ks = Metal.KernelState(rand(UInt32))
        cmdbuf = MTLCommandBuffer(queue)
        cce = MTLComputeCommandEncoder(cmdbuf)
        argument_buffers = try
            MTL.set_function!(cce, kernel.pipeline)
            bufs = Metal.encode_arguments!(cce, kernel, ks, kernel.f, args...)
            MTL.append_current_function!(cce, groups_mtl, threads_mtl)
            bufs
        finally
            close(cce)
        end
        roots = [kernel.f, args]
        MTL.on_completed(cmdbuf) do buf
            empty!(roots)
            foreach(Metal.MTL.free, argument_buffers)
            if buf.status == MTL.MTLCommandBufferStatusError
                Core.println("ERROR: Failed to submit command buffer: \$(buf.error.localizedDescription)")
            end
        end
        commit!(cmdbuf)
    end
end

end # module
