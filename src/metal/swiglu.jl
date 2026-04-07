"""
SwiGLU activation: silu(gate) * up

    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    swiglu(gate, up) = silu(gate) * up

This is a fused element-wise operation on two vectors of the same size.
For Llama-3.2-3B: intermediate_size=8192.
"""

# ── CPU reference ──

function swiglu_cpu!(out::AbstractArray{T}, gate::AbstractArray{T},
                     up::AbstractArray{T}) where T
    @assert size(out) == size(gate) == size(up)
    @inbounds for i in eachindex(out)
        g = Float32(gate[i])
        u = Float32(up[i])
        # silu(g) * u = g * sigmoid(g) * u
        out[i] = T(g / (1.0f0 + exp(-g)) * u)
    end
    return out
end

# ── Metal kernel ──

function swiglu_kernel!(out, gate, up, n::Int32)
    i = thread_position_in_grid_1d()
    if i <= n
        @inbounds begin
            g = Float32(gate[i])
            u = Float32(up[i])
            out[i] = typeof(out[i])(g / (1.0f0 + exp(-g)) * u)
        end
    end
    return nothing
end

function metal_swiglu!(out, gate, up)
    n = length(out)
    threads_per_group = 256
    groups = cld(n, threads_per_group)
    @metal threads=threads_per_group groups=groups swiglu_kernel!(
        out, gate, up, Int32(n))
    return out
end
