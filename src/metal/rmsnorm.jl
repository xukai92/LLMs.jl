"""
RMSNorm: Root Mean Square Layer Normalization.

    y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]

One threadgroup per row. Each threadgroup:
1. Computes sum of squares via simdgroup reductions
2. Computes 1/sqrt(mean + eps)
3. Multiplies each element by weight * scale

For Llama-3.2-3B: hidden_size=3072, eps=1e-5
"""

# ── CPU reference ──

function rmsnorm_cpu!(out::AbstractMatrix{T}, x::AbstractMatrix{T},
                      weight::AbstractVector{T}, eps::Float32) where T
    hidden, batch = size(x)
    @assert size(out) == size(x)
    @assert length(weight) == hidden
    for b in 1:batch
        ss = zero(Float32)
        @inbounds for i in 1:hidden
            ss += Float32(x[i, b])^2
        end
        scale = 1.0f0 / sqrt(ss / hidden + Float32(eps))
        @inbounds for i in 1:hidden
            out[i, b] = T(Float32(x[i, b]) * scale * Float32(weight[i]))
        end
    end
    return out
end

# ── Metal kernel ──

# Each threadgroup handles one row (one token's hidden dim).
# threadgroup size = min(hidden, 1024), must be multiple of 32 for simdgroup.
# Multiple elements per thread if hidden > threadgroup_size.

function rmsnorm_kernel!(out, x, weight, hidden::Int32, eps::Float32)
    # Row index (one threadgroup per row)
    row = threadgroup_position_in_grid().x

    tid = thread_position_in_threadgroup().x
    tg_size = threads_per_threadgroup().x
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    # Shared memory for partial sums from each simdgroup
    shared = MtlThreadGroupArray(Float32, 32)

    # Phase 1: Each thread accumulates sum of squares for its elements
    ss = 0.0f0
    i = tid
    while i <= hidden
        @inbounds val = Float32(x[i, row])
        ss += val * val
        i += tg_size
    end

    # Phase 2: Simdgroup reduction
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        ss += simd_shuffle_down(ss, offset)
        offset <<= 1
    end

    # Write simdgroup partial to shared memory
    if lane == UInt32(1)
        @inbounds shared[wid] = ss
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # Phase 3: First simdgroup reduces across all simdgroups
    if wid == UInt32(1)
        ss = if lane <= nwarps
            @inbounds shared[lane]
        else
            0.0f0
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            ss += simd_shuffle_down(ss, offset)
            offset <<= 1
        end
        # Thread 1 has the total sum of squares
        if lane == UInt32(1)
            @inbounds shared[1] = 1.0f0 / sqrt(ss / Float32(hidden) + eps)
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # Phase 4: Apply normalization
    @inbounds scale = shared[1]
    i = tid
    while i <= hidden
        @inbounds out[i, row] = typeof(x[i, row])(Float32(x[i, row]) * scale * Float32(weight[i]))
        i += tg_size
    end

    return nothing
end

# Fused RMSNorm + residual add: out = rmsnorm(x + residual, weight)
# Also writes x = x + residual (in-place residual update)
function rmsnorm_residual_kernel!(out, x, residual, weight, hidden::Int32, eps::Float32)
    row = threadgroup_position_in_grid().x
    tid = thread_position_in_threadgroup().x
    tg_size = threads_per_threadgroup().x
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    shared = MtlThreadGroupArray(Float32, 32)

    # Phase 1: Add residual to x in-place, compute sum of squares
    ss = 0.0f0
    i = tid
    while i <= hidden
        @inbounds val = Float32(x[i, row]) + Float32(residual[i, row])
        @inbounds x[i, row] = typeof(x[1,1])(val)  # write back x = x + residual
        ss += val * val
        i += tg_size
    end

    # Simdgroup reduction (same as rmsnorm_kernel!)
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        ss += simd_shuffle_down(ss, offset)
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared[wid] = ss
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if wid == UInt32(1)
        ss = if lane <= nwarps
            @inbounds shared[lane]
        else
            0.0f0
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            ss += simd_shuffle_down(ss, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds shared[1] = 1.0f0 / sqrt(ss / Float32(hidden) + eps)
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    @inbounds scale = shared[1]
    i = tid
    while i <= hidden
        @inbounds out[i, row] = typeof(x[1,1])(Float32(x[i, row]) * scale * Float32(weight[i]))
        i += tg_size
    end
    return nothing
end

"""Fused RMSNorm with residual add: x += residual, out = rmsnorm(x, weight)"""
function metal_rmsnorm_residual!(out, x, residual, weight, eps::Float32)
    hidden, batch = size(x)
    tg_size = min(hidden, 1024)
    tg_size = tg_size - (tg_size % 32)
    @metal threads=tg_size groups=batch rmsnorm_residual_kernel!(
        out, x, residual, weight, Int32(hidden), eps)
    return out
end

function metal_rmsnorm!(out, x, weight, eps::Float32)
    hidden, batch = size(x)
    # Use up to 1024 threads per group, rounded down to multiple of 32
    tg_size = min(hidden, 1024)
    tg_size = tg_size - (tg_size % 32)  # align to simdgroup

    @metal threads=tg_size groups=batch rmsnorm_kernel!(
        out, x, weight, Int32(hidden), eps)
    return out
end
