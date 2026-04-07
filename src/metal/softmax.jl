"""
Numerically stable softmax over the last dimension (columns).

    softmax(x)[i,j] = exp(x[i,j] - max_j(x[i,j])) / sum_j(exp(x[i,j] - max_j(x[i,j])))

For attention: applied to (seq_len_q, seq_len_kv) score matrices, one per head.
Each row is independently softmaxed.

One threadgroup per row. Uses simdgroup reductions for max and sum.
"""

# ── CPU reference ──

"""
    softmax_cpu!(out, x)

In-place softmax over columns (dim 2) for each row.
x is (rows, cols) — each row gets softmaxed independently.
"""
function softmax_cpu!(out::AbstractMatrix{T}, x::AbstractMatrix{T}) where T
    rows, cols = size(x)
    for r in 1:rows
        # Find max
        m = Float32(-Inf)
        @inbounds for c in 1:cols
            m = max(m, Float32(x[r, c]))
        end
        # Exp and sum
        s = 0.0f0
        @inbounds for c in 1:cols
            out[r, c] = T(exp(Float32(x[r, c]) - m))
            s += Float32(out[r, c])
        end
        # Normalize
        inv_s = 1.0f0 / s
        @inbounds for c in 1:cols
            out[r, c] = T(Float32(out[r, c]) * inv_s)
        end
    end
    return out
end

# ── Metal kernel ──

# Note: softmax is computed per-row of a 2D matrix.
# Layout: x[row, col] where we softmax over col dimension.
# One threadgroup per row.

function softmax_kernel!(out, x, cols::Int32)
    row = threadgroup_position_in_grid().x
    tid = thread_position_in_threadgroup().x
    tg_size = threads_per_threadgroup().x
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    shared = MtlThreadGroupArray(Float32, 32)

    # ── Phase 1: Find max ──
    local_max = -Inf32
    c = tid
    while c <= cols
        @inbounds local_max = max(local_max, Float32(x[row, c]))
        c += tg_size
    end

    # Simdgroup reduce max
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        local_max = max(local_max, simd_shuffle_down(local_max, offset))
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared[wid] = local_max
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if wid == UInt32(1)
        local_max = if lane <= nwarps
            @inbounds shared[lane]
        else
            -Inf32
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            local_max = max(local_max, simd_shuffle_down(local_max, offset))
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds shared[1] = local_max
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds row_max = shared[1]

    # ── Phase 2: Exp and sum ──
    local_sum = 0.0f0
    c = tid
    while c <= cols
        @inbounds begin
            val = exp(Float32(x[row, c]) - row_max)
            out[row, c] = typeof(x[1,1])(val)
            local_sum += val
        end
        c += tg_size
    end

    # Simdgroup reduce sum
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        local_sum += simd_shuffle_down(local_sum, offset)
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared[wid] = local_sum
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if wid == UInt32(1)
        local_sum = if lane <= nwarps
            @inbounds shared[lane]
        else
            0.0f0
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            local_sum += simd_shuffle_down(local_sum, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds shared[1] = 1.0f0 / local_sum
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds inv_sum = shared[1]

    # ── Phase 3: Normalize ──
    c = tid
    while c <= cols
        @inbounds out[row, c] = typeof(x[1,1])(Float32(out[row, c]) * inv_sum)
        c += tg_size
    end

    return nothing
end

function metal_softmax!(out, x)
    rows, cols = size(x)
    tg_size = min(cols, 1024)
    tg_size = tg_size - (tg_size % 32)
    tg_size = max(tg_size, 32)  # at least one simdgroup

    @metal threads=tg_size groups=rows softmax_kernel!(
        out, x, Int32(cols))
    return out
end
