"""
Scaled Dot-Product Attention with Grouped-Query Attention (GQA).

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

With GQA: n_q_heads=24, n_kv_heads=8, ratio=3. Each KV head is shared by 3 Q heads.

For the initial implementation we use a simple two-pass approach:
  1. Compute scores = Q @ K^T / sqrt(d_k)  [per head]
  2. Apply causal mask
  3. Softmax over seq_kv dimension
  4. Multiply by V

This is NOT flash attention — it materializes the full attention matrix.
Fine for short sequences; will need tiling for long contexts.

Shapes (Julia column-major, following MLX convention):
  Q: (head_dim, n_q_heads, seq_q)
  K: (head_dim, n_kv_heads, seq_kv)
  V: (head_dim, n_kv_heads, seq_kv)
  output: (head_dim, n_q_heads, seq_q)
"""

# ── CPU reference ──

"""
    attention_cpu!(out, Q, K, V, n_kv_heads, scale; mask=nothing)

Grouped-query attention on CPU. Q/K/V as 3D arrays.
`mask`: optional (seq_q, seq_kv) matrix where `true` means MASKED (set to -inf).
"""
function attention_cpu!(out::AbstractArray{T, 3}, Q::AbstractArray{T, 3},
                        K::AbstractArray{T, 3}, V::AbstractArray{T, 3},
                        scale::Float32;
                        mask::Union{Nothing, AbstractMatrix{Bool}}=nothing) where T
    head_dim, n_q_heads, seq_q = size(Q)
    _, n_kv_heads, seq_kv = size(K)
    gqa_ratio = n_q_heads ÷ n_kv_heads

    for qh in 1:n_q_heads
        kvh = (qh - 1) ÷ gqa_ratio + 1  # which KV head this Q head uses

        # Compute scores: (seq_q, seq_kv)
        scores = zeros(Float32, seq_q, seq_kv)
        for sq in 1:seq_q
            for sk in 1:seq_kv
                dot = 0.0f0
                @inbounds for d in 1:head_dim
                    dot += Float32(Q[d, qh, sq]) * Float32(K[d, kvh, sk])
                end
                scores[sq, sk] = dot * scale
            end
        end

        # Apply mask (causal or custom)
        if mask !== nothing
            for sq in 1:seq_q, sk in 1:seq_kv
                if mask[sq, sk]
                    scores[sq, sk] = -Inf32
                end
            end
        end

        # Softmax per query position
        for sq in 1:seq_q
            m = maximum(@view scores[sq, :])
            s = 0.0f0
            for sk in 1:seq_kv
                scores[sq, sk] = exp(scores[sq, sk] - m)
                s += scores[sq, sk]
            end
            scores[sq, :] ./= s
        end

        # Weighted sum of V
        for sq in 1:seq_q
            @inbounds for d in 1:head_dim
                acc = 0.0f0
                for sk in 1:seq_kv
                    acc += scores[sq, sk] * Float32(V[d, kvh, sk])
                end
                out[d, qh, sq] = T(acc)
            end
        end
    end
    return out
end

# ── Metal kernels ──

# Strategy: Two-kernel approach
# Kernel 1: Compute Q@K^T scores, apply mask and softmax (one threadgroup per (q_head, q_pos))
# Kernel 2: Multiply softmax weights by V (one threadgroup per (q_head, q_pos))
#
# For a simple first implementation, we'll launch one threadgroup per (q_head, q_pos)
# and have it iterate over all KV positions. This isn't optimal but is correct.

# Kernel 1: Compute attention scores + softmax
# One threadgroup per (q_head, q_pos). Threads cooperate over seq_kv.
# Output: scores matrix (n_q_heads, seq_q, seq_kv)

function attn_scores_softmax_kernel!(scores_out, Q, K,
                                     head_dim::Int32, n_kv_heads::Int32,
                                     gqa_ratio::Int32, seq_kv::Int32,
                                     scale::Float32, causal_offset::Int32,
                                     causal::Int32)
    # Grid: (n_q_heads, seq_q)
    qh = Int32(threadgroup_position_in_grid().x)
    sq = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    kvh = (qh - Int32(1)) ÷ gqa_ratio + Int32(1)

    shared = MtlThreadGroupArray(Float32, 32)

    # ── Compute dot products for this (qh, sq) across all sk ──
    # Each thread handles a subset of sk positions

    # We need to store per-sk scores. Since seq_kv could be large,
    # we do multiple passes: first compute+store scores, then softmax.
    # Scores are written to scores_out[qh, sq, sk].

    # Pass 1: Compute dot products and write to global memory
    sk = tid
    while sk <= seq_kv
        dot = 0.0f0
        d = Int32(1)
        while d <= head_dim
            @inbounds dot += Float32(Q[d, qh, sq]) * Float32(K[d, kvh, sk])
            d += Int32(1)
        end
        score = dot * scale

        # Causal mask: mask if sk > sq + causal_offset
        # causal_offset accounts for KV cache (positions before this chunk)
        if causal == Int32(1) && sk > sq + causal_offset
            score = -Inf32
        end

        @inbounds scores_out[qh, sq, sk] = score
        sk += tg_size
    end
    threadgroup_barrier(Metal.MemoryFlagDevice)

    # Pass 2: Softmax — find max
    local_max = -Inf32
    sk = tid
    while sk <= seq_kv
        @inbounds local_max = max(local_max, Float32(scores_out[qh, sq, sk]))
        sk += tg_size
    end

    # Reduce max
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

    # Pass 3: Exp and sum
    local_sum = 0.0f0
    sk = tid
    while sk <= seq_kv
        @inbounds begin
            val = exp(Float32(scores_out[qh, sq, sk]) - row_max)
            scores_out[qh, sq, sk] = val
            local_sum += val
        end
        sk += tg_size
    end

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

    # Pass 4: Normalize
    sk = tid
    while sk <= seq_kv
        @inbounds scores_out[qh, sq, sk] = Float32(scores_out[qh, sq, sk]) * inv_sum
        sk += tg_size
    end

    return nothing
end

# Kernel 2: Weighted sum of V using softmax scores
# One threadgroup per (q_head, q_pos). Threads cooperate over head_dim.

function attn_value_kernel!(out, scores, V,
                            head_dim::Int32, n_kv_heads::Int32,
                            gqa_ratio::Int32, seq_kv::Int32)
    qh = Int32(threadgroup_position_in_grid().x)
    sq = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)

    kvh = (qh - Int32(1)) ÷ gqa_ratio + Int32(1)

    # Each thread handles a subset of head_dim elements
    d = tid
    while d <= head_dim
        acc = 0.0f0
        sk = Int32(1)
        while sk <= seq_kv
            @inbounds acc += Float32(scores[qh, sq, sk]) * Float32(V[d, kvh, sk])
            sk += Int32(1)
        end
        @inbounds out[d, qh, sq] = typeof(out[1,1,1])(acc)
        d += tg_size
    end

    return nothing
end

"""
    metal_attention!(out, Q, K, V, scale; causal=true, causal_offset=0)

Compute grouped-query attention on Metal.
`causal`: whether to apply causal masking.
`causal_offset`: number of previous KV positions (for decode with KV cache).
"""
function metal_attention!(out::MtlArray{T, 3}, Q::MtlArray{T, 3},
                          K::MtlArray{T, 3}, V::MtlArray{T, 3},
                          scale::Float32;
                          causal::Bool=true,
                          causal_offset::Int=0) where T
    head_dim, n_q_heads, seq_q = size(Q)
    _, n_kv_heads, seq_kv = size(K)
    gqa_ratio = n_q_heads ÷ n_kv_heads

    # Allocate temporary scores buffer: (n_q_heads, seq_q, seq_kv)
    scores = MtlArray(zeros(Float32, n_q_heads, seq_q, seq_kv))

    # Kernel 1: scores + softmax
    tg1 = min(seq_kv, 1024)
    tg1 = max(tg1 - (tg1 % 32), 32)
    @metal threads=tg1 groups=(n_q_heads, seq_q) attn_scores_softmax_kernel!(
        scores, Q, K,
        Int32(head_dim), Int32(n_kv_heads), Int32(gqa_ratio), Int32(seq_kv),
        scale, Int32(causal_offset), Int32(causal ? 1 : 0))

    # Kernel 2: weighted V sum
    tg2 = min(head_dim, 256)
    @metal threads=tg2 groups=(n_q_heads, seq_q) attn_value_kernel!(
        out, scores, V,
        Int32(head_dim), Int32(n_kv_heads), Int32(gqa_ratio), Int32(seq_kv))

    return out
end
