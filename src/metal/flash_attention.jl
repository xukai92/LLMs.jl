"""
Flash Attention: tiled attention without materializing the full score matrix.

Based on FlashAttention-2 (Dao 2023), adapted for Metal.

For each query position, iterate over KV in tiles:
1. Compute partial scores for this KV tile
2. Update running max (for numerical stability)
3. Rescale previous accumulator by exp(old_max - new_max)
4. Add exp(scores - new_max) weighted V to accumulator
5. Track running sum for final normalization

Memory: O(Br × Bc) shared memory instead of O(N²) global memory.

For GQA: each Q head maps to a KV head via gqa_ratio.
"""

# Tile size for KV iteration
# Bc = number of KV positions processed per tile
# Must fit in shared memory: Bc * head_dim * sizeof(Float32)

function flash_attn_kernel!(out, Q, K, V,
                             head_dim::Int32, n_kv_heads::Int32, gqa_ratio::Int32,
                             seq_kv::Int32, scale::Float32,
                             causal_offset::Int32, causal::Int32)
    # Grid: (n_q_heads, seq_q)
    # Each threadgroup computes attention for one (q_head, q_pos) pair
    qh = Int32(threadgroup_position_in_grid().x)
    sq = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)

    kvh = (qh - Int32(1)) ÷ gqa_ratio + Int32(1)

    # Running statistics for online softmax
    running_max = -Inf32
    running_sum = 0.0f0

    # Output accumulator in shared memory: (head_dim,) per threadgroup
    # Each thread handles a subset of head_dim elements
    acc = MtlThreadGroupArray(Float32, 128)  # head_dim ≤ 128
    d = tid
    while d <= head_dim
        @inbounds acc[d] = 0.0f0
        d += tg_size
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # Shared memory for broadcasting max and sum
    shared_ms = MtlThreadGroupArray(Float32, 2)  # [max, sum]

    # Iterate over KV positions
    # For simplicity, process one KV position at a time (Bc=1)
    # This is suboptimal but correct; we can tile later
    sk = Int32(1)
    while sk <= seq_kv
        # Causal mask check
        if causal == Int32(1) && sk > sq + causal_offset
            sk += Int32(1)
            continue
        end

        # Compute dot product: score = Q[:,qh,sq] · K[:,kvh,sk] * scale
        # Threads cooperate on the dot product
        local_dot = 0.0f0
        d = tid
        while d <= head_dim
            @inbounds local_dot += Float32(Q[d, qh, sq]) * Float32(K[d, kvh, sk])
            d += tg_size
        end

        # Reduce dot product across threads using simdgroup shuffle
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            local_dot += simd_shuffle_down(local_dot, offset)
            offset <<= 1
        end

        # Thread 0 in each simdgroup writes to shared
        lane = thread_index_in_simdgroup()
        wid = simdgroup_index_in_threadgroup()
        nwarps = simdgroups_per_threadgroup()

        dot_shared = MtlThreadGroupArray(Float32, 32)
        if lane == UInt32(1)
            @inbounds dot_shared[wid] = local_dot
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        if wid == UInt32(1)
            local_dot = lane <= nwarps ? (@inbounds dot_shared[lane]) : 0.0f0
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                local_dot += simd_shuffle_down(local_dot, offset)
                offset <<= 1
            end
            if lane == UInt32(1)
                score = local_dot * scale

                # Online softmax update
                new_max = max(running_max, score)
                # Rescale factor for previous accumulator
                rescale = exp(running_max - new_max)
                # New weight
                new_weight = exp(score - new_max)
                new_sum = running_sum * rescale + new_weight

                @inbounds shared_ms[1] = rescale
                @inbounds shared_ms[2] = new_weight

                running_max = new_max
                running_sum = new_sum
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        @inbounds rescale = shared_ms[1]
        @inbounds new_weight = shared_ms[2]

        # Update accumulator: acc = acc * rescale + new_weight * V[:,kvh,sk]
        d = tid
        while d <= head_dim
            @inbounds acc[d] = acc[d] * rescale + new_weight * Float32(V[d, kvh, sk])
            d += tg_size
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        sk += Int32(1)
    end

    # Normalize by sum and write output
    # Thread 1 broadcasts the final sum
    if tid == Int32(1)
        @inbounds shared_ms[1] = running_sum > 0.0f0 ? (1.0f0 / running_sum) : 0.0f0
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds inv_sum = shared_ms[1]

    d = tid
    while d <= head_dim
        @inbounds out[d, qh, sq] = typeof(out[1,1,1])(acc[d] * inv_sum)
        d += tg_size
    end

    return nothing
end

"""
    metal_flash_attention!(out, Q, K, V, scale; causal=true, causal_offset=0)

Flash attention: tiled computation without materializing N×N score matrix.
Memory usage: O(head_dim) per threadgroup instead of O(seq_kv).
"""
function metal_flash_attention!(out, Q, K, V, scale::Float32;
                                 causal::Bool=true, causal_offset::Int=0)
    head_dim, n_q_heads, seq_q = size(Q)
    _, n_kv_heads, seq_kv = size(K)
    gqa_ratio = n_q_heads ÷ n_kv_heads

    tg = min(head_dim, 128)
    tg = max(tg - (tg % 32), 32)

    @metal threads=tg groups=(n_q_heads, seq_q) flash_attn_kernel!(
        out, Q, K, V,
        Int32(head_dim), Int32(n_kv_heads), Int32(gqa_ratio), Int32(seq_kv),
        scale, Int32(causal_offset), Int32(causal ? 1 : 0))
    return out
end
