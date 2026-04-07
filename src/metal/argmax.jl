"""
GPU argmax kernel — finds the index of the maximum value in a column.

Used to avoid copying the full 128K vocab logits vector back to CPU
just to find the top token.
"""

function argmax_kernel!(result, data, n::Int32)
    # Single threadgroup reduction to find argmax of data[1:n]
    # result[1] = argmax index (0-indexed for token ID)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    # Shared memory for partial results: (value, index) per simdgroup
    shared_val = MtlThreadGroupArray(Float32, 32)
    shared_idx = MtlThreadGroupArray(Int32, 32)

    # Phase 1: each thread finds its local max
    local_max = -Inf32
    local_idx = Int32(0)
    i = tid
    while i <= n
        @inbounds val = Float32(data[i])
        if val > local_max
            local_max = val
            local_idx = i
        end
        i += tg_size
    end

    # Phase 2: simdgroup reduction
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        other_val = simd_shuffle_down(local_max, offset)
        other_idx = simd_shuffle_down(local_idx, offset)
        if other_val > local_max
            local_max = other_val
            local_idx = other_idx
        end
        offset <<= 1
    end

    if lane == UInt32(1)
        @inbounds shared_val[wid] = local_max
        @inbounds shared_idx[wid] = local_idx
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # Phase 3: first simdgroup reduces across all simdgroups
    if wid == UInt32(1)
        local_max = if lane <= nwarps
            @inbounds shared_val[lane]
        else
            -Inf32
        end
        local_idx = if lane <= nwarps
            @inbounds shared_idx[lane]
        else
            Int32(0)
        end

        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            other_val = simd_shuffle_down(local_max, offset)
            other_idx = simd_shuffle_down(local_idx, offset)
            if other_val > local_max
                local_max = other_val
                local_idx = other_idx
            end
            offset <<= 1
        end

        if lane == UInt32(1)
            @inbounds result[1] = local_idx - Int32(1)  # 0-indexed token ID
        end
    end

    return nothing
end

"""
    metal_argmax_last_col(logits) -> Int32

Find the argmax of the last column of logits on GPU.
Returns a 0-indexed token ID.
Only copies 4 bytes (one Int32) back to CPU instead of the full logits vector.
"""
function metal_argmax_last_col(logits)
    vocab_size = size(logits, 1)
    seq_len = size(logits, 2)

    # Get last column
    last_col = view(logits, :, seq_len)

    result = MtlArray(Int32[0])
    tg = min(vocab_size, 1024)
    tg = max(tg - (tg % 32), 32)
    @metal threads=tg groups=1 argmax_kernel!(result, last_col, Int32(vocab_size))

    return Array(result)[1]
end

# Pre-allocated version that reuses the result buffer
function metal_argmax_last_col!(result::MtlVector{Int32}, logits)
    vocab_size = size(logits, 1)
    seq_len = size(logits, 2)
    last_col = view(logits, :, seq_len)
    tg = min(vocab_size, 1024)
    tg = max(tg - (tg % 32), 32)
    @metal threads=tg groups=1 argmax_kernel!(result, last_col, Int32(vocab_size))
    return result
end
