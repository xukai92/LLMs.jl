"""
Optimized quantized matmul v2: scalar dequant with register-level batch reuse.

Key insight: the natural access pattern for 4-bit weights is sequential
packed words (8 values each). Simdgroup matrix ops require reshuffling
into 8×8 tiles which is slower than just processing sequentially.

Strategy for B>1:
- One threadgroup per output row (NOT per (row, batch) pair)
- Threads cooperate on the inner dimension (packed columns)
- Each thread reads a packed word, unpacks 8 values, and multiplies
  against ALL B activation vectors (weights read once, reused B times)
- Final simdgroup reduction per batch element

This is the same approach as MLX's qmv kernel extended to multiple
batch elements — the weight read is amortized across the batch.
"""

# For B≤4: 4 accumulators in registers, unrolled
function qmm_v2_b4_kernel!(out, x, packed, scales, biases, B::Int32)
    row = Int32(threadgroup_position_in_grid().x)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(packed, 2))
    shared = MtlThreadGroupArray(Float32, 32)

    acc1 = 0.0f0; acc2 = 0.0f0; acc3 = 0.0f0; acc4 = 0.0f0

    pc = tid
    while pc <= packed_cols
        @inbounds pv = packed[row, pc]
        col_base = (pc - Int32(1)) << Int32(3)
        grp = (col_base >> Int32(6)) + Int32(1)
        @inbounds s = Float32(scales[row, grp])
        @inbounds bi = Float32(biases[row, grp])

        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            w = s * Float32((pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)) + bi
            if B >= Int32(1); @inbounds acc1 += w * Float32(x[col, 1]); end
            if B >= Int32(2); @inbounds acc2 += w * Float32(x[col, 2]); end
            if B >= Int32(3); @inbounds acc3 += w * Float32(x[col, 3]); end
            if B >= Int32(4); @inbounds acc4 += w * Float32(x[col, 4]); end
            k += Int32(1)
        end
        pc += tg_size
    end

    # Reduce and write each accumulator
    @inline function _reduce_and_write(acc, bidx)
        local a = acc
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            a += simd_shuffle_down(a, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds shared[wid] = a
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if wid == UInt32(1)
            a = lane <= nwarps ? (@inbounds shared[lane]) : 0.0f0
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                a += simd_shuffle_down(a, offset)
                offset <<= 1
            end
            if lane == UInt32(1)
                @inbounds out[row, bidx] = typeof(out[1,1])(a)
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    end

    if B >= Int32(1); _reduce_and_write(acc1, Int32(1)); end
    if B >= Int32(2); _reduce_and_write(acc2, Int32(2)); end
    if B >= Int32(3); _reduce_and_write(acc3, Int32(3)); end
    if B >= Int32(4); _reduce_and_write(acc4, Int32(4)); end

    return nothing
end

# For larger B: process batch in tiles of 4, one threadgroup per row
function qmm_v2_kernel!(out, x, packed, scales, biases, B::Int32)
    row = Int32(threadgroup_position_in_grid().x)
    btile = Int32(threadgroup_position_in_grid().y)  # batch tile index
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(packed, 2))
    shared = MtlThreadGroupArray(Float32, 32)

    TILE = Int32(4)
    b_base = (btile - Int32(1)) * TILE
    b_count = min(TILE, B - b_base)
    b1 = b_base + Int32(1)
    b2 = b_base + Int32(2)
    b3 = b_base + Int32(3)
    b4 = b_base + Int32(4)

    acc1 = 0.0f0; acc2 = 0.0f0; acc3 = 0.0f0; acc4 = 0.0f0

    pc = tid
    while pc <= packed_cols
        @inbounds pv = packed[row, pc]
        col_base = (pc - Int32(1)) << Int32(3)
        grp = (col_base >> Int32(6)) + Int32(1)
        @inbounds s = Float32(scales[row, grp])
        @inbounds bi = Float32(biases[row, grp])

        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            w = s * Float32((pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)) + bi
            if b_count >= Int32(1); @inbounds acc1 += w * Float32(x[col, b1]); end
            if b_count >= Int32(2); @inbounds acc2 += w * Float32(x[col, b2]); end
            if b_count >= Int32(3); @inbounds acc3 += w * Float32(x[col, b3]); end
            if b_count >= Int32(4); @inbounds acc4 += w * Float32(x[col, b4]); end
            k += Int32(1)
        end
        pc += tg_size
    end

    @inline function _reduce_write(acc, bidx)
        local a = acc
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            a += simd_shuffle_down(a, offset)
            offset <<= 1
        end
        if lane == UInt32(1); @inbounds shared[wid] = a; end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if wid == UInt32(1)
            a = lane <= nwarps ? (@inbounds shared[lane]) : 0.0f0
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                a += simd_shuffle_down(a, offset)
                offset <<= 1
            end
            if lane == UInt32(1); @inbounds out[row, bidx] = typeof(out[1,1])(a); end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    end

    if b_count >= Int32(1); _reduce_write(acc1, b1); end
    if b_count >= Int32(2); _reduce_write(acc2, b2); end
    if b_count >= Int32(3); _reduce_write(acc3, b3); end
    if b_count >= Int32(4); _reduce_write(acc4, b4); end

    return nothing
end

"""
Optimized quantized matmul v2: auto-selects kernel based on batch size.
- B=1: original qmv (one threadgroup per row)
- B=2-4: qmm_v2_b4 (one threadgroup per row, all B in registers)
- B>4: qmm_v2 (one threadgroup per (row, batch_tile), tile=4)
"""
function metal_qmatmul_v2!(out, x, packed, scales, biases;
                            group_size::Int=64)
    O = size(packed, 1)
    packed_cols = size(packed, 2)
    B = size(x, 2)

    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    if B == 1
        @metal threads=tg_size groups=(O, 1) qmatmul_kernel_g64!(
            out, x, packed, scales, biases)
    elseif B <= 4
        @metal threads=tg_size groups=(O, 1) qmm_v2_b4_kernel!(
            out, x, packed, scales, biases, Int32(B))
    else
        n_tiles = cld(B, 4)
        @metal threads=tg_size groups=(O, n_tiles) qmm_v2_kernel!(
            out, x, packed, scales, biases, Int32(B))
    end
    return out
end
