"""
Tiled quantized matrix multiplication for batched inference.

Two kernel variants:
- qmv: matrix-vector (B=1 decode) — one threadgroup per output row
- qmm: matrix-matrix (B>1 prefill) — tiled, weight reuse across batch via shared memory

For qmm, the tiling strategy:
- Each threadgroup computes a TILE_M × TILE_N output tile
- TILE_M output rows, TILE_N batch elements
- Weights are loaded into shared memory TILE_K at a time and reused across TILE_N
- This reduces weight reads by factor of TILE_N

Layout: out[O, B] = W_dequant[O, I] @ x[I, B]
  W stored as packed[O, I/8] uint32 + scales[O, I/G] + biases[O, I/G]
"""

# ── Tiled batch kernel (qmm) ──

# Tile parameters (compile-time constants)
# TILE_M: output rows per threadgroup
# TILE_N: batch elements per threadgroup (the key reuse dimension)
# TILE_K: inner dimension chunk size processed per iteration
# For group_size=64 and 8 values per uint32, process 8 columns per packed word

# Simple tiled kernel: each threadgroup handles TILE_N batch elements for one output row.
# Threads within the group cooperate on the inner-product reduction,
# and the weight data is read once and reused for all TILE_N batch elements.

function qmm_tiled_kernel!(out, x, packed, scales, biases, B::Int32)
    # Grid: (O, ceil(B/TILE_N))  where TILE_N = number of batch elements per threadgroup
    row = Int32(threadgroup_position_in_grid().x)     # output row
    bn = Int32(threadgroup_position_in_grid().y)       # batch tile index
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(packed, 2))

    # Each thread accumulates for up to 8 batch elements (TILE_N=8)
    # Use registers for accumulation — no shared memory needed for output tile
    acc1 = 0.0f0
    acc2 = 0.0f0
    acc3 = 0.0f0
    acc4 = 0.0f0
    acc5 = 0.0f0
    acc6 = 0.0f0
    acc7 = 0.0f0
    acc8 = 0.0f0

    # Batch indices for this tile (0-indexed)
    b_base = (bn - Int32(1)) * Int32(8)
    b1 = b_base + Int32(1)
    b2 = b_base + Int32(2)
    b3 = b_base + Int32(3)
    b4 = b_base + Int32(4)
    b5 = b_base + Int32(5)
    b6 = b_base + Int32(6)
    b7 = b_base + Int32(7)
    b8 = b_base + Int32(8)

    # How many batch elements are valid in this tile
    b_count = min(Int32(8), B - b_base)

    # Iterate over packed weight columns
    pc = tid
    while pc <= packed_cols
        @inbounds packed_val = packed[row, pc]
        col_base = (pc - Int32(1)) << Int32(3)
        group_idx = (col_base >> Int32(6)) + Int32(1)  # ÷64 + 1
        @inbounds s = Float32(scales[row, group_idx])
        @inbounds bi = Float32(biases[row, group_idx])

        # Unpack 8 4-bit values and accumulate against all batch elements
        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            quantized = (packed_val >> (UInt32(k) << UInt32(2))) & UInt32(0xF)
            w = s * Float32(quantized) + bi

            # Read activation for each batch element and multiply
            # The weight value w is reused across all batch elements
            if b_count >= Int32(1)
                @inbounds acc1 += w * Float32(x[col, b1])
            end
            if b_count >= Int32(2)
                @inbounds acc2 += w * Float32(x[col, b2])
            end
            if b_count >= Int32(3)
                @inbounds acc3 += w * Float32(x[col, b3])
            end
            if b_count >= Int32(4)
                @inbounds acc4 += w * Float32(x[col, b4])
            end
            if b_count >= Int32(5)
                @inbounds acc5 += w * Float32(x[col, b5])
            end
            if b_count >= Int32(6)
                @inbounds acc6 += w * Float32(x[col, b6])
            end
            if b_count >= Int32(7)
                @inbounds acc7 += w * Float32(x[col, b7])
            end
            if b_count >= Int32(8)
                @inbounds acc8 += w * Float32(x[col, b8])
            end

            k += Int32(1)
        end
        pc += tg_size
    end

    # Simdgroup reduction — reduce each accumulator independently
    # Process one batch element at a time through the reduction tree
    shared = MtlThreadGroupArray(Float32, 32)

    # Helper: reduce one accumulator and write to output
    @inline function _reduce_write(acc_val, b_idx)
        local_acc = acc_val
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            local_acc += simd_shuffle_down(local_acc, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds shared[wid] = local_acc
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if wid == UInt32(1)
            local_acc = if lane <= nwarps
                @inbounds shared[lane]
            else
                0.0f0
            end
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                local_acc += simd_shuffle_down(local_acc, offset)
                offset <<= 1
            end
            if lane == UInt32(1)
                @inbounds out[row, b_idx] = typeof(out[1,1])(local_acc)
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    end

    if b_count >= Int32(1); _reduce_write(acc1, b1); end
    if b_count >= Int32(2); _reduce_write(acc2, b2); end
    if b_count >= Int32(3); _reduce_write(acc3, b3); end
    if b_count >= Int32(4); _reduce_write(acc4, b4); end
    if b_count >= Int32(5); _reduce_write(acc5, b5); end
    if b_count >= Int32(6); _reduce_write(acc6, b6); end
    if b_count >= Int32(7); _reduce_write(acc7, b7); end
    if b_count >= Int32(8); _reduce_write(acc8, b8); end

    return nothing
end

# ── Dispatch logic: auto-select qmv vs qmm ──

"""
Quantized matmul with automatic kernel selection:
- B=1: use existing qmv kernel (one threadgroup per row)
- B>1: use tiled qmm kernel (weight reuse across batch)
"""
function metal_quantized_matmul_auto!(out, x, packed, scales, biases;
                                      group_size::Int=64)
    O = size(packed, 1)
    packed_cols = size(packed, 2)
    B = size(x, 2)

    if B >= 16 && O % 8 == 0
        # B≥16: use simdgroup matrix multiply kernel
        I = packed_cols * 8
        @metal threads=32 groups=(O÷8, cld(B, 8)) qmm_simd_kernel!(
            out, x, packed, scales, biases, Int32(O), Int32(B), Int32(I))
    else
        # B<16 or non-aligned: use scalar kernel
        tg_size = min(packed_cols, 256)
        tg_size = max(tg_size - (tg_size % 32), 32)
        @metal threads=tg_size groups=(O, B) qmatmul_kernel_g64!(
            out, x, packed, scales, biases)
    end
    return out
end

"""Quantized linear layer with auto kernel selection."""
function qlinear_auto!(out, layer, x)
    metal_quantized_matmul_auto!(out, x, layer.weight, layer.scales, layer.biases;
                                  group_size=layer.group_size)
    return out
end
