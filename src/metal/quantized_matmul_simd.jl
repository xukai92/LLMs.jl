"""
Quantized matmul using simdgroup matrix multiply-accumulate.

Uses Metal's simdgroup_multiply_accumulate for 8×8 matrix tiles,
which maps to hardware matrix multiply units on Apple Silicon (M1+).

Strategy for batched qmm (B>1):
- Each simdgroup computes a 8×8 output tile
- Weights are dequantized into 8×8 tiles in registers
- Activations are loaded as 8×8 tiles
- simdgroup_multiply_accumulate does the 8×8 matmul in hardware
- Iterate over K dimension in chunks of 8

For the dequantized matmul: out[O, B] = dequant(W)[O, I] @ x[I, B]
We tile: M=O (output rows), N=B (batch), K=I (inner/reduction)
Each simdgroup handles an 8×8 output tile: out[m:m+8, n:n+8]
"""

# The simdgroup matrix ops work on 8x8 tiles stored across 32 threads.
# simdgroup_load reads an 8x8 tile from a 2D array
# simdgroup_store writes an 8x8 tile back
# simdgroup_multiply_accumulate: C = A * B + C (all 8x8)

# For our quantized matmul, we need to:
# 1. Dequantize an 8×8 tile of W into threadgroup memory
# 2. Load it as a simdgroup matrix
# 3. Load an 8×8 tile of x
# 4. Multiply-accumulate

function qmm_simd_kernel!(out, x, packed, scales, biases,
                           M::Int32, N::Int32, K::Int32)
    # Grid: (M/8, N/8) simdgroups
    # Each simdgroup computes one 8×8 output tile
    sg_m = Int32(threadgroup_position_in_grid().x)  # which 8-row tile
    sg_n = Int32(threadgroup_position_in_grid().y)  # which 8-col tile
    lane = thread_index_in_simdgroup()

    # Base indices for this tile (1-indexed)
    m_base = (sg_m - Int32(1)) * Int32(8)
    n_base = (sg_n - Int32(1)) * Int32(8)

    # Check bounds
    if m_base + Int32(8) > M || n_base + Int32(8) > N
        return nothing
    end

    # Shared memory for dequantized weight tile (8×8) and activation tile (8×8)
    w_tile = MtlThreadGroupArray(Float32, (8, 8))
    x_tile = MtlThreadGroupArray(Float32, (8, 8))

    # Accumulator tile (in simdgroup registers)
    # Initialize to zero
    acc = simdgroup_load(MtlThreadGroupArray(Float32, (8, 8)), (1, 1))
    # Zero it
    zero_tile = MtlThreadGroupArray(Float32, (8, 8))
    # Fill zero_tile with zeros (each thread fills its portion)
    tid = Int32(thread_index_in_simdgroup())
    if tid <= Int32(32)
        row = ((tid - Int32(1)) % Int32(8)) + Int32(1)
        col = ((tid - Int32(1)) ÷ Int32(8)) + Int32(1)
        if col <= Int32(8)
            @inbounds zero_tile[row, col] = 0.0f0
        end
    end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(zero_tile, (1, 1))

    # Iterate over K in chunks of 8
    k = Int32(0)
    while k < K
        # Dequantize 8×8 weight tile: W[m_base+1:m_base+8, k+1:k+8]
        # Each of the 32 threads fills some elements
        if tid <= Int32(32)
            # Thread fills 2 elements (8×8 = 64 elements / 32 threads)
            for elem in Int32(0):Int32(1)
                flat_idx = (Int32(tid) - Int32(1)) * Int32(2) + elem
                wr = (flat_idx % Int32(8)) + Int32(1)  # row within tile
                wc = (flat_idx ÷ Int32(8)) + Int32(1)  # col within tile

                if wc <= Int32(8)
                    global_row = m_base + wr
                    global_col = k + wc

                    if global_row <= M && global_col <= K
                        # Dequantize this element
                        pc = (global_col - Int32(1)) ÷ Int32(8) + Int32(1)
                        bit_idx = (global_col - Int32(1)) % Int32(8)
                        @inbounds packed_val = packed[global_row, pc]
                        quantized = (packed_val >> (UInt32(bit_idx) << UInt32(2))) & UInt32(0xF)

                        group_idx = (global_col - Int32(1)) ÷ Int32(64) + Int32(1)
                        @inbounds s = Float32(scales[global_row, group_idx])
                        @inbounds bi = Float32(biases[global_row, group_idx])

                        @inbounds w_tile[wr, wc] = s * Float32(quantized) + bi
                    else
                        @inbounds w_tile[wr, wc] = 0.0f0
                    end

                    # Load activation tile
                    global_x_row = k + wr  # K dim
                    global_x_col = n_base + wc  # B dim
                    if global_x_row <= K && global_x_col <= N
                        @inbounds x_tile[wr, wc] = Float32(x[global_x_row, global_x_col])
                    else
                        @inbounds x_tile[wr, wc] = 0.0f0
                    end
                end
            end
        end
        simdgroup_barrier(Metal.MemoryFlagThreadGroup)

        # Load tiles into simdgroup registers and multiply-accumulate
        w_mat = simdgroup_load(w_tile, (1, 1))
        x_mat = simdgroup_load(x_tile, (1, 1))
        acc = simdgroup_multiply_accumulate(w_mat, x_mat, acc)

        simdgroup_barrier(Metal.MemoryFlagThreadGroup)
        k += Int32(8)
    end

    # Store result tile
    result_tile = MtlThreadGroupArray(Float32, (8, 8))
    simdgroup_store(acc, result_tile, (1, 1))
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)

    # Write back to global memory (convert to output type)
    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat_idx = (Int32(tid) - Int32(1)) * Int32(2) + elem
            wr = (flat_idx % Int32(8)) + Int32(1)
            wc = (flat_idx ÷ Int32(8)) + Int32(1)
            if wc <= Int32(8)
                global_row = m_base + wr
                global_col = n_base + wc
                if global_row <= M && global_col <= N
                    @inbounds out[global_row, global_col] = typeof(out[1,1])(result_tile[wr, wc])
                end
            end
        end
    end

    return nothing
end

"""
Quantized matmul using simdgroup matrix multiply (for B≥8).
Falls back to scalar kernel for B<8.
"""
function metal_quantized_matmul_simd!(out, x, packed, scales, biases;
                                       group_size::Int=64)
    O = size(packed, 1)
    packed_cols = size(packed, 2)
    I = packed_cols * 8
    B = size(x, 2)

    if B >= 8 && O % 8 == 0
        # Use simdgroup kernel — one simdgroup per 8×8 output tile
        # Grid: (O/8, B/8), each group is one simdgroup (32 threads)
        @metal threads=32 groups=(O÷8, cld(B, 8)) qmm_simd_kernel!(
            out, x, packed, scales, biases, Int32(O), Int32(B), Int32(I))
    else
        # Fallback to scalar kernel
        tg_size = min(packed_cols, 256)
        tg_size = max(tg_size - (tg_size % 32), 32)
        @metal threads=tg_size groups=(O, B) qmatmul_kernel_g64!(
            out, x, packed, scales, biases)
    end
    return out
end
