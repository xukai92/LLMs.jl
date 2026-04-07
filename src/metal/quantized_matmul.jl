"""
4-bit Quantized Matrix-Vector / Matrix-Matrix Multiplication.

MLX quantization format:
- weight: UInt32 array (O, I/8) — each uint32 packs 8 4-bit values
- scales: Float16 array (O, I/group_size) — per-group scale
- biases: Float16 array (O, I/group_size) — per-group bias
- Dequantization: w_float = scale * float(unpack_4bit(packed)) + bias

Computes: out = x @ W^T (with on-the-fly dequantization)
  x: (I, B) input activations, Float16
  W: (O, I/8) packed weights, UInt32
  out: (O, B) output, Float16

For Llama-3.2-3B with group_size=64, bits=4:
  q_proj: I=3072, O=3072, packed=(3072, 384), scales=(3072, 48)
  k_proj: I=3072, O=1024, packed=(1024, 384), scales=(1024, 48)
"""

# ── CPU reference ──

"""
    dequantize_cpu(packed_weights, scales, biases, bits, group_size) -> Matrix{Float32}

Fully dequantize a quantized weight matrix on CPU.
Returns Float32 matrix of shape (O, I).
"""
function dequantize_cpu(packed::AbstractMatrix{UInt32},
                        scales::AbstractMatrix,
                        biases::AbstractMatrix;
                        bits::Int=4, group_size::Int=64)
    elems_per_u32 = 32 ÷ bits
    mask = UInt32((1 << bits) - 1)  # 0xF for 4-bit

    O, packed_cols = size(packed)
    I = packed_cols * elems_per_u32

    W = zeros(Float32, O, I)

    for row in 1:O
        for pc in 1:packed_cols
            @inbounds packed_val = packed[row, pc]
            for k in 0:elems_per_u32-1
                col = (pc - 1) * elems_per_u32 + k + 1
                shift = k * bits
                quantized = (packed_val >> shift) & mask

                group_idx = (col - 1) ÷ group_size + 1
                @inbounds s = Float32(scales[row, group_idx])
                @inbounds b = Float32(biases[row, group_idx])
                W[row, col] = s * Float32(quantized) + b
            end
        end
    end
    return W
end

"""
    quantized_matmul_cpu!(out, x, packed_weights, scales, biases; bits, group_size)

Compute out = x^T @ W^T = (W @ x)^T ... actually:
out[o, b] = sum_i W[o, i] * x[i, b]

where W is dequantized on-the-fly from packed_weights/scales/biases.
x: (I, B), out: (O, B)
"""
function quantized_matmul_cpu!(out::AbstractMatrix, x::AbstractMatrix,
                               packed::AbstractMatrix{UInt32},
                               scales::AbstractMatrix,
                               biases::AbstractMatrix;
                               bits::Int=4, group_size::Int=64)
    elems_per_u32 = 32 ÷ bits
    mask = UInt32((1 << bits) - 1)

    O, packed_cols = size(packed)
    I = packed_cols * elems_per_u32
    _, B = size(x)

    for b in 1:B
        for row in 1:O
            acc = 0.0f0
            for pc in 1:packed_cols
                @inbounds packed_val = packed[row, pc]
                for k in 0:elems_per_u32-1
                    col = (pc - 1) * elems_per_u32 + k + 1
                    shift = k * bits
                    quantized = (packed_val >> shift) & mask

                    group_idx = (col - 1) ÷ group_size + 1
                    @inbounds s = Float32(scales[row, group_idx])
                    @inbounds bi = Float32(biases[row, group_idx])
                    w = s * Float32(quantized) + bi

                    @inbounds acc += w * Float32(x[col, b])
                end
            end
            @inbounds out[row, b] = typeof(out[1,1])(acc)
        end
    end
    return out
end

# ── Metal kernel ──

# Strategy for first implementation:
# - One threadgroup per output row (one row of O)
# - Threads in the group cooperate to compute the dot product over I dimension
# - Each thread handles a chunk of the packed columns
# - Simdgroup reduction to sum partial results
#
# For batched input (B > 1), we dispatch (O, B) threadgroups.

function qmatmul_kernel!(out, x, packed, scales, biases,
                          O::Int32, I::Int32, B::Int32,
                          packed_cols::Int32, group_size::Int32)
    # Grid: (O, B)
    row = Int32(threadgroup_position_in_grid().x)  # output row
    b = Int32(threadgroup_position_in_grid().y)     # batch index
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    shared = MtlThreadGroupArray(Float32, 32)

    # Each thread accumulates a partial dot product
    acc = 0.0f0

    pc = tid  # packed column index
    while pc <= packed_cols
        @inbounds packed_val = packed[row, pc]

        # Unpack 8 4-bit values and multiply-accumulate
        col_base = (pc - Int32(1)) * Int32(8)  # 0-indexed base column

        # Determine group index for this set of 8 values
        # Since group_size is typically 64 and we process 8 at a time,
        # all 8 values may fall in the same group
        group_idx = col_base ÷ group_size + Int32(1)
        @inbounds s = Float32(scales[row, group_idx])
        @inbounds bi = Float32(biases[row, group_idx])

        # Unpack and accumulate all 8 values
        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)  # 1-indexed
            quantized = (packed_val >> (UInt32(k) * UInt32(4))) & UInt32(0xF)
            w = s * Float32(quantized) + bi
            @inbounds acc += w * Float32(x[col, b])
            k += Int32(1)
        end

        pc += tg_size
    end

    # Simdgroup reduction
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        acc += simd_shuffle_down(acc, offset)
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared[wid] = acc
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if wid == UInt32(1)
        acc = if lane <= nwarps
            @inbounds shared[lane]
        else
            0.0f0
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            acc += simd_shuffle_down(acc, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds out[row, b] = typeof(out[1,1])(acc)
        end
    end

    return nothing
end

"""
    metal_quantized_matmul!(out, x, packed_weights, scales, biases; group_size=64)

Compute out = dequant(W) @ x where W is 4-bit quantized.
x: (I, B) Float16, out: (O, B) Float16.
"""
function metal_quantized_matmul!(out, x, packed, scales, biases;
                                  group_size::Int=64)
    O, packed_cols = size(packed)
    I = packed_cols * 8  # 4-bit: 8 values per uint32
    _, B = size(x)

    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    @metal threads=tg_size groups=(O, B) qmatmul_kernel!(
        out, x, packed, scales, biases,
        Int32(O), Int32(I), Int32(B), Int32(packed_cols), Int32(group_size))
    return out
end
