"""
Quantized matmul using simdgroup 8×8 matrix ops with 4-bit dequantization.

out[O, B] = dequant(packed[packed_cols, O]) @ x[I, B]

Same tiling strategy as fp16_matmul_4x4_k8_ptr! but with inline dequantization
of 4-bit packed weights into Float32 shared memory tiles.

Weight format (MLX 4-bit):
- packed[packed_cols, O]: each UInt32 holds 8 consecutive 4-bit values
- scales[n_groups, O], biases[n_groups, O]: per-group dequant params
- group_size=64: 64 input elements share one scale/bias pair
- Dequant: w_float = scale * float((packed >> (k*4)) & 0xF) + bias
"""

const _QF16x2 = NTuple{2, VecElement{Float16}}

# 4×4 simdgroup grid, K×8 unrolling, dequant W tiles into shmem
function qmatmul_sg_kernel!(out, x, packed, scales, biases,
                             O::Int32, I::Int32, B::Int32, packed_cols::Int32)
    tile_o_grp = Int32(threadgroup_position_in_grid().x)  # output dim tile
    tile_b_grp = Int32(threadgroup_position_in_grid().y)  # batch dim tile
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(4); sg_n = (sg - Int32(1)) ÷ Int32(4)

    # Shared memory tiles: W dequantized (32×8) and x (8×32)
    w1 = MtlThreadGroupArray(Float32, (32, 8)); x1 = MtlThreadGroupArray(Float32, (8, 32))
    w2 = MtlThreadGroupArray(Float32, (32, 8)); x2 = MtlThreadGroupArray(Float32, (8, 32))
    w3 = MtlThreadGroupArray(Float32, (32, 8)); x3 = MtlThreadGroupArray(Float32, (8, 32))
    w4 = MtlThreadGroupArray(Float32, (32, 8)); x4 = MtlThreadGroupArray(Float32, (8, 32))
    zt = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (32, 32))

    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zt[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zt, (1, 1))

    # W tile loader: dequantize packed 4-bit weights into Float32 shmem
    # W tile is 32 rows (O dim) × 8 cols (I dim) = 256 elements
    # Each packed UInt32 holds 8 values along I dim, so we need 1 packed word per row per col-chunk
    # 32 rows × 1 packed word = 32 loads, each unpacked to 8 values
    # With 256 threads available (gtid 1-256), each thread handles ~1 row
    @inline function _lw_q(dst, k_base)
        if gtid <= Int32(256)
            f = gtid - Int32(1); row_in_tile = (f % Int32(32)) + Int32(1); col_chunk = (f ÷ Int32(32))
            # Global output row
            o_row = (tile_o_grp - Int32(1)) * Int32(32) + row_in_tile
            # Which packed column? k_base is the K offset (0-indexed), col_chunk is 0-7
            # Each packed word covers 8 I values, so packed_col = (k_base + col_chunk*8) ÷ 8 + 1... no
            # Wait: k_base is the start of this 8-wide K chunk. We load 8 K columns.
            # col_chunk goes 0..7, each is one K column
            # The packed word for K column (k_base + col_chunk) is:
            #   packed_col_idx = (k_base + col_chunk) ÷ 8 + 1
            #   bit_idx = (k_base + col_chunk) % 8
            k_col = k_base + col_chunk
            pc_idx = (k_col >> Int32(3)) + Int32(1)  # packed column index (1-based)
            bit_idx = k_col & Int32(7)               # bit position within packed word

            @inbounds pv = packed[pc_idx, o_row]
            grp = (k_col >> Int32(6)) + Int32(1)     # group index (group_size=64)
            @inbounds s = Float32(scales[grp, o_row])
            @inbounds bi = Float32(biases[grp, o_row])
            w_val = s * Float32((pv >> (UInt32(bit_idx) << UInt32(2))) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, col_chunk + Int32(1)] = w_val
        end
    end

    # x tile loader: same as FP16 kernel, vec2 Float16 loads
    @inline function _lx(dst, k_base)
        if gtid > Int32(256) && gtid <= Int32(384)
            f = gtid - Int32(257); pair = f % Int32(4); c = (f ÷ Int32(4)) + Int32(1)
            r1 = pair * Int32(2) + Int32(1)
            gk = k_base + r1; gn = (tile_b_grp - Int32(1)) * Int32(32) + c
            p = pointer(x) + (Int64(gn - Int32(1)) * Int64(I) + Int64(gk - Int32(1))) * Int64(2)
            vec = unsafe_load(reinterpret(Core.LLVMPtr{_QF16x2, Metal.AS.Device}, p))
            @inbounds dst[r1, c] = Float32(vec[1].value)
            @inbounds dst[r1 + Int32(1), c] = Float32(vec[2].value)
        end
    end

    wo = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    xo = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(32) <= I
        _lw_q(w1, k); _lx(x1, k)
        _lw_q(w2, k + Int32(8)); _lx(x2, k + Int32(8))
        _lw_q(w3, k + Int32(16)); _lx(x3, k + Int32(16))
        _lw_q(w4, k + Int32(24)); _lx(x4, k + Int32(24))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, wo), simdgroup_load(x2, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w3, wo), simdgroup_load(x3, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w4, wo), simdgroup_load(x4, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(32)
    end
    while k + Int32(8) <= I
        _lw_q(w1, k); _lx(x1, k)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end

    # Store results
    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(1024)
            r = ((idx - Int32(1)) % Int32(32)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(32)) + Int32(1)
            go = (tile_o_grp - Int32(1)) * Int32(32) + r; gb = (tile_b_grp - Int32(1)) * Int32(32) + c
            if go <= O && gb <= B
                @inbounds out[go, gb] = Float16(res[r, c])
            end
        end
    end; return nothing
end

"""
Quantized matmul with simdgroup tiling.
out[O, B] = dequant(packed) @ x[I, B]

Uses 4×4 simdgroup grid with inline dequantization of 4-bit packed weights.
Requires B padded to multiple of 32 for optimal performance.
"""
function metal_qmatmul_sg!(out, x, packed, scales, biases; group_size::Int=64)
    packed_cols = Int32(size(packed, 1))
    O = Int32(size(packed, 2))
    I = Int32(packed_cols * Int32(8))
    B = Int32(size(x, 2))
    @metal threads=512 groups=(cld(Int(O), 32), cld(Int(B), 32)) qmatmul_sg_kernel!(
        out, x, packed, scales, biases, O, I, B, packed_cols)
    return out
end
