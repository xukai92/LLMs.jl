"""
Quantized matmul using simdgroup 8×8 matrix ops with 4-bit dequantization.

out[O, B] = dequant(packed[packed_cols, O]) @ x[I, B]

Key optimizations:
- Full packed-word unpack: 1 UInt32 → 8 dequanted Float32 (scale/bias amortized 8×)
- K×8 unrolling: 64 K values per barrier pair
- Vec2 Float16 loads for x tiles
- Unified loader: W dequant (32 threads) and x load (128 threads) in single call
"""

const _QF16x2 = NTuple{2, VecElement{Float16}}

function qmatmul_sg_kernel!(out, x, packed, scales, biases,
                             O::Int32, I::Int32, B::Int32, packed_cols::Int32)
    tile_o = Int32(threadgroup_position_in_grid().x)
    tile_b = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(4); sg_n = (sg - Int32(1)) ÷ Int32(4)

    w1 = MtlThreadGroupArray(Float32, (32, 8)); x1 = MtlThreadGroupArray(Float32, (8, 32))
    w2 = MtlThreadGroupArray(Float32, (32, 8)); x2 = MtlThreadGroupArray(Float32, (8, 32))
    w3 = MtlThreadGroupArray(Float32, (32, 8)); x3 = MtlThreadGroupArray(Float32, (8, 32))
    w4 = MtlThreadGroupArray(Float32, (32, 8)); x4 = MtlThreadGroupArray(Float32, (8, 32))
    w5 = MtlThreadGroupArray(Float32, (32, 8)); x5 = MtlThreadGroupArray(Float32, (8, 32))
    w6 = MtlThreadGroupArray(Float32, (32, 8)); x6 = MtlThreadGroupArray(Float32, (8, 32))
    w7 = MtlThreadGroupArray(Float32, (32, 8)); x7 = MtlThreadGroupArray(Float32, (8, 32))
    w8 = MtlThreadGroupArray(Float32, (32, 8)); x8 = MtlThreadGroupArray(Float32, (8, 32))
    zt = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (32, 32))

    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zt[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zt, (1, 1))

    # Unified loader: W (threads 1-32) and x (threads 33-160) execute simultaneously
    @inline function _load_wx(w_dst, x_dst, k_base)
        if gtid <= Int32(32)
            # W: 1 thread per row, full packed-word unpack
            row_in_tile = gtid
            o_row = (tile_o - Int32(1)) * Int32(32) + row_in_tile
            pc_idx = (k_base >> Int32(3)) + Int32(1)
            grp = (k_base >> Int32(6)) + Int32(1)
            @inbounds pv = packed[pc_idx, o_row]
            @inbounds s = Float32(scales[grp, o_row])
            @inbounds bi = Float32(biases[grp, o_row])
            @inbounds w_dst[row_in_tile, 1] = s * Float32((pv) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 2] = s * Float32((pv >> UInt32(4)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 3] = s * Float32((pv >> UInt32(8)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 4] = s * Float32((pv >> UInt32(12)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 5] = s * Float32((pv >> UInt32(16)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 6] = s * Float32((pv >> UInt32(20)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 7] = s * Float32((pv >> UInt32(24)) & UInt32(0xF)) + bi
            @inbounds w_dst[row_in_tile, 8] = s * Float32((pv >> UInt32(28)) & UInt32(0xF)) + bi
        elseif gtid <= Int32(160)
            # x: load 2 elements per thread
            f = gtid - Int32(33); pair = f % Int32(4); c = (f ÷ Int32(4)) + Int32(1)
            r1 = pair * Int32(2) + Int32(1)
            gk = k_base + r1; gn = (tile_b - Int32(1)) * Int32(32) + c
            @inbounds x_dst[r1, c] = Float32(x[gk, gn])
            @inbounds x_dst[r1 + Int32(1), c] = Float32(x[gk + Int32(1), gn])
        end
    end

    wo = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    xo = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(64) <= I
        _load_wx(w1, x1, k); _load_wx(w2, x2, k + Int32(8))
        _load_wx(w3, x3, k + Int32(16)); _load_wx(w4, x4, k + Int32(24))
        _load_wx(w5, x5, k + Int32(32)); _load_wx(w6, x6, k + Int32(40))
        _load_wx(w7, x7, k + Int32(48)); _load_wx(w8, x8, k + Int32(56))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, wo), simdgroup_load(x2, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w3, wo), simdgroup_load(x3, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w4, wo), simdgroup_load(x4, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w5, wo), simdgroup_load(x5, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w6, wo), simdgroup_load(x6, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w7, wo), simdgroup_load(x7, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w8, wo), simdgroup_load(x8, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(64)
    end
    while k + Int32(8) <= I
        _load_wx(w1, x1, k)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end

    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(1024)
            r = ((idx - Int32(1)) % Int32(32)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(32)) + Int32(1)
            go = (tile_o - Int32(1)) * Int32(32) + r; gb = (tile_b - Int32(1)) * Int32(32) + c
            if go <= O && gb <= B
                @inbounds out[go, gb] = Float16(res[r, c])
            end
        end
    end; return nothing
end

"""Quantized matmul with simdgroup tiling. out[O, B] = dequant(packed) @ x[I, B]"""
function metal_qmatmul_sg!(out, x, packed, scales, biases; group_size::Int=64)
    packed_cols = Int32(size(packed, 1))
    O = Int32(size(packed, 2))
    I = Int32(packed_cols * Int32(8))
    B = Int32(size(x, 2))
    @metal threads=512 groups=(cld(Int(O), 32), cld(Int(B), 32)) qmatmul_sg_kernel!(
        out, x, packed, scales, biases, O, I, B, packed_cols)
    return out
end
