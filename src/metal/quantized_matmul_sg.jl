"""
Quantized matmul using simdgroup 8×8 matrix ops with 4-bit dequantization.

out[O, B] = dequant(packed[packed_cols, O]) @ x[I, B]

Key optimization: one packed UInt32 holds exactly 8 consecutive K values,
which is exactly one column-set of the 32×8 W tile. So each thread reads
one packed word + one scale + one bias, and produces all 8 columns for its row.
This amortizes weight reads 8× vs reading one element at a time.

Weight format (MLX 4-bit):
- packed[packed_cols, O]: each UInt32 holds 8 consecutive 4-bit values
- scales[n_groups, O], biases[n_groups, O]: per-group dequant params
- group_size=64: 64 input elements share one scale/bias pair
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

    # W tile loader: 32 threads, each unpacks one full packed word → 8 Float32 values
    # One packed word = 8 consecutive K values = all 8 columns of one row in the 32×8 tile
    @inline function _lw_q(dst, k_base)
        if gtid <= Int32(32)
            row_in_tile = gtid  # 1-32
            o_row = (tile_o - Int32(1)) * Int32(32) + row_in_tile
            pc_idx = (k_base >> Int32(3)) + Int32(1)  # packed column (k_base is 8-aligned)
            grp = (k_base >> Int32(6)) + Int32(1)
            @inbounds pv = packed[pc_idx, o_row]
            @inbounds s = Float32(scales[grp, o_row])
            @inbounds bi = Float32(biases[grp, o_row])
            # Unpack all 8 values from this packed word
            @inbounds dst[row_in_tile, 1] = s * Float32((pv) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 2] = s * Float32((pv >> UInt32(4)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 3] = s * Float32((pv >> UInt32(8)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 4] = s * Float32((pv >> UInt32(12)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 5] = s * Float32((pv >> UInt32(16)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 6] = s * Float32((pv >> UInt32(20)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 7] = s * Float32((pv >> UInt32(24)) & UInt32(0xF)) + bi
            @inbounds dst[row_in_tile, 8] = s * Float32((pv >> UInt32(28)) & UInt32(0xF)) + bi
        end
    end

    # x tile loader: vec2 Float16 loads, threads 33-160
    @inline function _lx(dst, k_base)
        if gtid > Int32(32) && gtid <= Int32(160)
            f = gtid - Int32(33); pair = f % Int32(4); c = (f ÷ Int32(4)) + Int32(1)
            r1 = pair * Int32(2) + Int32(1)
            gk = k_base + r1; gn = (tile_b - Int32(1)) * Int32(32) + c
            p = pointer(x) + (Int64(gn - Int32(1)) * Int64(I) + Int64(gk - Int32(1))) * Int64(2)
            vec = unsafe_load(reinterpret(Core.LLVMPtr{_QF16x2, Metal.AS.Device}, p))
            @inbounds dst[r1, c] = Float32(vec[1].value)
            @inbounds dst[r1 + Int32(1), c] = Float32(vec[2].value)
        end
    end

    wo = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    xo = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(64) <= I
        _lw_q(w1, k); _lx(x1, k); _lw_q(w2, k+Int32(8)); _lx(x2, k+Int32(8))
        _lw_q(w3, k+Int32(16)); _lx(x3, k+Int32(16)); _lw_q(w4, k+Int32(24)); _lx(x4, k+Int32(24))
        _lw_q(w5, k+Int32(32)); _lx(x5, k+Int32(32)); _lw_q(w6, k+Int32(40)); _lx(x6, k+Int32(40))
        _lw_q(w7, k+Int32(48)); _lx(x7, k+Int32(48)); _lw_q(w8, k+Int32(56)); _lx(x8, k+Int32(56))
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
            go = (tile_o - Int32(1)) * Int32(32) + r; gb = (tile_b - Int32(1)) * Int32(32) + c
            if go <= O && gb <= B
                @inbounds out[go, gb] = Float16(res[r, c])
            end
        end
    end; return nothing
end

"""
Quantized matmul with simdgroup tiling. Auto-selects best kernel by batch size.
out[O, B] = dequant(packed) @ x[I, B]
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
