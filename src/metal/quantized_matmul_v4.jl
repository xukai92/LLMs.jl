"""
Quantized matmul v4: One threadgroup per row, ALL batch elements.
Eliminates the 16x weight read amplification at B=64.

Key insight: weights are read once per threadgroup and reused across
all batch elements. Each thread accumulates into per-batch-element
variables in registers (not shared memory).

We process batch elements in chunks of 8 (matching v2's approach
but with single-pass weight reads).
"""

function qmm_v4_kernel!(out, x, packed, scales, biases, B::Int32)
    row = Int32(threadgroup_position_in_grid().x)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(packed, 1))
    shared = MtlThreadGroupArray(Float32, 32)

    # Process batch in chunks of 8 (register accumulators)
    CHUNK = Int32(8)
    bchunk = Int32(0)

    while bchunk < B
        b_count = min(CHUNK, B - bchunk)
        b1 = bchunk+Int32(1); b2 = bchunk+Int32(2); b3 = bchunk+Int32(3); b4 = bchunk+Int32(4)
        b5 = bchunk+Int32(5); b6 = bchunk+Int32(6); b7 = bchunk+Int32(7); b8 = bchunk+Int32(8)

        a1=0f0; a2=0f0; a3=0f0; a4=0f0; a5=0f0; a6=0f0; a7=0f0; a8=0f0

        # Read each packed weight word ONCE, reuse for all 8 batch elements
        pc = tid
        while pc <= packed_cols
            @inbounds pv = packed[pc, row]
            col_base = (pc - Int32(1)) << Int32(3)
            grp = (col_base >> Int32(6)) + Int32(1)
            @inbounds s = Float32(scales[grp, row])
            @inbounds bi = Float32(biases[grp, row])

            # Unrolled unpack
            w0 = s * Float32(pv & UInt32(0xF)) + bi
            w1 = s * Float32((pv >> UInt32(4)) & UInt32(0xF)) + bi
            w2 = s * Float32((pv >> UInt32(8)) & UInt32(0xF)) + bi
            w3 = s * Float32((pv >> UInt32(12)) & UInt32(0xF)) + bi
            w4 = s * Float32((pv >> UInt32(16)) & UInt32(0xF)) + bi
            w5 = s * Float32((pv >> UInt32(20)) & UInt32(0xF)) + bi
            w6 = s * Float32((pv >> UInt32(24)) & UInt32(0xF)) + bi
            w7 = s * Float32((pv >> UInt32(28)) & UInt32(0xF)) + bi

            col = col_base + Int32(1)
            if b_count >= Int32(1)
                @inbounds a1 += w0*Float32(x[col,b1]) + w1*Float32(x[col+Int32(1),b1]) + w2*Float32(x[col+Int32(2),b1]) + w3*Float32(x[col+Int32(3),b1]) + w4*Float32(x[col+Int32(4),b1]) + w5*Float32(x[col+Int32(5),b1]) + w6*Float32(x[col+Int32(6),b1]) + w7*Float32(x[col+Int32(7),b1])
            end
            if b_count >= Int32(2)
                @inbounds a2 += w0*Float32(x[col,b2]) + w1*Float32(x[col+Int32(1),b2]) + w2*Float32(x[col+Int32(2),b2]) + w3*Float32(x[col+Int32(3),b2]) + w4*Float32(x[col+Int32(4),b2]) + w5*Float32(x[col+Int32(5),b2]) + w6*Float32(x[col+Int32(6),b2]) + w7*Float32(x[col+Int32(7),b2])
            end
            if b_count >= Int32(3)
                @inbounds a3 += w0*Float32(x[col,b3]) + w1*Float32(x[col+Int32(1),b3]) + w2*Float32(x[col+Int32(2),b3]) + w3*Float32(x[col+Int32(3),b3]) + w4*Float32(x[col+Int32(4),b3]) + w5*Float32(x[col+Int32(5),b3]) + w6*Float32(x[col+Int32(6),b3]) + w7*Float32(x[col+Int32(7),b3])
            end
            if b_count >= Int32(4)
                @inbounds a4 += w0*Float32(x[col,b4]) + w1*Float32(x[col+Int32(1),b4]) + w2*Float32(x[col+Int32(2),b4]) + w3*Float32(x[col+Int32(3),b4]) + w4*Float32(x[col+Int32(4),b4]) + w5*Float32(x[col+Int32(5),b4]) + w6*Float32(x[col+Int32(6),b4]) + w7*Float32(x[col+Int32(7),b4])
            end
            if b_count >= Int32(5)
                @inbounds a5 += w0*Float32(x[col,b5]) + w1*Float32(x[col+Int32(1),b5]) + w2*Float32(x[col+Int32(2),b5]) + w3*Float32(x[col+Int32(3),b5]) + w4*Float32(x[col+Int32(4),b5]) + w5*Float32(x[col+Int32(5),b5]) + w6*Float32(x[col+Int32(6),b5]) + w7*Float32(x[col+Int32(7),b5])
            end
            if b_count >= Int32(6)
                @inbounds a6 += w0*Float32(x[col,b6]) + w1*Float32(x[col+Int32(1),b6]) + w2*Float32(x[col+Int32(2),b6]) + w3*Float32(x[col+Int32(3),b6]) + w4*Float32(x[col+Int32(4),b6]) + w5*Float32(x[col+Int32(5),b6]) + w6*Float32(x[col+Int32(6),b6]) + w7*Float32(x[col+Int32(7),b6])
            end
            if b_count >= Int32(7)
                @inbounds a7 += w0*Float32(x[col,b7]) + w1*Float32(x[col+Int32(1),b7]) + w2*Float32(x[col+Int32(2),b7]) + w3*Float32(x[col+Int32(3),b7]) + w4*Float32(x[col+Int32(4),b7]) + w5*Float32(x[col+Int32(5),b7]) + w6*Float32(x[col+Int32(6),b7]) + w7*Float32(x[col+Int32(7),b7])
            end
            if b_count >= Int32(8)
                @inbounds a8 += w0*Float32(x[col,b8]) + w1*Float32(x[col+Int32(1),b8]) + w2*Float32(x[col+Int32(2),b8]) + w3*Float32(x[col+Int32(3),b8]) + w4*Float32(x[col+Int32(4),b8]) + w5*Float32(x[col+Int32(5),b8]) + w6*Float32(x[col+Int32(6),b8]) + w7*Float32(x[col+Int32(7),b8])
            end

            pc += tg_size
        end

        # Reduce each accumulator
        @inline function _rw(a, bidx)
            local v = a
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                v += simd_shuffle_down(v, offset); offset <<= 1
            end
            if lane == UInt32(1); @inbounds shared[wid] = v; end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            if wid == UInt32(1)
                v = lane <= nwarps ? (@inbounds shared[lane]) : 0.0f0
                offset = UInt32(1)
                while offset < threads_per_simdgroup()
                    v += simd_shuffle_down(v, offset); offset <<= 1
                end
                if lane == UInt32(1); @inbounds out[row, bidx] = typeof(out[1,1])(v); end
            end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        end

        if b_count>=Int32(1); _rw(a1,b1); end; if b_count>=Int32(2); _rw(a2,b2); end
        if b_count>=Int32(3); _rw(a3,b3); end; if b_count>=Int32(4); _rw(a4,b4); end
        if b_count>=Int32(5); _rw(a5,b5); end; if b_count>=Int32(6); _rw(a6,b6); end
        if b_count>=Int32(7); _rw(a7,b7); end; if b_count>=Int32(8); _rw(a8,b8); end

        bchunk += CHUNK
    end

    return nothing
end

"""v4: single weight read per row, batch chunked in groups of 8."""
function metal_qmatmul_v4!(out, x, packed, scales, biases; group_size::Int=64)
    packed_cols = size(packed, 1)
    O = size(packed, 2)
    B = size(x, 2)
    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    if B <= 1
        @metal threads=tg_size groups=(O, 1) qmatmul_kernel_g64!(out, x, packed, scales, biases)
    else
        # One threadgroup per output row, processes ALL batch elements in chunks of 8
        @metal threads=tg_size groups=(O, 1) qmm_v4_kernel!(out, x, packed, scales, biases, Int32(B))
    end
    return out
end
