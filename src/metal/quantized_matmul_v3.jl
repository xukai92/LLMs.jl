"""
Quantized matmul v3: Unrolled inner loop for better ILP.

Key change from v2: unroll the 8-value-per-packed-word loop and process
2 packed words per iteration (16 values). This gives the GPU more
independent operations to schedule in parallel across ALU pipelines.

Also unrolls the batch accumulation to reduce branch overhead.
"""

# Unrolled kernel: processes 2 packed words per outer loop iteration
function qmm_v3_kernel!(out, x, packed, scales, biases, B::Int32)
    row = Int32(threadgroup_position_in_grid().x)
    btile = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(packed, 2))

    TILE_N = Int32(8)
    b_base = (btile - Int32(1)) * TILE_N
    b_count = min(TILE_N, B - b_base)
    b1=b_base+Int32(1); b2=b_base+Int32(2); b3=b_base+Int32(3); b4=b_base+Int32(4)
    b5=b_base+Int32(5); b6=b_base+Int32(6); b7=b_base+Int32(7); b8=b_base+Int32(8)

    shared = MtlThreadGroupArray(Float32, 32)

    a1=0f0; a2=0f0; a3=0f0; a4=0f0; a5=0f0; a6=0f0; a7=0f0; a8=0f0

    # Process 2 packed words (16 values) per iteration for better ILP
    pc = tid * Int32(2) - Int32(1)  # start at odd index, step by tg_size*2
    while pc <= packed_cols - Int32(1)
        # Load 2 packed words and their shared group info
        @inbounds pv0 = packed[row, pc]
        @inbounds pv1 = packed[row, pc + Int32(1)]
        col_base0 = (pc - Int32(1)) << Int32(3)
        col_base1 = pc << Int32(3)
        grp0 = (col_base0 >> Int32(6)) + Int32(1)
        grp1 = (col_base1 >> Int32(6)) + Int32(1)
        @inbounds s0 = Float32(scales[row, grp0])
        @inbounds bi0 = Float32(biases[row, grp0])
        @inbounds s1 = Float32(scales[row, grp1])
        @inbounds bi1 = Float32(biases[row, grp1])

        # Unrolled: first packed word (8 values)
        col = col_base0 + Int32(1)
        w0 = s0 * Float32(pv0 & UInt32(0xF)) + bi0
        w1 = s0 * Float32((pv0 >> UInt32(4)) & UInt32(0xF)) + bi0
        w2 = s0 * Float32((pv0 >> UInt32(8)) & UInt32(0xF)) + bi0
        w3 = s0 * Float32((pv0 >> UInt32(12)) & UInt32(0xF)) + bi0
        w4 = s0 * Float32((pv0 >> UInt32(16)) & UInt32(0xF)) + bi0
        w5 = s0 * Float32((pv0 >> UInt32(20)) & UInt32(0xF)) + bi0
        w6 = s0 * Float32((pv0 >> UInt32(24)) & UInt32(0xF)) + bi0
        w7 = s0 * Float32((pv0 >> UInt32(28)) & UInt32(0xF)) + bi0

        if b_count >= Int32(1)
            @inbounds x0=Float32(x[col,b1]); @inbounds x1=Float32(x[col+Int32(1),b1])
            @inbounds x2=Float32(x[col+Int32(2),b1]); @inbounds x3=Float32(x[col+Int32(3),b1])
            @inbounds x4=Float32(x[col+Int32(4),b1]); @inbounds x5=Float32(x[col+Int32(5),b1])
            @inbounds x6=Float32(x[col+Int32(6),b1]); @inbounds x7=Float32(x[col+Int32(7),b1])
            a1 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(2)
            @inbounds x0=Float32(x[col,b2]); @inbounds x1=Float32(x[col+Int32(1),b2])
            @inbounds x2=Float32(x[col+Int32(2),b2]); @inbounds x3=Float32(x[col+Int32(3),b2])
            @inbounds x4=Float32(x[col+Int32(4),b2]); @inbounds x5=Float32(x[col+Int32(5),b2])
            @inbounds x6=Float32(x[col+Int32(6),b2]); @inbounds x7=Float32(x[col+Int32(7),b2])
            a2 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(3)
            @inbounds x0=Float32(x[col,b3]); @inbounds x1=Float32(x[col+Int32(1),b3])
            @inbounds x2=Float32(x[col+Int32(2),b3]); @inbounds x3=Float32(x[col+Int32(3),b3])
            @inbounds x4=Float32(x[col+Int32(4),b3]); @inbounds x5=Float32(x[col+Int32(5),b3])
            @inbounds x6=Float32(x[col+Int32(6),b3]); @inbounds x7=Float32(x[col+Int32(7),b3])
            a3 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(4)
            @inbounds x0=Float32(x[col,b4]); @inbounds x1=Float32(x[col+Int32(1),b4])
            @inbounds x2=Float32(x[col+Int32(2),b4]); @inbounds x3=Float32(x[col+Int32(3),b4])
            @inbounds x4=Float32(x[col+Int32(4),b4]); @inbounds x5=Float32(x[col+Int32(5),b4])
            @inbounds x6=Float32(x[col+Int32(6),b4]); @inbounds x7=Float32(x[col+Int32(7),b4])
            a4 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(5)
            @inbounds x0=Float32(x[col,b5]); @inbounds x1=Float32(x[col+Int32(1),b5])
            @inbounds x2=Float32(x[col+Int32(2),b5]); @inbounds x3=Float32(x[col+Int32(3),b5])
            @inbounds x4=Float32(x[col+Int32(4),b5]); @inbounds x5=Float32(x[col+Int32(5),b5])
            @inbounds x6=Float32(x[col+Int32(6),b5]); @inbounds x7=Float32(x[col+Int32(7),b5])
            a5 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(6)
            @inbounds x0=Float32(x[col,b6]); @inbounds x1=Float32(x[col+Int32(1),b6])
            @inbounds x2=Float32(x[col+Int32(2),b6]); @inbounds x3=Float32(x[col+Int32(3),b6])
            @inbounds x4=Float32(x[col+Int32(4),b6]); @inbounds x5=Float32(x[col+Int32(5),b6])
            @inbounds x6=Float32(x[col+Int32(6),b6]); @inbounds x7=Float32(x[col+Int32(7),b6])
            a6 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(7)
            @inbounds x0=Float32(x[col,b7]); @inbounds x1=Float32(x[col+Int32(1),b7])
            @inbounds x2=Float32(x[col+Int32(2),b7]); @inbounds x3=Float32(x[col+Int32(3),b7])
            @inbounds x4=Float32(x[col+Int32(4),b7]); @inbounds x5=Float32(x[col+Int32(5),b7])
            @inbounds x6=Float32(x[col+Int32(6),b7]); @inbounds x7=Float32(x[col+Int32(7),b7])
            a7 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(8)
            @inbounds x0=Float32(x[col,b8]); @inbounds x1=Float32(x[col+Int32(1),b8])
            @inbounds x2=Float32(x[col+Int32(2),b8]); @inbounds x3=Float32(x[col+Int32(3),b8])
            @inbounds x4=Float32(x[col+Int32(4),b8]); @inbounds x5=Float32(x[col+Int32(5),b8])
            @inbounds x6=Float32(x[col+Int32(6),b8]); @inbounds x7=Float32(x[col+Int32(7),b8])
            a8 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end

        # Second packed word
        col1 = col_base1 + Int32(1)
        w0 = s1 * Float32(pv1 & UInt32(0xF)) + bi1
        w1 = s1 * Float32((pv1 >> UInt32(4)) & UInt32(0xF)) + bi1
        w2 = s1 * Float32((pv1 >> UInt32(8)) & UInt32(0xF)) + bi1
        w3 = s1 * Float32((pv1 >> UInt32(12)) & UInt32(0xF)) + bi1
        w4 = s1 * Float32((pv1 >> UInt32(16)) & UInt32(0xF)) + bi1
        w5 = s1 * Float32((pv1 >> UInt32(20)) & UInt32(0xF)) + bi1
        w6 = s1 * Float32((pv1 >> UInt32(24)) & UInt32(0xF)) + bi1
        w7 = s1 * Float32((pv1 >> UInt32(28)) & UInt32(0xF)) + bi1

        if b_count >= Int32(1)
            @inbounds x0=Float32(x[col1,b1]); @inbounds x1=Float32(x[col1+Int32(1),b1])
            @inbounds x2=Float32(x[col1+Int32(2),b1]); @inbounds x3=Float32(x[col1+Int32(3),b1])
            @inbounds x4=Float32(x[col1+Int32(4),b1]); @inbounds x5=Float32(x[col1+Int32(5),b1])
            @inbounds x6=Float32(x[col1+Int32(6),b1]); @inbounds x7=Float32(x[col1+Int32(7),b1])
            a1 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        if b_count >= Int32(2)
            @inbounds x0=Float32(x[col1,b2]); @inbounds x1=Float32(x[col1+Int32(1),b2])
            @inbounds x2=Float32(x[col1+Int32(2),b2]); @inbounds x3=Float32(x[col1+Int32(3),b2])
            @inbounds x4=Float32(x[col1+Int32(4),b2]); @inbounds x5=Float32(x[col1+Int32(5),b2])
            @inbounds x6=Float32(x[col1+Int32(6),b2]); @inbounds x7=Float32(x[col1+Int32(7),b2])
            a2 += w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7
        end
        # ... batch elements 3-8 follow the same pattern
        if b_count >= Int32(3)
            @inbounds a3 += w0*Float32(x[col1,b3]) + w1*Float32(x[col1+Int32(1),b3]) + w2*Float32(x[col1+Int32(2),b3]) + w3*Float32(x[col1+Int32(3),b3]) + w4*Float32(x[col1+Int32(4),b3]) + w5*Float32(x[col1+Int32(5),b3]) + w6*Float32(x[col1+Int32(6),b3]) + w7*Float32(x[col1+Int32(7),b3])
        end
        if b_count >= Int32(4)
            @inbounds a4 += w0*Float32(x[col1,b4]) + w1*Float32(x[col1+Int32(1),b4]) + w2*Float32(x[col1+Int32(2),b4]) + w3*Float32(x[col1+Int32(3),b4]) + w4*Float32(x[col1+Int32(4),b4]) + w5*Float32(x[col1+Int32(5),b4]) + w6*Float32(x[col1+Int32(6),b4]) + w7*Float32(x[col1+Int32(7),b4])
        end
        if b_count >= Int32(5)
            @inbounds a5 += w0*Float32(x[col1,b5]) + w1*Float32(x[col1+Int32(1),b5]) + w2*Float32(x[col1+Int32(2),b5]) + w3*Float32(x[col1+Int32(3),b5]) + w4*Float32(x[col1+Int32(4),b5]) + w5*Float32(x[col1+Int32(5),b5]) + w6*Float32(x[col1+Int32(6),b5]) + w7*Float32(x[col1+Int32(7),b5])
        end
        if b_count >= Int32(6)
            @inbounds a6 += w0*Float32(x[col1,b6]) + w1*Float32(x[col1+Int32(1),b6]) + w2*Float32(x[col1+Int32(2),b6]) + w3*Float32(x[col1+Int32(3),b6]) + w4*Float32(x[col1+Int32(4),b6]) + w5*Float32(x[col1+Int32(5),b6]) + w6*Float32(x[col1+Int32(6),b6]) + w7*Float32(x[col1+Int32(7),b6])
        end
        if b_count >= Int32(7)
            @inbounds a7 += w0*Float32(x[col1,b7]) + w1*Float32(x[col1+Int32(1),b7]) + w2*Float32(x[col1+Int32(2),b7]) + w3*Float32(x[col1+Int32(3),b7]) + w4*Float32(x[col1+Int32(4),b7]) + w5*Float32(x[col1+Int32(5),b7]) + w6*Float32(x[col1+Int32(6),b7]) + w7*Float32(x[col1+Int32(7),b7])
        end
        if b_count >= Int32(8)
            @inbounds a8 += w0*Float32(x[col1,b8]) + w1*Float32(x[col1+Int32(1),b8]) + w2*Float32(x[col1+Int32(2),b8]) + w3*Float32(x[col1+Int32(3),b8]) + w4*Float32(x[col1+Int32(4),b8]) + w5*Float32(x[col1+Int32(5),b8]) + w6*Float32(x[col1+Int32(6),b8]) + w7*Float32(x[col1+Int32(7),b8])
        end

        pc += tg_size * Int32(2)
    end

    # Handle leftover (odd packed_cols)
    pc_leftover = tid + (packed_cols ÷ Int32(2)) * Int32(2) - tg_size * Int32(2) + tg_size * Int32(2)
    # Actually just handle remaining with v2-style loop
    pc2 = (packed_cols ÷ Int32(2)) * Int32(2) + tid
    if pc2 <= packed_cols && packed_cols % Int32(2) != Int32(0)
        @inbounds pv = packed[row, pc2]
        col_base = (pc2 - Int32(1)) << Int32(3)
        grp = (col_base >> Int32(6)) + Int32(1)
        @inbounds s = Float32(scales[row, grp])
        @inbounds bi = Float32(biases[row, grp])
        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            w = s * Float32((pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)) + bi
            if b_count>=Int32(1); @inbounds a1 += w*Float32(x[col,b1]); end
            if b_count>=Int32(2); @inbounds a2 += w*Float32(x[col,b2]); end
            if b_count>=Int32(3); @inbounds a3 += w*Float32(x[col,b3]); end
            if b_count>=Int32(4); @inbounds a4 += w*Float32(x[col,b4]); end
            if b_count>=Int32(5); @inbounds a5 += w*Float32(x[col,b5]); end
            if b_count>=Int32(6); @inbounds a6 += w*Float32(x[col,b6]); end
            if b_count>=Int32(7); @inbounds a7 += w*Float32(x[col,b7]); end
            if b_count>=Int32(8); @inbounds a8 += w*Float32(x[col,b8]); end
            k += Int32(1)
        end
    end

    # ── Reduce and write ──
    @inline function _rw(acc, bidx)
        local a = acc
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            a += simd_shuffle_down(a, offset); offset <<= 1
        end
        if lane == UInt32(1); @inbounds shared[wid] = a; end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if wid == UInt32(1)
            a = lane <= nwarps ? (@inbounds shared[lane]) : 0.0f0
            offset = UInt32(1)
            while offset < threads_per_simdgroup()
                a += simd_shuffle_down(a, offset); offset <<= 1
            end
            if lane == UInt32(1); @inbounds out[row, bidx] = typeof(out[1,1])(a); end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    end

    if b_count>=Int32(1); _rw(a1,b1); end; if b_count>=Int32(2); _rw(a2,b2); end
    if b_count>=Int32(3); _rw(a3,b3); end; if b_count>=Int32(4); _rw(a4,b4); end
    if b_count>=Int32(5); _rw(a5,b5); end; if b_count>=Int32(6); _rw(a6,b6); end
    if b_count>=Int32(7); _rw(a7,b7); end; if b_count>=Int32(8); _rw(a8,b8); end

    return nothing
end

"""Auto-selecting v3 matmul: unrolled inner loop for B>4."""
function metal_qmatmul_v3!(out, x, packed, scales, biases; group_size::Int=64)
    O = size(packed, 1)
    packed_cols = size(packed, 2)
    B = size(x, 2)
    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    if B <= 1
        @metal threads=tg_size groups=(O, 1) qmatmul_kernel_g64!(out, x, packed, scales, biases)
    elseif B <= 4
        @metal threads=tg_size groups=(O, cld(B, 4)) qmm_v2_kernel!(out, x, packed, scales, biases, Int32(B))
    else
        # Ensure packed_cols is even for the 2-word processing
        tg_v3 = min(packed_cols ÷ 2, 256)
        tg_v3 = max(tg_v3 - (tg_v3 % 32), 32)
        @metal threads=tg_v3 groups=(O, cld(B, 8)) qmm_v3_kernel!(out, x, packed, scales, biases, Int32(B))
    end
    return out
end
