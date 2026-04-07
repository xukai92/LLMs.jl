"""
Vectorized quantized matmul using LLVM vector types.

Uses NTuple{4, VecElement{Float32}} to hint the compiler to emit
vectorized Metal IR for the dequant + multiply-accumulate inner loop.

Processes 2 packed words (16 values) per iteration, vectorized as
4 groups of 4 values each.
"""

const F32x4 = NTuple{4, Core.VecElement{Float32}}

@inline function _splat4(v::Float32)::F32x4
    (Core.VecElement(v), Core.VecElement(v), Core.VecElement(v), Core.VecElement(v))
end

@inline function _unpack_lo4(pv::UInt32)::F32x4
    (Core.VecElement(Float32(pv & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(4)) & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(8)) & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(12)) & UInt32(0xF))))
end

@inline function _unpack_hi4(pv::UInt32)::F32x4
    (Core.VecElement(Float32((pv >> UInt32(16)) & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(20)) & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(24)) & UInt32(0xF))),
     Core.VecElement(Float32((pv >> UInt32(28)) & UInt32(0xF))))
end

@inline function _fma4(a::F32x4, b::F32x4, c::F32x4)::F32x4
    (Core.VecElement(muladd(a[1].value, b[1].value, c[1].value)),
     Core.VecElement(muladd(a[2].value, b[2].value, c[2].value)),
     Core.VecElement(muladd(a[3].value, b[3].value, c[3].value)),
     Core.VecElement(muladd(a[4].value, b[4].value, c[4].value)))
end

@inline function _mul4(a::F32x4, b::F32x4)::F32x4
    (Core.VecElement(a[1].value * b[1].value),
     Core.VecElement(a[2].value * b[2].value),
     Core.VecElement(a[3].value * b[3].value),
     Core.VecElement(a[4].value * b[4].value))
end

@inline function _add4(a::F32x4, b::F32x4)::F32x4
    (Core.VecElement(a[1].value + b[1].value),
     Core.VecElement(a[2].value + b[2].value),
     Core.VecElement(a[3].value + b[3].value),
     Core.VecElement(a[4].value + b[4].value))
end

@inline function _hsum4(v::F32x4)::Float32
    v[1].value + v[2].value + v[3].value + v[4].value
end

@inline function _load4(x, col::Int32, b::Int32)::F32x4
    (Core.VecElement(Float32(@inbounds x[col, b])),
     Core.VecElement(Float32(@inbounds x[col + Int32(1), b])),
     Core.VecElement(Float32(@inbounds x[col + Int32(2), b])),
     Core.VecElement(Float32(@inbounds x[col + Int32(3), b])))
end

function qmm_vec_kernel!(out, x, packed, scales, biases, B::Int32)
    row = Int32(threadgroup_position_in_grid().x)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()
    packed_cols = Int32(size(packed, 1))
    shared = MtlThreadGroupArray(Float32, 32)

    CHUNK = Int32(8)
    bchunk = Int32(0)

    while bchunk < B
        b_count = min(CHUNK, B - bchunk)
        b1=bchunk+Int32(1); b2=bchunk+Int32(2); b3=bchunk+Int32(3); b4=bchunk+Int32(4)
        b5=bchunk+Int32(5); b6=bchunk+Int32(6); b7=bchunk+Int32(7); b8=bchunk+Int32(8)

        a1=0f0; a2=0f0; a3=0f0; a4=0f0; a5=0f0; a6=0f0; a7=0f0; a8=0f0

        pc = tid
        while pc <= packed_cols
            @inbounds pv = packed[pc, row]
            col_base = (pc - Int32(1)) << Int32(3)
            grp = (col_base >> Int32(6)) + Int32(1)
            @inbounds s = Float32(scales[grp, row])
            @inbounds bi = Float32(biases[grp, row])

            # Vectorized unpack: 8 values as 2 groups of 4
            s4 = _splat4(s)
            b4 = _splat4(bi)
            w_lo = _fma4(_unpack_lo4(pv), s4, b4)  # scale * val + bias for first 4
            w_hi = _fma4(_unpack_hi4(pv), s4, b4)   # scale * val + bias for last 4

            col = col_base + Int32(1)

            # For each batch element: vectorized dot product (4+4 = 8 values)
            if b_count >= Int32(1)
                x_lo = _load4(x, col, b1)
                x_hi = _load4(x, col + Int32(4), b1)
                a1 += _hsum4(_mul4(w_lo, x_lo)) + _hsum4(_mul4(w_hi, x_hi))
            end
            if b_count >= Int32(2)
                x_lo = _load4(x, col, b2)
                x_hi = _load4(x, col + Int32(4), b2)
                a2 += _hsum4(_mul4(w_lo, x_lo)) + _hsum4(_mul4(w_hi, x_hi))
            end
            if b_count >= Int32(3)
                x_lo = _load4(x, col, b3)
                x_hi = _load4(x, col + Int32(4), b3)
                a3 += _hsum4(_mul4(w_lo, x_lo)) + _hsum4(_mul4(w_hi, x_hi))
            end
            if b_count >= Int32(4)
                x_lo = _load4(x, col, b4)
                x_hi = _load4(x, col + Int32(4), b4)
                a4 += _hsum4(_mul4(w_lo, x_lo)) + _hsum4(_mul4(w_hi, x_hi))
            end
            if b_count >= Int32(5)
                a5 += _hsum4(_mul4(w_lo, _load4(x,col,b5))) + _hsum4(_mul4(w_hi, _load4(x,col+Int32(4),b5)))
            end
            if b_count >= Int32(6)
                a6 += _hsum4(_mul4(w_lo, _load4(x,col,b6))) + _hsum4(_mul4(w_hi, _load4(x,col+Int32(4),b6)))
            end
            if b_count >= Int32(7)
                a7 += _hsum4(_mul4(w_lo, _load4(x,col,b7))) + _hsum4(_mul4(w_hi, _load4(x,col+Int32(4),b7)))
            end
            if b_count >= Int32(8)
                a8 += _hsum4(_mul4(w_lo, _load4(x,col,b8))) + _hsum4(_mul4(w_hi, _load4(x,col+Int32(4),b8)))
            end

            pc += tg_size
        end

        # Reduce
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

"""Vectorized quantized matmul."""
function metal_qmatmul_vec!(out, x, packed, scales, biases; group_size::Int=64)
    packed_cols = size(packed, 1); O = size(packed, 2); B = size(x, 2)
    tg = min(packed_cols, 256)
    tg = max(tg - (tg % 32), 32)

    if B <= 1
        @metal threads=tg groups=(O, 1) qmatmul_kernel_g64!(out, x, packed, scales, biases)
    else
        @metal threads=tg groups=(O, 1) qmm_vec_kernel!(out, x, packed, scales, biases, Int32(B))
    end
    return out
end
