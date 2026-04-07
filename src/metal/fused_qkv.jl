"""
Fused Q+K+V projection: three quantized matmuls sharing the same input.

Reads the normed input once, computes all three projections, writes to
three separate output buffers. Saves 2 dispatches per layer.
"""

function fused_qkv_kernel!(q_out, k_out, v_out, x,
                            q_packed, q_scales, q_biases,
                            k_packed, k_scales, k_biases,
                            v_packed, v_scales, v_biases)
    # Grid: (max_O, 1) where max_O = max(q_O, k_O, v_O)
    # Actually we need separate rows for each projection. Use y-dim:
    # y=1: q_proj, y=2: k_proj, y=3: v_proj
    proj = Int32(threadgroup_position_in_grid().y)
    row = Int32(threadgroup_position_in_grid().x)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    shared = MtlThreadGroupArray(Float32, 32)

    # Select which projection
    if proj == Int32(1)
        p = q_packed; s = q_scales; bi = q_biases; out = q_out
        packed_cols = Int32(size(q_packed, 1))
        O = Int32(size(q_packed, 2))
    elseif proj == Int32(2)
        p = k_packed; s = k_scales; bi = k_biases; out = k_out
        packed_cols = Int32(size(k_packed, 1))
        O = Int32(size(k_packed, 2))
    else
        p = v_packed; s = v_scales; bi = v_biases; out = v_out
        packed_cols = Int32(size(v_packed, 1))
        O = Int32(size(v_packed, 2))
    end

    if row > O
        return nothing
    end

    acc = 0.0f0
    pc = tid
    while pc <= packed_cols
        @inbounds pv = p[pc, row]
        col_base = (pc - Int32(1)) << Int32(3)
        grp = (col_base >> Int32(6)) + Int32(1)
        @inbounds sc = Float32(s[grp, row])
        @inbounds bias = Float32(bi[grp, row])

        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            w = sc * Float32((pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)) + bias
            @inbounds acc += w * Float32(x[col, 1])
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
        acc = lane <= nwarps ? (@inbounds shared[lane]) : 0.0f0
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            acc += simd_shuffle_down(acc, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds out[row, 1] = typeof(out[1,1])(acc)
        end
    end

    return nothing
end

"""
Fused Q+K+V projection for B=1. One dispatch instead of three.
"""
function metal_fused_qkv!(q_out, k_out, v_out, x, q_layer, k_layer, v_layer)
    max_O = max(q_layer.out_features, k_layer.out_features, v_layer.out_features)
    packed_cols = size(q_layer.weight, 1)
    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    @metal threads=tg_size groups=(max_O, 3) fused_qkv_kernel!(
        q_out, k_out, v_out, x,
        q_layer.weight, q_layer.scales, q_layer.biases,
        k_layer.weight, k_layer.scales, k_layer.biases,
        v_layer.weight, v_layer.scales, v_layer.biases)
    return nothing
end
