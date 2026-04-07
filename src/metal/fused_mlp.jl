"""
Fused MLP kernel: gate_proj + up_proj + SwiGLU in one dispatch.

Computes: out[i, b] = silu(gate_W[i,:] @ x[:,b]) * (up_W[i,:] @ x[:,b])

Both gate and up projections read the same input x, so we read x once
and compute both dot products simultaneously. Then apply SwiGLU in-place.

This replaces 3 dispatches (gate_proj, up_proj, swiglu) with 1.
"""

# B=1 variant: one threadgroup per output row, each computes gate + up + swiglu
function fused_gate_up_swiglu_kernel!(out, x, gate_packed, gate_scales, gate_biases,
                                       up_packed, up_scales, up_biases)
    row = Int32(threadgroup_position_in_grid().x)
    b = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    packed_cols = Int32(size(gate_packed, 1))

    # Two accumulators: one for gate, one for up
    gate_acc = 0.0f0
    up_acc = 0.0f0

    shared_gate = MtlThreadGroupArray(Float32, 32)
    shared_up = MtlThreadGroupArray(Float32, 32)

    pc = tid
    while pc <= packed_cols
        # Read input activation column indices (shared between gate and up)
        col_base = (pc - Int32(1)) << Int32(3)

        # Gate weights
        @inbounds gate_pv = gate_packed[pc, row]
        gate_group = (col_base >> Int32(6)) + Int32(1)
        @inbounds gs = Float32(gate_scales[gate_group, row])
        @inbounds gb = Float32(gate_biases[gate_group, row])

        # Up weights
        @inbounds up_pv = up_packed[pc, row]
        up_group = gate_group  # same group index (same K dimension)
        @inbounds us = Float32(up_scales[up_group, row])
        @inbounds ub = Float32(up_biases[up_group, row])

        # Unpack and accumulate both projections
        k = Int32(0)
        while k < Int32(8)
            col = col_base + k + Int32(1)
            @inbounds xval = Float32(x[col, b])

            # Gate
            gq = (gate_pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)
            gate_acc += (gs * Float32(gq) + gb) * xval

            # Up
            uq = (up_pv >> (UInt32(k) << UInt32(2))) & UInt32(0xF)
            up_acc += (us * Float32(uq) + ub) * xval

            k += Int32(1)
        end
        pc += tg_size
    end

    # Reduce gate accumulator
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        gate_acc += simd_shuffle_down(gate_acc, offset)
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared_gate[wid] = gate_acc
    end

    # Reduce up accumulator
    offset = UInt32(1)
    while offset < threads_per_simdgroup()
        up_acc += simd_shuffle_down(up_acc, offset)
        offset <<= 1
    end
    if lane == UInt32(1)
        @inbounds shared_up[wid] = up_acc
    end

    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if wid == UInt32(1)
        gate_acc = if lane <= nwarps
            @inbounds shared_gate[lane]
        else
            0.0f0
        end
        up_acc = if lane <= nwarps
            @inbounds shared_up[lane]
        else
            0.0f0
        end

        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            gate_acc += simd_shuffle_down(gate_acc, offset)
            up_acc += simd_shuffle_down(up_acc, offset)
            offset <<= 1
        end

        if lane == UInt32(1)
            # SwiGLU: silu(gate) * up = gate / (1 + exp(-gate)) * up
            g = gate_acc
            u = up_acc
            result = g / (1.0f0 + exp(-g)) * u
            @inbounds out[row, b] = typeof(out[1,1])(result)
        end
    end

    return nothing
end

"""
    metal_fused_gate_up_swiglu!(out, x, gate_layer, up_layer)

Fused gate_proj + up_proj + SwiGLU in one kernel dispatch.
Replaces: gate = gate_proj(x); up = up_proj(x); out = silu(gate) * up
"""
function metal_fused_gate_up_swiglu!(out, x, gate_layer, up_layer)
    O = gate_layer.out_features
    B = size(x, 2)
    packed_cols = size(gate_layer.weight, 1)

    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    @metal threads=tg_size groups=(O, B) fused_gate_up_swiglu_kernel!(
        out, x,
        gate_layer.weight, gate_layer.scales, gate_layer.biases,
        up_layer.weight, up_layer.scales, up_layer.biases)
    return out
end
