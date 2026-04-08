"""
FP16 forward pass: dequantizes all weights at load time, uses metal_fp16_matmul!.

This isolates the infrastructure overhead from the quantized matmul gap,
since our FP16 kernel is competitive with MLX (0.78-0.93x at B=32-128).
"""

# ── Dequantized linear layer ──

struct FP16Linear
    weight::MtlMatrix{Float16}  # (out_features, in_features) dequantized
    out_features::Int
    in_features::Int
end

"""Dequantize a QuantizedLinear layer to FP16 weights on GPU."""
function dequantize_linear(ql::QuantizedLinear)
    O, I = ql.out_features, ql.in_features
    packed_cols = I ÷ 8
    gs = ql.group_size

    # Dequantize on CPU (one-time cost at model load)
    pk = Array(ql.weight); sc = Array(ql.scales); bi = Array(ql.biases)
    W = zeros(Float16, O, I)
    for o in 1:O, pc in 1:packed_cols
        pv = pk[pc, o]
        col_base = (pc - 1) * 8
        grp = col_base ÷ gs + 1
        s = Float32(sc[grp, o]); b = Float32(bi[grp, o])
        for k in 0:7
            W[o, col_base + k + 1] = Float16(s * Float32((pv >> (UInt32(k) * 4)) & 0xF) + b)
        end
    end

    # Pad to multiples of 32 for our FP16 kernel
    O_pad = cld(O, 32) * 32
    I_pad = cld(I, 32) * 32
    if O_pad != O || I_pad != I
        W_pad = zeros(Float16, O_pad, I_pad)
        W_pad[1:O, 1:I] = W
        FP16Linear(MtlArray(W_pad), O, I)
    else
        FP16Linear(MtlArray(W), O, I)
    end
end

"""Pre-allocated padded buffers for FP16 forward pass, avoiding SubArray views."""
struct FP16BufferPool
    bufs::Dict{Tuple{Int,Int}, MtlMatrix{Float16}}
end
FP16BufferPool() = FP16BufferPool(Dict{Tuple{Int,Int}, MtlMatrix{Float16}}())

function get_padded!(pool::FP16BufferPool, rows::Int, cols::Int)
    rp = cld(rows, 32) * 32; cp = cld(cols, 32) * 32
    key = (rp, cp)
    get!(pool.bufs, key) do
        MtlArray(zeros(Float16, rp, cp))
    end
end

const _F16x2_fwd = NTuple{2, VecElement{Float16}}

# FP16 matmul kernel for forward pass: pointer() for W (padded MtlArray),
# array indexing for x and out (may be SubArray views)
function fp16_matmul_fwd!(out, W, x, M::Int32, N::Int32, K::Int32)
    tmg = Int32(threadgroup_position_in_grid().x); tng = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sgm = (sg - Int32(1)) % Int32(4); sgn = (sg - Int32(1)) ÷ Int32(4)
    w1=MtlThreadGroupArray(Float32,(32,8));x1=MtlThreadGroupArray(Float32,(8,32))
    w2=MtlThreadGroupArray(Float32,(32,8));x2=MtlThreadGroupArray(Float32,(8,32))
    w3=MtlThreadGroupArray(Float32,(32,8));x3=MtlThreadGroupArray(Float32,(8,32))
    w4=MtlThreadGroupArray(Float32,(32,8));x4=MtlThreadGroupArray(Float32,(8,32))
    w5=MtlThreadGroupArray(Float32,(32,8));x5=MtlThreadGroupArray(Float32,(8,32))
    w6=MtlThreadGroupArray(Float32,(32,8));x6=MtlThreadGroupArray(Float32,(8,32))
    w7=MtlThreadGroupArray(Float32,(32,8));x7=MtlThreadGroupArray(Float32,(8,32))
    w8=MtlThreadGroupArray(Float32,(32,8));x8=MtlThreadGroupArray(Float32,(8,32))
    zt=MtlThreadGroupArray(Float32,(8,8));res=MtlThreadGroupArray(Float32,(32,32))
    if sg==Int32(1)&&tid<=Int32(32)
        for e in Int32(0):Int32(1); f=(tid-Int32(1))*Int32(2)+e;r=(f%Int32(8))+Int32(1);c=(f÷Int32(8))+Int32(1)
            if c<=Int32(8); @inbounds zt[r,c]=0f0; end; end; end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc=simdgroup_load(zt,(1,1))
    # W: pointer + vec2 (W is padded MtlArray, not a view)
    M_W = Int32(size(W, 1))  # padded M dimension of W
    @inline function _lw(dst, ko)
        if gtid <= Int32(128)
            f=gtid-Int32(1);pair=f%Int32(16);c=(f÷Int32(16))+Int32(1);r1=pair*Int32(2)+Int32(1)
            gm=(tmg-Int32(1))*Int32(32)+r1; gk=ko+c
            p=pointer(W)+(Int64(gk-Int32(1))*Int64(M_W)+Int64(gm-Int32(1)))*Int64(2)
            vec=unsafe_load(reinterpret(Core.LLVMPtr{_F16x2_fwd,Metal.AS.Device},p))
            @inbounds dst[r1,c]=Float32(vec[1].value); @inbounds dst[r1+Int32(1),c]=Float32(vec[2].value)
        end
    end
    # x: array indexing (x may be a SubArray view)
    @inline function _lx(dst, ko)
        if gtid>Int32(128)&&gtid<=Int32(256)
            f=gtid-Int32(129);pair=f%Int32(4);c=(f÷Int32(4))+Int32(1);r1=pair*Int32(2)+Int32(1)
            gk=ko+r1; gn=(tng-Int32(1))*Int32(32)+c
            @inbounds dst[r1,c]=Float32(x[gk,gn]); @inbounds dst[r1+Int32(1),c]=Float32(x[gk+Int32(1),gn])
        end
    end
    wo=(Int64(sgm)*Int64(8)+Int64(1),Int64(1)); xo=(Int64(1),Int64(sgn)*Int64(8)+Int64(1))
    k=Int32(0)
    while k+Int32(64)<=K
        _lw(w1,k);_lx(x1,k);_lw(w2,k+Int32(8));_lx(x2,k+Int32(8))
        _lw(w3,k+Int32(16));_lx(x3,k+Int32(16));_lw(w4,k+Int32(24));_lx(x4,k+Int32(24))
        _lw(w5,k+Int32(32));_lx(x5,k+Int32(32));_lw(w6,k+Int32(40));_lx(x6,k+Int32(40))
        _lw(w7,k+Int32(48));_lx(x7,k+Int32(48));_lw(w8,k+Int32(56));_lx(x8,k+Int32(56))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w1,wo),simdgroup_load(x1,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w2,wo),simdgroup_load(x2,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w3,wo),simdgroup_load(x3,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w4,wo),simdgroup_load(x4,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w5,wo),simdgroup_load(x5,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w6,wo),simdgroup_load(x6,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w7,wo),simdgroup_load(x7,xo),acc)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w8,wo),simdgroup_load(x8,xo),acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup);k+=Int32(64)
    end
    while k<K; _lw(w1,k);_lx(x1,k);threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc=simdgroup_multiply_accumulate(simdgroup_load(w1,wo),simdgroup_load(x1,xo),acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup);k+=Int32(8); end
    simdgroup_store(acc,res,(Int64(sgm)*Int64(8)+Int64(1),Int64(sgn)*Int64(8)+Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for pp in Int32(0):Int32(1); idx=(gtid-Int32(1))*Int32(2)+pp+Int32(1)
        if idx<=Int32(1024); r=((idx-Int32(1))%Int32(32))+Int32(1);c=((idx-Int32(1))÷Int32(32))+Int32(1)
            gm=(tmg-Int32(1))*Int32(32)+r;gn=(tng-Int32(1))*Int32(32)+c
            if gm<=M&&gn<=N; @inbounds out[gm,gn]=Float16(res[r,c]); end
        end; end; return nothing
end

"""Unwrap SubArray/ReshapedArray to parent MtlArray if the view covers the full extent."""
@inline _unwrap(x::MtlMatrix) = x
@inline function _unwrap(x::SubArray)
    p = parent(x)
    size(x) == size(p) && p isa MtlMatrix{Float16} ? p : x
end
@inline _unwrap(x) = x  # fallback for ReshapedArray etc

function fp16_linear!(out, layer::FP16Linear, x)
    # Try to unwrap SubArray views to get the fast ptr+vec2 kernel
    x_raw = _unwrap(x); out_raw = _unwrap(out)
    if x_raw isa MtlMatrix{Float16} && out_raw isa MtlMatrix{Float16}
        metal_fp16_matmul!(out_raw, layer.weight, x_raw)
    else
        # Fallback: array indexing kernel for non-trivial views
        M = Int32(layer.out_features); K = Int32(layer.in_features); N = Int32(size(x, 2))
        @metal threads=512 groups=(cld(Int(M), 32), cld(Int(N), 32)) fp16_matmul_fwd!(
            out, layer.weight, x, M, N, K)
    end
    return out
end

# ── FP16 model wrapper ──

struct FP16Layer
    input_layernorm::Any  # RMSNorm weights
    post_attention_layernorm::Any
    q_proj::FP16Linear
    k_proj::FP16Linear
    v_proj::FP16Linear
    o_proj::FP16Linear
    gate_proj::FP16Linear
    up_proj::FP16Linear
    down_proj::FP16Linear
    # Fused weights (pre-concatenated at load time)
    qkv_proj::FP16Linear      # Q+K+V fused: (q_dim+k_dim+v_dim, in_features)
    gate_up_proj::FP16Linear   # gate+up fused: (2*intermediate, in_features)
end

struct FP16Model
    embed_table::MtlMatrix{Float16}
    layers::Vector{FP16Layer}
    norm::Any
    lm_head::Union{FP16Linear, Nothing}
    config::LlamaConfig
    cos_table::MtlMatrix{Float16}
    sin_table::MtlMatrix{Float16}
end

"""Convert a quantized LlamaModel to FP16 (dequantize all weights)."""
function to_fp16(model::LlamaModel)
    println("Dequantizing model to FP16...")
    layers = FP16Layer[]
    for (i, l) in enumerate(model.layers)
        print("  Layer $i/$(length(model.layers))\r")
        q = dequantize_linear(l.self_attn.q_proj)
        k = dequantize_linear(l.self_attn.k_proj)
        v = dequantize_linear(l.self_attn.v_proj)
        gate = dequantize_linear(l.mlp.gate_proj)
        up = dequantize_linear(l.mlp.up_proj)

        # Fuse QKV: concatenate weight matrices along output dim
        qkv_out = q.out_features + k.out_features + v.out_features
        qkv_out_pad = cld(qkv_out, 32) * 32
        qkv_w = zeros(Float16, qkv_out_pad, q.in_features)
        qkv_w[1:q.out_features, :] = Array(q.weight)[1:q.out_features, :]
        qkv_w[q.out_features+1:q.out_features+k.out_features, :] = Array(k.weight)[1:k.out_features, :]
        qkv_w[q.out_features+k.out_features+1:qkv_out, :] = Array(v.weight)[1:v.out_features, :]
        qkv = FP16Linear(MtlArray(qkv_w), qkv_out, q.in_features)

        # Fuse gate+up: concatenate along output dim
        gu_out = gate.out_features + up.out_features
        gu_out_pad = cld(gu_out, 32) * 32
        gu_w = zeros(Float16, gu_out_pad, gate.in_features)
        gu_w[1:gate.out_features, :] = Array(gate.weight)[1:gate.out_features, :]
        gu_w[gate.out_features+1:gu_out, :] = Array(up.weight)[1:up.out_features, :]
        gate_up = FP16Linear(MtlArray(gu_w), gu_out, gate.in_features)

        push!(layers, FP16Layer(
            l.input_layernorm, l.post_attention_layernorm,
            q, k, v,
            dequantize_linear(l.self_attn.o_proj),
            gate, up,
            dequantize_linear(l.mlp.down_proj),
            qkv, gate_up,
        ))
    end
    println("  Done.                    ")

    lm_head = model.lm_head !== nothing ? dequantize_linear(model.lm_head) : nothing

    FP16Model(model.embed.table, layers, model.norm, lm_head,
              model.config, model.cos_table, model.sin_table)
end

# ── FP16 forward pass ──

function forward_fp16!(model::FP16Model, token_ids::MtlVector{Int32},
                       cache::KVCache, pool::BufferPool, dc::DispatchConfig)
    seq_len = length(token_ids)
    h = Int(dc.hidden); hd = Int(dc.head_dim)
    n_q = Int(dc.n_q_heads); n_kv = Int(dc.n_kv_heads)
    inter = Int(dc.intermediate)

    # Embedding lookup
    x = MtlArray(zeros(Float16, h, seq_len))
    for i in 1:seq_len
        tid = Int(Array(token_ids)[i]) + 1
        copyto!(view(x, :, i), view(model.embed_table, :, tid))
    end

    # Use sized views for non-matmul ops (rmsnorm, rope, attention, add)
    normed = sized(pool.normed, h, seq_len)
    q_buf = sized(pool.q, n_q * hd, seq_len)
    k_buf = sized(pool.k, n_kv * hd, seq_len)
    v_buf = sized(pool.v, n_kv * hd, seq_len)
    o_buf = sized(pool.o_out, h, seq_len)
    gate_buf = sized(pool.gate, inter, seq_len)
    up_buf = sized(pool.up, inter, seq_len)
    swiglu_buf = sized(pool.swiglu_out, inter, seq_len)
    mlp_buf = sized(pool.mlp_out, h, seq_len)
    attn_out = sized(pool.attn_out_3d, hd, n_q, seq_len)

    start_pos = cache.seq_len + 1

    # Pre-allocate fused QKV output buffer
    qkv_dim = n_q * hd + n_kv * hd + n_kv * hd
    qkv_buf = MtlArray(zeros(Float16, cld(qkv_dim, 32) * 32, seq_len))
    # Pre-allocate fused gate+up output buffer
    gu_dim = inter * 2
    gu_buf = MtlArray(zeros(Float16, cld(gu_dim, 32) * 32, seq_len))

    for (layer_idx, layer) in enumerate(model.layers)
        metal_rmsnorm!(normed, x, layer.input_layernorm, dc.eps)

        # Fused QKV: one matmul instead of 3, read directly from output
        fp16_linear!(qkv_buf, layer.qkv_proj, normed)
        q_dim = n_q * hd; k_dim = n_kv * hd

        q_3d = reshape(view(qkv_buf, 1:q_dim, :), hd, n_q, seq_len)
        k_3d = reshape(view(qkv_buf, q_dim+1:q_dim+k_dim, :), hd, n_kv, seq_len)
        v_3d = reshape(view(qkv_buf, q_dim+k_dim+1:q_dim+2*k_dim, :), hd, n_kv, seq_len)

        metal_rope!(q_3d, model.cos_table, model.sin_table, start_pos)
        metal_rope!(k_3d, model.cos_table, model.sin_table, start_pos)
        append_kv!(cache, layer_idx, k_3d, v_3d)

        metal_flash_attention!(attn_out, q_3d, cache.k_cache[layer_idx],
                               cache.v_cache[layer_idx], dc.scale;
                               causal=true, causal_offset=cache.seq_len)

        fp16_linear!(o_buf, layer.o_proj, reshape(attn_out, h, seq_len))
        metal_rmsnorm_residual!(normed, x, o_buf, layer.post_attention_layernorm, dc.eps)

        # Fused gate+up: one matmul instead of 2, then swiglu on slices
        fp16_linear!(gu_buf, layer.gate_up_proj, normed)
        metal_swiglu!(swiglu_buf, view(gu_buf, 1:inter, :), view(gu_buf, inter+1:2*inter, :))
        fp16_linear!(mlp_buf, layer.down_proj, swiglu_buf)
        metal_add!(x, mlp_buf)
    end

    cache.seq_len += seq_len

    metal_rmsnorm!(normed, x, model.norm, dc.eps)
    logits_view = sized(pool.logits, Int(dc.vocab_size), seq_len)
    if model.lm_head !== nothing
        fp16_linear!(logits_view, model.lm_head, normed)
    end

    return logits_view
end
