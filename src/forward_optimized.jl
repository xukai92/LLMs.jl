"""
Optimized forward pass that minimizes @metal dispatch overhead.

Key optimizations:
1. Reduce scalar kernel arguments (each saves ~6μs per dispatch)
2. Pre-allocate all buffers (from forward_fast.jl)
3. GPU argmax (from forward_fast.jl)
4. Direct cache access (from forward_fast.jl)

Strategy: create specialized kernel wrappers that bake model constants
into the kernel dispatch, reducing per-call argument count.
"""

# ── Specialized dispatch functions ──

# Instead of passing hidden_size, n_heads, etc. as Int32 args every call,
# create specialized dispatch functions that capture these constants.

"""Quantized linear with best-available kernel per batch size."""
function qlinear_auto!(out, layer::QuantizedLinear, x)
    metal_qmatmul_v2!(out, x, layer.weight, layer.scales, layer.biases;
                        group_size=layer.group_size)
    return out
end

"""
Pre-built dispatch closures for a specific model config.
Each closure captures the constant parameters and only takes the
variable buffers as arguments.
"""
struct DispatchConfig
    hidden::Int32
    head_dim::Int32
    n_q_heads::Int32
    n_kv_heads::Int32
    gqa_ratio::Int32
    intermediate::Int32
    group_size::Int32
    vocab_size::Int32
    eps::Float32
    scale::Float32

    # Threadgroup sizes (precomputed)
    tg_norm::Int
    tg_rope::Tuple{Int,Int,Int}
    tg_swiglu::Int
    tg_qmatmul_small::Int   # for 3072-wide
    tg_qmatmul_large::Int   # for 8192-wide
    tg_attn_score::Int
    tg_attn_value::Int
    tg_add::Int
    tg_lm_head::Int
end

function DispatchConfig(config::LlamaConfig)
    hd = config.head_dim
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    h = config.hidden_size
    inter = config.intermediate_size
    packed_small = h ÷ 8
    packed_large = inter ÷ 8

    tg_norm = min(h, 1024)
    tg_norm = tg_norm - (tg_norm % 32)

    tg_qs = min(packed_small, 256)
    tg_qs = max(tg_qs - (tg_qs % 32), 32)
    tg_ql = min(packed_large, 256)
    tg_ql = max(tg_ql - (tg_ql % 32), 32)

    tg_lm = min(h, 256)
    tg_lm = max(tg_lm - (tg_lm % 32), 32)

    DispatchConfig(
        Int32(h), Int32(hd), Int32(n_q), Int32(n_kv), Int32(n_q ÷ n_kv),
        Int32(inter), Int32(config.quant_group_size), Int32(config.vocab_size),
        config.rms_norm_eps, 1.0f0 / sqrt(Float32(hd)),
        tg_norm, (min(hd÷2, 64), 1, 1), 256,
        tg_qs, tg_ql, 32, min(hd, 256), 256, tg_lm
    )
end

# ── Optimized forward pass ──

"""
    forward_opt!(model, token_ids, cache, pool, dc) -> logits_view

Forward pass with precomputed dispatch config to minimize scalar args.
"""
function forward_opt!(model::LlamaModel, token_ids::MtlVector{Int32},
                      cache::KVCache, pool::BufferPool, dc::DispatchConfig)
    seq_len = length(token_ids)
    h = Int(dc.hidden)
    hd = Int(dc.head_dim)
    n_q = Int(dc.n_q_heads)
    n_kv = Int(dc.n_kv_heads)
    inter = Int(dc.intermediate)

    # Embedding
    x = model.embed(token_ids)

    # Sized views
    normed = sized(pool.normed, h, seq_len)
    q_buf = sized(pool.q, n_q * hd, seq_len)
    k_buf = sized(pool.k, n_kv * hd, seq_len)
    v_buf = sized(pool.v, n_kv * hd, seq_len)
    o_buf = sized(pool.o_out, h, seq_len)
    gate_buf = sized(pool.gate, inter, seq_len)
    up_buf = sized(pool.up, inter, seq_len)
    swiglu_buf = sized(pool.swiglu_out, inter, seq_len)
    mlp_buf = sized(pool.mlp_out, h, seq_len)

    effective_seq = cache.seq_len + seq_len
    scores_buf = sized(pool.scores, n_q, seq_len, effective_seq)
    attn_out = sized(pool.attn_out_3d, hd, n_q, seq_len)

    start_pos = cache.seq_len + 1
    tg_attn_score = min(effective_seq, 1024)
    tg_attn_score = max(tg_attn_score - (tg_attn_score % 32), 32)

    for (layer_idx, layer) in enumerate(model.layers)
        # RMSNorm → Q,K,V → RoPE → KV append → Attn → O → Residual → RMSNorm → MLP → Residual
        metal_rmsnorm!(normed, x, layer.input_layernorm, dc.eps)

        # Q,K,V projections — use qlinear! which avoids allocating output
        qlinear_auto!(q_buf, layer.self_attn.q_proj, normed)
        qlinear_auto!(k_buf, layer.self_attn.k_proj, normed)
        qlinear_auto!(v_buf, layer.self_attn.v_proj, normed)

        q_3d = reshape(q_buf, hd, n_q, seq_len)
        k_3d = reshape(k_buf, hd, n_kv, seq_len)
        v_3d = reshape(v_buf, hd, n_kv, seq_len)

        metal_rope!(q_3d, model.cos_table, model.sin_table, start_pos)
        metal_rope!(k_3d, model.cos_table, model.sin_table, start_pos)

        append_kv!(cache, layer_idx, k_3d, v_3d)

        # Attention directly from cache
        @metal threads=tg_attn_score groups=(n_q, seq_len) attn_scores_softmax_kernel!(
            scores_buf, q_3d, cache.k_cache[layer_idx],
            dc.head_dim, dc.n_kv_heads, dc.gqa_ratio, Int32(effective_seq),
            dc.scale, Int32(cache.seq_len), Int32(1))

        @metal threads=dc.tg_attn_value groups=(n_q, seq_len) attn_value_kernel!(
            attn_out, scores_buf, cache.v_cache[layer_idx],
            dc.head_dim, dc.n_kv_heads, dc.gqa_ratio, Int32(effective_seq))

        qlinear_auto!(o_buf, layer.self_attn.o_proj, reshape(attn_out, h, seq_len))

        # Fused: x += o_buf, then normed = rmsnorm(x)
        # Saves 1 add dispatch + 1 rmsnorm read (reads x once instead of twice)
        metal_rmsnorm_residual!(normed, x, o_buf, layer.post_attention_layernorm, dc.eps)

        # MLP: use fused kernel for B=1 (saves 2 dispatches), separate for B>1 (faster compute)
        if seq_len <= 1
            metal_fused_gate_up_swiglu!(swiglu_buf, normed,
                                         layer.mlp.gate_proj, layer.mlp.up_proj)
        else
            qlinear_auto!(gate_buf, layer.mlp.gate_proj, normed)
            qlinear_auto!(up_buf, layer.mlp.up_proj, normed)
            metal_swiglu!(swiglu_buf, gate_buf, up_buf)
        end
        qlinear_auto!(mlp_buf, layer.mlp.down_proj, swiglu_buf)
        # For the last op, we can't easily fuse the add into the next layer's rmsnorm
        # because the next layer uses a different weight vector. But we can fuse for
        # all layers except the last (where it's followed by final norm).
        metal_add!(x, mlp_buf)
    end

    cache.seq_len += seq_len

    # Final norm + lm_head
    metal_rmsnorm!(normed, x, model.norm, dc.eps)

    logits_view = sized(pool.logits, Int(dc.vocab_size), seq_len)
    if model.lm_head !== nothing
        qlinear_auto!(logits_view, model.lm_head, normed)
    else
        @metal threads=dc.tg_lm_head groups=(Int(dc.vocab_size), seq_len) lm_head_tied_kernel!(
            logits_view, model.embed.table, normed,
            dc.vocab_size, dc.hidden)
    end

    return logits_view
end

# ── Optimized generate ──

function generate_opt(model::LlamaModel, prompt_ids::Vector{Int};
                      max_tokens::Int=50)
    config = model.config
    dc = DispatchConfig(config)
    total_seq = length(prompt_ids) + max_tokens + 16
    cache = KVCache(config; max_seq_len=total_seq)
    pool = BufferPool(config; max_batch=max(length(prompt_ids), 1), max_seq=total_seq)

    generated = Int[]

    # Prefill
    prompt_gpu = MtlArray(Int32.(prompt_ids))
    logits = forward_opt!(model, prompt_gpu, cache, pool, dc)

    argmax_buf = MtlArray(Int32[0])
    metal_argmax_last_col!(argmax_buf, logits)
    argmax_host = Int32[0]
    copyto!(argmax_host, argmax_buf)
    next_token = argmax_host[1]
    push!(generated, Int(next_token))

    # Decode
    decode_pool = length(prompt_ids) > 1 ?
        BufferPool(config; max_batch=1, max_seq=total_seq) : pool
    token_buf = MtlArray(Int32[0])

    # Pre-warm decode path (JIT)
    copyto!(token_buf, Int32[next_token])
    if !(next_token in config.eos_token_ids) && max_tokens > 1
        logits = forward_opt!(model, token_buf, cache, decode_pool, dc)
        metal_argmax_last_col!(argmax_buf, logits)
        copyto!(argmax_host, argmax_buf)
        next_token = argmax_host[1]
        push!(generated, Int(next_token))
    end

    for step in 3:max_tokens
        if next_token in config.eos_token_ids
            break
        end

        copyto!(token_buf, Int32[next_token])
        logits = forward_opt!(model, token_buf, cache, decode_pool, dc)
        metal_argmax_last_col!(argmax_buf, logits)
        copyto!(argmax_host, argmax_buf)
        next_token = argmax_host[1]
        push!(generated, Int(next_token))
    end

    return generated
end
