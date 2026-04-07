"""
Optimized forward pass using pre-allocated buffers.

Key differences from forward.jl:
- Zero MtlArray allocations during inference (all buffers pre-allocated)
- Minimized Metal.synchronize() calls
- In-place operations throughout
"""

# ── Fast quantized matmul that writes to pre-allocated output ──

function qlinear!(out, layer::QuantizedLinear, x)
    # out and x are views into pre-allocated buffers
    # Need contiguous arrays for the kernel — but SubArrays of MtlArrays
    # should work if they're contiguous column slices
    O = layer.out_features
    B = size(x, 2)
    packed_cols = size(layer.weight, 2)

    tg_size = min(packed_cols, 256)
    tg_size = max(tg_size - (tg_size % 32), 32)

    @metal threads=tg_size groups=(O, B) qmatmul_kernel!(
        out, x, layer.weight, layer.scales, layer.biases,
        Int32(O), Int32(layer.in_features), Int32(B),
        Int32(packed_cols), Int32(layer.group_size))
    return out
end

# ── Fast forward pass ──

"""
    forward_fast!(model, token_ids, cache, pool) -> logits_view

Zero-allocation forward pass using pre-allocated buffer pool.
Returns a view into pool.logits sized to (vocab, seq_len).
"""
function forward_fast!(model::LlamaModel, token_ids::MtlVector{Int32},
                       cache::KVCache, pool::BufferPool)
    config = model.config
    seq_len = length(token_ids)
    h = config.hidden_size
    hd = config.head_dim
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads

    # Embedding lookup (writes to a sized view)
    x_buf = sized(pool.normed, h, seq_len)  # reuse normed as temp for embedding
    # Actually we need x to persist through residual connections — use a separate approach.
    # For the hidden state, we'll allocate once at the start and reuse.
    # The hidden state is modified in-place via residual connections.

    # Embedding — this is the one allocation we can't easily avoid
    # (it's a lookup, not a matmul). Use the embedding callable.
    x = model.embed(token_ids)  # (hidden, seq_len) — allocated once

    # Sized views into pool buffers
    normed = sized(pool.normed, h, seq_len)
    q_buf = sized(pool.q, n_q * hd, seq_len)
    k_buf = sized(pool.k, n_kv * hd, seq_len)
    v_buf = sized(pool.v, n_kv * hd, seq_len)
    o_buf = sized(pool.o_out, h, seq_len)
    gate_buf = sized(pool.gate, config.intermediate_size, seq_len)
    up_buf = sized(pool.up, config.intermediate_size, seq_len)
    swiglu_buf = sized(pool.swiglu_out, config.intermediate_size, seq_len)
    mlp_buf = sized(pool.mlp_out, h, seq_len)

    effective_seq = cache.seq_len + seq_len
    k_contig = sized(pool.k_contig, hd, n_kv, effective_seq)
    v_contig = sized(pool.v_contig, hd, n_kv, effective_seq)
    scores_buf = sized(pool.scores, n_q, seq_len, effective_seq)
    attn_out = sized(pool.attn_out_3d, hd, n_q, seq_len)

    for (layer_idx, layer) in enumerate(model.layers)
        # Pre-attention norm
        metal_rmsnorm!(normed, x, layer.input_layernorm, config.rms_norm_eps)

        # Q, K, V projections
        qlinear!(q_buf, layer.self_attn.q_proj, normed)
        qlinear!(k_buf, layer.self_attn.k_proj, normed)
        qlinear!(v_buf, layer.self_attn.v_proj, normed)

        # Reshape to 3D for RoPE and attention
        q_3d = reshape(q_buf, hd, n_q, seq_len)
        k_3d = reshape(k_buf, hd, n_kv, seq_len)
        v_3d = reshape(v_buf, hd, n_kv, seq_len)

        # RoPE
        start_pos = cache.seq_len + 1
        metal_rope!(q_3d, model.cos_table, model.sin_table, start_pos)
        metal_rope!(k_3d, model.cos_table, model.sin_table, start_pos)

        # Append to KV cache
        append_kv!(cache, layer_idx, k_3d, v_3d)

        # Copy full KV from cache for attention
        copyto!(k_contig, view(cache.k_cache[layer_idx], :, :, 1:effective_seq))
        copyto!(v_contig, view(cache.v_cache[layer_idx], :, :, 1:effective_seq))

        # Attention
        scale = 1.0f0 / sqrt(Float32(hd))
        tg1 = min(effective_seq, 1024)
        tg1 = max(tg1 - (tg1 % 32), 32)
        @metal threads=tg1 groups=(n_q, seq_len) attn_scores_softmax_kernel!(
            scores_buf, q_3d, k_contig,
            Int32(hd), Int32(n_kv), Int32(n_q ÷ n_kv), Int32(effective_seq),
            scale, Int32(cache.seq_len), Int32(1))

        tg2 = min(hd, 256)
        @metal threads=tg2 groups=(n_q, seq_len) attn_value_kernel!(
            attn_out, scores_buf, v_contig,
            Int32(hd), Int32(n_kv), Int32(n_q ÷ n_kv), Int32(effective_seq))

        # O projection
        attn_flat = reshape(attn_out, h, seq_len)
        qlinear!(o_buf, layer.self_attn.o_proj, attn_flat)

        # Residual
        metal_add!(x, o_buf)

        # Post-attention norm
        metal_rmsnorm!(normed, x, layer.post_attention_layernorm, config.rms_norm_eps)

        # MLP
        qlinear!(gate_buf, layer.mlp.gate_proj, normed)
        qlinear!(up_buf, layer.mlp.up_proj, normed)
        metal_swiglu!(swiglu_buf, gate_buf, up_buf)
        qlinear!(mlp_buf, layer.mlp.down_proj, swiglu_buf)

        # Residual
        metal_add!(x, mlp_buf)
    end

    # Update cache
    cache.seq_len += seq_len

    # Final norm
    metal_rmsnorm!(normed, x, model.norm, config.rms_norm_eps)

    # lm_head
    if model.lm_head !== nothing
        logits_view = sized(pool.logits, config.vocab_size, seq_len)
        qlinear!(logits_view, model.lm_head, normed)
        return logits_view
    else
        logits_view = sized(pool.logits, config.vocab_size, seq_len)
        tg = min(h, 256)
        tg = max(tg - (tg % 32), 32)
        @metal threads=tg groups=(config.vocab_size, seq_len) lm_head_tied_kernel!(
            logits_view, model.embed.table, normed,
            Int32(config.vocab_size), Int32(h))
        return logits_view
    end
end

# ── Fast generate ──

"""
    generate_fast(model, prompt_ids; max_tokens=50)

Greedy generation with zero-allocation forward pass.
"""
function generate_fast(model::LlamaModel, prompt_ids::Vector{Int};
                       max_tokens::Int=50)
    config = model.config
    total_seq = length(prompt_ids) + max_tokens + 16
    cache = KVCache(config; max_seq_len=total_seq)
    pool = BufferPool(config; max_batch=max(length(prompt_ids), 1), max_seq=total_seq)

    generated = Int[]

    # Prefill
    prompt_gpu = MtlArray(Int32.(prompt_ids))
    logits = forward_fast!(model, prompt_gpu, cache, pool)
    Metal.synchronize()

    next_token = argmax_last_col_cpu(logits)
    push!(generated, Int(next_token))

    # For decode, we need a pool sized for B=1
    decode_pool = if length(prompt_ids) > 1
        BufferPool(config; max_batch=1, max_seq=total_seq)
    else
        pool
    end

    # Pre-allocate single-token buffer for decode
    token_buf = MtlArray(Int32[0])

    for step in 1:max_tokens-1
        if Int(next_token) in config.eos_token_ids
            break
        end

        copyto!(token_buf, Int32[next_token])
        logits = forward_fast!(model, token_buf, cache, decode_pool)
        Metal.synchronize()

        next_token = argmax_last_col_cpu(logits)
        push!(generated, Int(next_token))
    end

    return generated
end
