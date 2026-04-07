"""
Forward pass for Llama model.

Implements the full inference pipeline:
  embed → [RMSNorm → Attention → Residual → RMSNorm → FFN → Residual] × N → Norm → lm_head
"""

# ── Attention forward ──

"""
    attention_forward(attn, x, cos_table, sin_table, cache, layer_idx, config)

Single attention layer forward pass.
x: (hidden_size, seq_len) Float16
Returns: (hidden_size, seq_len) Float16
"""
function attention_forward(attn::LlamaAttention, x::MtlMatrix{Float16},
                          cos_table::MtlMatrix{Float32}, sin_table::MtlMatrix{Float32},
                          cache::KVCache, layer_idx::Int, config::LlamaConfig)
    hidden, seq_len = size(x)

    # Project Q, K, V
    q = attn.q_proj(x)   # (n_heads * head_dim, seq_len)
    k = attn.k_proj(x)   # (n_kv_heads * head_dim, seq_len)
    v = attn.v_proj(x)   # (n_kv_heads * head_dim, seq_len)

    # Reshape to (head_dim, n_heads, seq_len)
    q_3d = reshape(q, config.head_dim, config.num_attention_heads, seq_len)
    k_3d = reshape(k, config.head_dim, config.num_key_value_heads, seq_len)
    v_3d = reshape(v, config.head_dim, config.num_key_value_heads, seq_len)

    # Apply RoPE (in-place)
    start_pos = cache.seq_len + 1  # 1-indexed position for the new tokens
    metal_rope!(q_3d, cos_table, sin_table, start_pos)
    metal_rope!(k_3d, cos_table, sin_table, start_pos)

    # Append K, V to cache
    append_kv!(cache, layer_idx, k_3d, v_3d)

    # Get full K, V from cache (including all previous tokens)
    # We need contiguous arrays for attention, not views
    total_seq = cache.seq_len + seq_len
    k_full = MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, total_seq))
    v_full = MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, total_seq))

    # Copy from cache — this is the filled portion
    # Actually, we already appended to cache above. But cache.seq_len hasn't been
    # updated yet (it's done at the model level after all layers). So we need the
    # full range [1 : cache.seq_len + seq_len].
    #
    # Wait — we need to be careful about the sequence of operations.
    # Let's just use the cache directly. After append_kv!, the data is at
    # positions [cache.seq_len+1 : cache.seq_len+seq_len].
    # The full K/V for attention spans [1 : cache.seq_len + seq_len].

    # Use a view of the cache for attention
    effective_seq_len = cache.seq_len + seq_len
    k_attn = cache.k_cache[layer_idx][:, :, 1:effective_seq_len]
    v_attn = cache.v_cache[layer_idx][:, :, 1:effective_seq_len]

    # Need contiguous copy for attention kernel (views don't work with @metal)
    k_contig = MtlArray{Float16}(undef, config.head_dim, config.num_key_value_heads, effective_seq_len)
    v_contig = MtlArray{Float16}(undef, config.head_dim, config.num_key_value_heads, effective_seq_len)
    copyto!(k_contig, k_attn)
    copyto!(v_contig, v_attn)

    # Compute attention
    scale = 1.0f0 / sqrt(Float32(config.head_dim))
    attn_out = MtlArray(zeros(Float16, config.head_dim, config.num_attention_heads, seq_len))

    # causal_offset = number of cached positions before this chunk
    metal_attention!(attn_out, q_3d, k_contig, v_contig, scale;
                     causal=true, causal_offset=cache.seq_len)

    # Reshape back to (hidden_size, seq_len) and project
    attn_flat = reshape(attn_out, hidden, seq_len)
    out = attn.o_proj(attn_flat)

    return out
end

# ── MLP forward ──

function mlp_forward(mlp::LlamaMLP, x::MtlMatrix{Float16})
    gate = mlp.gate_proj(x)  # (intermediate, seq_len)
    up = mlp.up_proj(x)      # (intermediate, seq_len)

    # SwiGLU: silu(gate) * up
    activated = MtlArray(zeros(Float16, size(gate)))
    metal_swiglu!(activated, gate, up)

    out = mlp.down_proj(activated)
    return out
end

# ── Single layer forward ──

function layer_forward(layer::LlamaLayer, x::MtlMatrix{Float16},
                      cos_table::MtlMatrix{Float32}, sin_table::MtlMatrix{Float32},
                      cache::KVCache, layer_idx::Int, config::LlamaConfig)
    hidden, seq_len = size(x)

    # Pre-attention norm
    normed = MtlArray(zeros(Float16, hidden, seq_len))
    metal_rmsnorm!(normed, x, layer.input_layernorm, config.rms_norm_eps)

    # Attention + residual
    attn_out = attention_forward(layer.self_attn, normed, cos_table, sin_table,
                                cache, layer_idx, config)
    metal_add!(x, attn_out)  # x = x + attn_out

    # Post-attention norm
    metal_rmsnorm!(normed, x, layer.post_attention_layernorm, config.rms_norm_eps)

    # MLP + residual
    mlp_out = mlp_forward(layer.mlp, normed)
    metal_add!(x, mlp_out)  # x = x + mlp_out

    return x
end

# ── Full model forward ──

"""
    forward(model, token_ids, cache) -> logits

Run a full forward pass through the model.
token_ids: Vector{Int32} of 0-indexed token IDs
cache: KVCache (mutated — K/V appended for each layer)
Returns: logits matrix (vocab_size, seq_len) Float16
"""
function forward(model::LlamaModel, token_ids::MtlVector{Int32}, cache::KVCache)
    config = model.config

    # Embedding lookup
    x = model.embed(token_ids)  # (hidden_size, seq_len)
    seq_len = size(x, 2)

    # Transformer layers
    for (i, layer) in enumerate(model.layers)
        x = layer_forward(layer, x, model.cos_table, model.sin_table,
                         cache, i, config)
    end

    # Update cache sequence length (after all layers have appended)
    cache.seq_len += seq_len

    # Final norm
    normed = MtlArray(zeros(Float16, config.hidden_size, seq_len))
    metal_rmsnorm!(normed, x, model.norm, config.rms_norm_eps)

    # lm_head projection
    if model.lm_head !== nothing
        logits = model.lm_head(normed)
    else
        # tie_word_embeddings: use embedding table transposed as lm_head
        # logits[v, s] = sum_d embed_table[d, v] * normed[d, s]
        # This is a standard matmul: embed_table^T @ normed
        # embed_table is (embed_dim, vocab_size), normed is (embed_dim, seq_len)
        # Result: (vocab_size, seq_len)
        logits = lm_head_tied(model.embed, normed)
    end

    return logits
end

"""
Compute logits using tied embedding weights.
Uses the dequantized embedding table for the matrix multiply.
"""
function lm_head_tied_kernel!(logits, table, x, vocab_size::Int32, embed_dim::Int32)
    # Grid: (vocab_size, seq_len)
    v = Int32(threadgroup_position_in_grid().x)
    s = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_position_in_threadgroup().x)
    tg_size = Int32(threads_per_threadgroup().x)
    lane = thread_index_in_simdgroup()
    wid = simdgroup_index_in_threadgroup()
    nwarps = simdgroups_per_threadgroup()

    shared = MtlThreadGroupArray(Float32, 32)

    # Dot product: sum_d table[d, v] * x[d, s]
    acc = 0.0f0
    d = tid
    while d <= embed_dim
        @inbounds acc += Float32(table[d, v]) * Float32(x[d, s])
        d += tg_size
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
        acc = if lane <= nwarps
            @inbounds shared[lane]
        else
            0.0f0
        end
        offset = UInt32(1)
        while offset < threads_per_simdgroup()
            acc += simd_shuffle_down(acc, offset)
            offset <<= 1
        end
        if lane == UInt32(1)
            @inbounds logits[v, s] = Float16(acc)
        end
    end

    return nothing
end

function lm_head_tied(embed::QuantizedEmbedding, x::MtlMatrix{Float16})
    embed_dim, seq_len = size(x)
    vocab_size = embed.vocab_size

    logits = MtlArray(zeros(Float16, vocab_size, seq_len))

    tg = min(embed_dim, 256)
    tg = max(tg - (tg % 32), 32)

    @metal threads=tg groups=(vocab_size, seq_len) lm_head_tied_kernel!(
        logits, embed.table, x, Int32(vocab_size), Int32(embed_dim))

    return logits
end

# ── Greedy decoding ──

"""
    argmax_kernel! — find the argmax of each column

Not performance critical — just a simple reduction.
"""
function argmax_last_col_cpu(logits)
    # Get last column (last token's logits)
    h = Array(logits)
    last_col = h[:, end]
    return Int32(argmax(last_col) - 1)  # 0-indexed
end

"""
    generate(model, token_ids; max_tokens=50, temperature=0.0)

Greedy autoregressive generation.
token_ids: Vector{Int} of 0-indexed token IDs (the prompt)
Returns: Vector{Int} of generated token IDs (0-indexed, not including prompt)
"""
function generate(model::LlamaModel, prompt_ids::Vector{Int};
                  max_tokens::Int=50, temperature::Float64=0.0)
    config = model.config
    cache = KVCache(config; max_seq_len=length(prompt_ids) + max_tokens + 16)

    generated = Int[]

    # Prefill: process the entire prompt at once
    prompt_gpu = MtlArray(Int32.(prompt_ids))
    logits = forward(model, prompt_gpu, cache)
    Metal.synchronize()

    # Get next token from last position
    next_token = argmax_last_col_cpu(logits)
    push!(generated, Int(next_token))

    # Decode: one token at a time
    for step in 1:max_tokens-1
        if Int(next_token) in config.eos_token_ids
            break
        end

        token_gpu = MtlArray(Int32[next_token])
        logits = forward(model, token_gpu, cache)
        Metal.synchronize()

        next_token = argmax_last_col_cpu(logits)
        push!(generated, Int(next_token))
    end

    return generated
end
