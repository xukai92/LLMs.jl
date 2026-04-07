"""
Llama model definition and weight loading.

Supports LlamaForCausalLM with MLX 4-bit quantization format.
"""

# ── Configuration ──

struct LlamaConfig
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    num_key_value_heads::Int
    head_dim::Int
    vocab_size::Int
    max_position_embeddings::Int
    rms_norm_eps::Float32
    rope_theta::Float64
    rope_scaling::Union{Nothing, Dict{String, Any}}
    tie_word_embeddings::Bool
    bos_token_id::Int
    eos_token_ids::Vector{Int}
    # Quantization
    quant_bits::Int
    quant_group_size::Int
end

function LlamaConfig(config_path::String)
    cfg = JSON3.read(read(config_path, String))

    eos = if cfg.eos_token_id isa AbstractVector
        Int[e for e in cfg.eos_token_id]
    else
        Int[cfg.eos_token_id]
    end

    rope_scaling = if haskey(cfg, :rope_scaling) && cfg.rope_scaling !== nothing
        Dict{String, Any}(
            "factor" => Float64(cfg.rope_scaling.factor),
            "high_freq_factor" => Float64(cfg.rope_scaling.high_freq_factor),
            "low_freq_factor" => Float64(cfg.rope_scaling.low_freq_factor),
            "original_max_position_embeddings" => Int(cfg.rope_scaling.original_max_position_embeddings),
        )
    else
        nothing
    end

    qcfg = cfg.quantization
    LlamaConfig(
        Int(cfg.hidden_size),
        Int(cfg.intermediate_size),
        Int(cfg.num_hidden_layers),
        Int(cfg.num_attention_heads),
        Int(cfg.num_key_value_heads),
        Int(cfg.head_dim),
        Int(cfg.vocab_size),
        Int(cfg.max_position_embeddings),
        Float32(cfg.rms_norm_eps),
        Float64(cfg.rope_theta),
        rope_scaling,
        Bool(cfg.tie_word_embeddings),
        Int(cfg.bos_token_id),
        eos,
        Int(qcfg.bits),
        Int(qcfg.group_size),
    )
end

# ── Quantized Linear Layer ──

struct QuantizedLinear
    weight::MtlMatrix{UInt32}     # (O, I/8) packed 4-bit
    scales::MtlMatrix{Float16}    # (O, I/group_size)
    biases::MtlMatrix{Float16}    # (O, I/group_size)
    out_features::Int
    in_features::Int
    group_size::Int
end

function (layer::QuantizedLinear)(x::MtlMatrix{Float16})
    out = MtlArray(zeros(Float16, layer.out_features, size(x, 2)))
    metal_quantized_matmul!(out, x, layer.weight, layer.scales, layer.biases;
                             group_size=layer.group_size)
    return out
end

# ── Embedding (quantized) ──

struct QuantizedEmbedding
    weight::MtlMatrix{UInt32}
    scales::MtlMatrix{Float16}
    biases::MtlMatrix{Float16}
    vocab_size::Int
    embed_dim::Int
    group_size::Int
    # Dequantized embedding table (computed once at load time)
    table::MtlMatrix{Float16}
end

"""
Embedding lookup kernel: each thread loads one (token, dim) pair.
"""
function embed_lookup_kernel!(out, table, token_ids, embed_dim::Int32)
    tid = thread_position_in_grid_1d()
    n_tokens = Int32(length(token_ids))
    total = embed_dim * n_tokens

    if tid <= total
        # tid goes 1..embed_dim*n_tokens
        d = ((tid - Int32(1)) % embed_dim) + Int32(1)
        t = ((tid - Int32(1)) ÷ embed_dim) + Int32(1)
        @inbounds token_id = token_ids[t] + Int32(1)  # 0-indexed token → 1-indexed row
        @inbounds out[d, t] = table[d, token_id]
    end
    return nothing
end

function (emb::QuantizedEmbedding)(token_ids::MtlVector{Int32})
    n_tokens = length(token_ids)
    out = MtlArray(zeros(Float16, emb.embed_dim, n_tokens))
    total = emb.embed_dim * n_tokens
    tg = 256
    @metal threads=tg groups=cld(total, tg) embed_lookup_kernel!(
        out, emb.table, token_ids, Int32(emb.embed_dim))
    return out
end

# ── Layer components ──

struct LlamaAttention
    q_proj::QuantizedLinear
    k_proj::QuantizedLinear
    v_proj::QuantizedLinear
    o_proj::QuantizedLinear
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

struct LlamaMLP
    gate_proj::QuantizedLinear
    up_proj::QuantizedLinear
    down_proj::QuantizedLinear
end

struct LlamaLayer
    input_layernorm::MtlVector{Float16}
    self_attn::LlamaAttention
    post_attention_layernorm::MtlVector{Float16}
    mlp::LlamaMLP
end

struct LlamaModel
    config::LlamaConfig
    embed::QuantizedEmbedding
    layers::Vector{LlamaLayer}
    norm::MtlVector{Float16}
    # lm_head reuses embed weights when tie_word_embeddings=true
    lm_head::Union{QuantizedLinear, Nothing}
    # Precomputed RoPE tables
    cos_table::MtlMatrix{Float32}
    sin_table::MtlMatrix{Float32}
end

# ── Weight loading ──

function load_quantized_linear(weights::Dict{String, Array}, prefix::String, group_size::Int)
    w = MtlArray(Float16.(reinterpret(Float16, weights["$(prefix).weight"])))  # UInt16 → Float16
    # Actually the weight is UInt32 packed
    w_u32 = MtlArray(weights["$(prefix).weight"])
    s = MtlArray(Float16.(reinterpret(Float16, weights["$(prefix).scales"])))
    b = MtlArray(Float16.(reinterpret(Float16, weights["$(prefix).biases"])))

    # Safetensors row-major [O, packed_cols] → Julia col-major (packed_cols, O).
    # Keep this layout! It gives coalesced GPU memory access:
    # Kernel threads iterate over packed_cols (first dim = contiguous in memory).
    # Kernels now index as packed[pc, row], scales[grp, row], biases[grp, row].

    w_packed = w_u32  # (packed_cols, O) — no transpose!
    s_kept = s         # (n_groups, O)
    b_kept = b         # (n_groups, O)

    packed_cols = size(w_packed, 1)
    O = size(w_packed, 2)
    I = packed_cols * 8

    return QuantizedLinear(w_packed, s_kept, b_kept, O, I, group_size)
end

function load_embedding(weights::Dict{String, Array}, prefix::String,
                        vocab_size::Int, embed_dim::Int, group_size::Int)
    w_u32 = MtlArray(weights["$(prefix).weight"])
    s = MtlArray(Float16.(reinterpret(Float16, weights["$(prefix).scales"])))
    b = MtlArray(Float16.(reinterpret(Float16, weights["$(prefix).biases"])))

    # Keep natural layout (packed_cols, O) = (embed_dim/8, vocab_size)
    w_packed = w_u32
    s_kept = s
    b_kept = b

    # Dequantize the full embedding table on CPU for direct lookup
    w_cpu = Array(w_packed)
    s_cpu = Array(s_kept)
    b_cpu = Array(b_kept)
    table_f32 = dequantize_cpu(w_cpu, s_cpu, b_cpu; bits=4, group_size=group_size)
    # dequantize_cpu returns (O, I) = (vocab_size, embed_dim) with new layout
    # Need (embed_dim, vocab_size) for our column-major embedding lookup
    table = MtlArray(Float16.(permutedims(table_f32, (2, 1))))

    return QuantizedEmbedding(w_packed, s_kept, b_kept, vocab_size, embed_dim, group_size, table)
end

function load_llama_model(model_dir::String)
    config = LlamaConfig(joinpath(model_dir, "config.json"))
    println("Loading Llama model: $(config.num_hidden_layers) layers, " *
            "hidden=$(config.hidden_size), heads=$(config.num_attention_heads)/" *
            "$(config.num_key_value_heads)")

    # Load safetensors
    st_path = joinpath(model_dir, "model.safetensors")
    println("Loading weights from $(basename(st_path))...")
    weights = load_safetensors(st_path; mmap_data=true)

    gs = config.quant_group_size

    # Embedding
    println("  Loading embedding...")
    embed = load_embedding(weights, "model.embed_tokens",
                          config.vocab_size, config.hidden_size, gs)

    # Layers
    layers = LlamaLayer[]
    for i in 0:config.num_hidden_layers-1
        print("  Loading layer $i/$(config.num_hidden_layers-1)\r")
        prefix = "model.layers.$i"

        input_ln = MtlArray(Float16.(reinterpret(Float16, weights["$prefix.input_layernorm.weight"])))
        post_ln = MtlArray(Float16.(reinterpret(Float16, weights["$prefix.post_attention_layernorm.weight"])))

        attn = LlamaAttention(
            load_quantized_linear(weights, "$prefix.self_attn.q_proj", gs),
            load_quantized_linear(weights, "$prefix.self_attn.k_proj", gs),
            load_quantized_linear(weights, "$prefix.self_attn.v_proj", gs),
            load_quantized_linear(weights, "$prefix.self_attn.o_proj", gs),
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )

        mlp = LlamaMLP(
            load_quantized_linear(weights, "$prefix.mlp.gate_proj", gs),
            load_quantized_linear(weights, "$prefix.mlp.up_proj", gs),
            load_quantized_linear(weights, "$prefix.mlp.down_proj", gs),
        )

        push!(layers, LlamaLayer(input_ln, attn, post_ln, mlp))
    end
    println("  Loaded $(config.num_hidden_layers) layers.        ")

    # Final norm
    norm = MtlArray(Float16.(reinterpret(Float16, weights["model.norm.weight"])))

    # lm_head — tied with embedding when tie_word_embeddings=true
    lm_head = if config.tie_word_embeddings
        nothing  # reuse embed weights
    else
        load_quantized_linear(weights, "lm_head", gs)
    end

    # RoPE tables
    max_seq = 4096  # precompute for up to 4096 positions (extend as needed)
    cos_table, sin_table = compute_rope_freqs(
        config.head_dim, max_seq;
        theta=config.rope_theta,
        scaling_config=config.rope_scaling)
    cos_mtl = MtlArray(cos_table)
    sin_mtl = MtlArray(sin_table)

    println("Model loaded successfully.")
    return LlamaModel(config, embed, layers, norm, lm_head, cos_mtl, sin_mtl)
end
