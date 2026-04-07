"""
Inspect the weight layout of a quantized model from HuggingFace.

Usage:
    julia --project=. scripts/inspect_weights.jl [model_dir]

Default model_dir: ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-Instruct-4bit/snapshots/*/
"""

using JSON3
using LLMs

function find_model_dir(hint::Union{String, Nothing}=nothing)
    if hint !== nothing && isdir(hint)
        return hint
    end

    # Try common HuggingFace cache locations
    hf_cache = expanduser("~/.cache/huggingface/hub")
    model_name = "models--mlx-community--Qwen3.5-4B-Instruct-4bit"
    model_dir = joinpath(hf_cache, model_name)

    if isdir(model_dir)
        # Find the latest snapshot
        snapshots_dir = joinpath(model_dir, "snapshots")
        if isdir(snapshots_dir)
            snaps = readdir(snapshots_dir; join=true)
            if !isempty(snaps)
                return snaps[end]  # latest snapshot
            end
        end
    end

    error("""
    Model not found. Please download it first:
        pip install huggingface-hub
        huggingface-cli download mlx-community/Qwen3.5-4B-Instruct-4bit
    Or pass the model directory as an argument.
    """)
end

function inspect_config(model_dir::String)
    config_path = joinpath(model_dir, "config.json")
    if !isfile(config_path)
        println("⚠  No config.json found")
        return nothing
    end

    config = JSON3.read(read(config_path, String))
    println("═══ Model Config ═══")
    for key in [:model_type, :hidden_size, :intermediate_size, :num_hidden_layers,
                :num_attention_heads, :num_key_value_heads, :max_position_embeddings,
                :vocab_size, :rms_norm_eps, :rope_theta, :tie_word_embeddings,
                :head_dim, :quantization]
        if haskey(config, key)
            println("  $key: $(config[key])")
        end
    end

    # Check for quantization config
    if haskey(config, :quantization)
        println("\n═══ Quantization Config ═══")
        qconfig = config.quantization
        for (k, v) in pairs(qconfig)
            println("  $k: $v")
        end
    end

    return config
end

function inspect_safetensors(model_dir::String)
    # Find all safetensors files
    st_files = filter(f -> endswith(f, ".safetensors"), readdir(model_dir; join=true))

    if isempty(st_files)
        println("⚠  No .safetensors files found in $model_dir")
        return
    end

    println("\n═══ Safetensors Files ═══")
    for f in st_files
        println("  $(basename(f))  ($(round(filesize(f) / 1024^2, digits=1)) MB)")
    end

    # Inspect each file
    total_params = 0
    total_bytes = 0
    layer_patterns = Dict{String, Vector{Tuple{String, DataType, Vector{Int}}}}()

    for st_file in st_files
        println("\n═══ $(basename(st_file)) ═══")
        infos, _ = load_safetensors_lazy(st_file)

        for info in sort(infos; by=x -> x.name)
            nbytes = info.data_offset_end - info.data_offset_start
            nparams = prod(info.shape)
            total_bytes += nbytes
            total_params += nparams

            # Categorize: extract layer number and component
            m = match(r"model\.layers\.(\d+)\.(.*)", info.name)
            if m !== nothing
                layer_num = parse(Int, m.captures[1])
                component = m.captures[2]
                key = component
                if !haskey(layer_patterns, key)
                    layer_patterns[key] = []
                end
                push!(layer_patterns[key], (info.name, info.dtype, info.shape))
            end

            println("  $(rpad(info.name, 60)) $(rpad(string(info.dtype), 8)) $(info.shape)  ($(round(nbytes/1024, digits=1)) KB)")
        end
    end

    println("\n═══ Summary ═══")
    println("  Total tensors: counted above")
    println("  Total bytes: $(round(total_bytes / 1024^2, digits=1)) MB")
    println("  Total params (elements): $(total_params)")

    # Show unique layer patterns (from layer 0 as representative)
    if !isempty(layer_patterns)
        println("\n═══ Per-Layer Weight Pattern (from layer 0) ═══")
        for (component, entries) in sort(collect(layer_patterns); by=first)
            # Find layer 0 entry
            l0 = filter(e -> startswith(e[1], "model.layers.0."), entries)
            if !isempty(l0)
                name, dtype, shape = l0[1]
                println("  $(rpad(component, 50)) $dtype  $shape")
            end
        end
    end

    # Check for quantized weight patterns
    println("\n═══ Quantization Pattern Check ═══")
    has_quantized = false
    for (component, _) in layer_patterns
        if endswith(component, ".weight") && haskey(layer_patterns, replace(component, ".weight" => ".scales"))
            has_quantized = true
            w_entries = filter(e -> startswith(e[1], "model.layers.0."), layer_patterns[component])
            s_entries = filter(e -> startswith(e[1], "model.layers.0."), layer_patterns[replace(component, ".weight" => ".scales")])
            b_entries = filter(e -> startswith(e[1], "model.layers.0."), layer_patterns[replace(component, ".weight" => ".biases")])

            if !isempty(w_entries) && !isempty(s_entries)
                w = w_entries[1]
                s = s_entries[1]
                b_shape = isempty(b_entries) ? "N/A" : string(b_entries[1][3])
                println("  $(replace(component, "model.layers.0." => ""))")
                println("    weight: $(w[2]) $(w[3])")
                println("    scales: $(s[2]) $(s[3])")
                println("    biases: $b_shape")

                # Infer quantization params
                if w[2] == UInt32 && length(w[3]) == 2 && length(s[3]) == 2
                    O, packed_cols = w[3]
                    _, n_groups = s[3]
                    bits = 4  # assume 4-bit
                    elems_per_u32 = 32 ÷ bits
                    I_inferred = packed_cols * elems_per_u32
                    group_size = I_inferred ÷ n_groups
                    println("    → Inferred: out=$O, in=$I_inferred, bits=$bits, group_size=$group_size")
                end
            end
        end
    end

    if !has_quantized
        println("  No quantized weight pattern (weight/scales/biases triplet) detected")
    end
end

function main()
    model_dir = length(ARGS) >= 1 ? ARGS[1] : nothing
    model_dir = find_model_dir(model_dir)
    println("Model directory: $model_dir\n")

    inspect_config(model_dir)
    inspect_safetensors(model_dir)
end

main()
