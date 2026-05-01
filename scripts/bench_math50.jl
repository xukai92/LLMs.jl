#!/usr/bin/env julia

"""
MATH50 benchmark for LLMs.jl.

Runs the existing quantized Julia inference backend on the same 50 problems
used by scripts/bench_math50.py, verifies boxed answers, and writes JSONL
per-problem artifacts plus a summary JSON.

Usage:
    julia --project=. scripts/bench_math50.jl [--backend q4|q4-opt] [--model MODEL_DIR] [--max-tokens 512] [--limit 50]
"""

using Dates
using JSON3
using LLMs
using Metal

const DEFAULT_MODEL_DIR = expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
)

function extract_boxed(text::AbstractString)
    needle = "\\boxed{"
    idx = findlast(needle, text)
    idx === nothing && return nothing

    i = last(idx) + 1
    depth = 1
    start = i
    while i <= lastindex(text) && depth > 0
        c = text[i]
        if c == '{'
            depth += 1
        elseif c == '}'
            depth -= 1
        end
        i = nextind(text, i)
    end

    depth == 0 || return nothing
    return strip(text[start:prevind(text, prevind(text, i))])
end

function normalize_answer(s::AbstractString)
    out = strip(String(s))
    out = replace(out, r"^\$|\$$" => "")
    out = replace(out, r"^\\text\{(.*)\}$" => s"\1")
    out = replace(out, r"\s+" => " ")

    out = replace(out, r"\\frac\{([^{}]+)\}\{([^{}]+)\}" => function(m)
        num = tryparse(Float64, m.captures[1])
        den = tryparse(Float64, m.captures[2])
        if num !== nothing && den !== nothing && den != 0
            return string(num / den)
        end
        return "$(m.captures[1])/$(m.captures[2])"
    end)

    out = replace(out, r"\\left|\\right" => "")
    out = replace(out, "\\infty" => "inf", "\\pi" => "pi")
    out = replace(out, "\\," => "", "\\;" => "", "\\!" => "")
    out = replace(out, r"\\text\{[^}]*\}" => "")
    return strip(out)
end

function to_float(s::AbstractString)
    v = tryparse(Float64, String(s))
    v !== nothing && return v

    m = match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s)
    if m !== nothing
        num = tryparse(Float64, m.captures[1])
        den = tryparse(Float64, m.captures[2])
        if num !== nothing && den !== nothing && den != 0
            return num / den
        end
    end
    return nothing
end

function answers_match(predicted::AbstractString, reference::AbstractString)
    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    pred == ref && return true

    pred_val = to_float(pred)
    ref_val = to_float(ref)
    return pred_val !== nothing && ref_val !== nothing && abs(pred_val - ref_val) < 1e-6
end

function parse_args(args)
    model_dir = DEFAULT_MODEL_DIR
    backend = "q4-opt"
    max_tokens = 512
    limit = nothing
    out_dir = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--model"
            i += 1
            i <= length(args) || error("--model requires a value")
            model_dir = args[i]
        elseif arg == "--backend"
            i += 1
            i <= length(args) || error("--backend requires a value")
            backend = args[i]
            backend in ("q4", "q4-opt") || error("--backend must be q4 or q4-opt")
        elseif arg == "--max-tokens"
            i += 1
            i <= length(args) || error("--max-tokens requires a value")
            max_tokens = parse(Int, args[i])
        elseif arg == "--limit"
            i += 1
            i <= length(args) || error("--limit requires a value")
            limit = parse(Int, args[i])
        elseif arg == "--out-dir"
            i += 1
            i <= length(args) || error("--out-dir requires a value")
            out_dir = args[i]
        elseif arg in ("-h", "--help")
            println("""
            Usage:
                julia --project=. scripts/bench_math50.jl [options]

            Options:
                --backend NAME         Backend: q4-opt or q4 (default: q4-opt)
                --model MODEL_DIR      Local model directory
                --max-tokens N         Maximum generated tokens per problem (default: 512)
                --limit N              Run only the first N problems
                --out-dir DIR          Output directory (default: results/)
            """)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    return model_dir, backend, max_tokens, limit, out_dir
end

function timestamp_slug()
    return Dates.format(now(), "yyyymmdd-HHMMSS")
end

function clip_for_log(s::AbstractString, n::Int=30)
    chars = collect(s)
    return join(chars[1:min(length(chars), n)])
end

function write_json_line(io, obj)
    write(io, JSON3.write(obj))
    write(io, '\n')
    flush(io)
end

function generate_backend(model, prompt_ids; backend::String, max_tokens::Int)
    if backend == "q4-opt"
        return generate_opt(model, prompt_ids; max_tokens=max_tokens)
    elseif backend == "q4"
        return generate(model, prompt_ids; max_tokens=max_tokens)
    else
        error("Unknown backend: $backend")
    end
end

function main()
    model_dir, backend, max_tokens, limit, out_dir = parse_args(ARGS)

    repo_root = dirname(@__DIR__)
    data_path = joinpath(repo_root, "data", "math50.json")
    results_dir = out_dir === nothing ? joinpath(repo_root, "results") : out_dir
    mkpath(results_dir)

    problems_all = JSON3.read(read(data_path, String))
    n_problems = limit === nothing ? length(problems_all) : min(limit, length(problems_all))
    problems = problems_all[1:n_problems]

    run_id = "math50-llmsjl-$(backend)-" * timestamp_slug()
    jsonl_path = joinpath(results_dir, run_id * ".jsonl")
    summary_path = joinpath(results_dir, run_id * ".summary.json")

    println("Loaded $(length(problems)) problems from MATH50")
    println("Backend: LLMs.jl $backend")
    println("Model: $model_dir")
    println("Max tokens: $max_tokens")
    println("Results: $jsonl_path")

    println("\nLoading model...")
    model = load_llama_model(model_dir)

    println("\nLoading tokenizer...")
    tokenizer = Tokenizer(model_dir)

    # Compile/warm up the safe current backend before measuring problem timings.
    println("\nWarmup...")
    warm_ids = encode(tokenizer, "Hello"; add_special=true)
    _ = generate_backend(model, warm_ids; backend=backend, max_tokens=2)
    Metal.synchronize()

    correct = 0
    total_generated_tokens = 0
    total_generation_time = 0.0
    bench_start = time()

    open(jsonl_path, "w") do io
        for (i, p) in enumerate(problems)
            problem = String(p.problem)
            reference = String(p.answer)
            prompt_ids, prompt_text = encode_chat(tokenizer, [
                ("user", "Solve the following math problem. Show your work and put your final answer in \\boxed{}.\n\n" * problem)
            ])

            t0 = time()
            generated_ids = generate_backend(model, prompt_ids; backend=backend, max_tokens=max_tokens)
            Metal.synchronize()
            elapsed = time() - t0

            output = decode(tokenizer, generated_ids; skip_special=true)
            predicted = extract_boxed(output)
            is_correct = predicted !== nothing && answers_match(predicted, reference)
            correct += is_correct ? 1 : 0

            n_generated = length(generated_ids)
            total_generated_tokens += n_generated
            total_generation_time += elapsed
            tok_s = elapsed > 0 ? n_generated / elapsed : 0.0

            wall = time() - bench_start
            eta = wall / i * (length(problems) - i)
            status = is_correct ? "PASS" : "FAIL"
            pred_str = predicted === nothing ? "(no \\boxed)" : predicted

            println("  [$i/$(length(problems))] $status ($(n_generated) tok, $(round(tok_s, digits=1)) tok/s)" *
                    " | pred=$(clip_for_log(pred_str))" *
                    " | ref=$(clip_for_log(reference))" *
                    " | wall=$(round(Int, wall))s eta=$(round(Int, eta))s")

            write_json_line(io, Dict{String, Any}(
                "run_id" => run_id,
                "index" => i,
                "backend" => "llmsjl-$backend",
                "model" => model_dir,
                "subject" => hasproperty(p, :subject) ? String(p.subject) : nothing,
                "level" => hasproperty(p, :level) ? Int(p.level) : nothing,
                "prompt_tokens" => length(prompt_ids),
                "generated_tokens" => n_generated,
                "elapsed_sec" => elapsed,
                "tok_s" => tok_s,
                "predicted" => predicted,
                "reference" => reference,
                "correct" => is_correct,
                "problem" => problem,
                "prompt" => prompt_text,
                "output" => output,
            ))
        end
    end

    total = length(problems)
    accuracy = total > 0 ? correct / total : 0.0
    avg_tok_s = total_generation_time > 0 ? total_generated_tokens / total_generation_time : 0.0
    wall_total = time() - bench_start

    summary = Dict{String, Any}(
        "run_id" => run_id,
        "backend" => "llmsjl-$backend",
        "model" => model_dir,
        "max_tokens" => max_tokens,
        "problems" => total,
        "correct" => correct,
        "accuracy" => accuracy,
        "total_generated_tokens" => total_generated_tokens,
        "generation_time_sec" => total_generation_time,
        "wall_time_sec" => wall_total,
        "avg_tok_s" => avg_tok_s,
        "jsonl_path" => jsonl_path,
        "created_at" => string(now()),
    )

    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
        write(io, '\n')
    end

    println()
    println("=" ^ 70)
    println("Results:")
    println("  Accuracy:   $correct / $total ($(round(accuracy * 100, digits=1))%)")
    println("  Throughput: $(round(avg_tok_s, digits=1)) tok/s avg")
    println("  Total:      $total_generated_tokens tokens in $(round(wall_total, digits=1))s wall")
    println("  JSONL:      $jsonl_path")
    println("  Summary:    $summary_path")
    println("=" ^ 70)
end

main()
