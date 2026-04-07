"""
Test generation with Llama-3.2-3B-Instruct-4bit.

Usage:
    julia --project=. scripts/generate.jl [model_dir] [prompt]
"""

using LLMs
using Metal

function main()
    model_dir = get(ARGS, 1,
        expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"))
    prompt = get(ARGS, 2, "The capital of France is")

    println("Loading model from: $model_dir")
    model = load_llama_model(model_dir)

    println("\nLoading tokenizer...")
    tok = Tokenizer(model_dir)

    println("\nEncoding prompt: \"$prompt\"")
    token_ids = encode(tok, prompt)
    println("Token IDs: $token_ids ($(length(token_ids)) tokens)")

    println("\nGenerating (greedy, max 50 tokens)...")
    t0 = time()
    generated_ids = generate(model, token_ids; max_tokens=50)
    t1 = time()

    generated_text = decode(tok, generated_ids; skip_special=true)
    total_tokens = length(generated_ids)
    elapsed = t1 - t0

    println("\n=== Generated ===")
    println("$prompt$generated_text")
    println("\n=== Stats ===")
    println("  Tokens generated: $total_tokens")
    println("  Time: $(round(elapsed, digits=2))s")
    println("  Speed: $(round(total_tokens / elapsed, digits=1)) tok/s")
end

main()
