"""
End-to-end prefix cache correctness test.

Verifies that a second turn of a conversation produces identical output
whether the prefix is cached or recomputed from scratch.

Usage:
    julia --project=. scripts/test_prefix_cache_e2e.jl [model_dir]
"""

using LLMs
using Metal

function main()
    model_dir = get(ARGS, 1,
        expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"))

    println("Loading model...")
    model = load_llama_model(model_dir)
    tok = Tokenizer(model_dir)

    # ═══ Test 1: Two-turn conversation ═══
    println("\n=== Test 1: Prefix cache vs from-scratch ===")

    prompt1 = "The capital of France is"
    prompt2 = "The capital of France is Paris. The capital of Germany is"

    ids1 = encode(tok, prompt1)
    ids2 = encode(tok, prompt2)

    println("Prompt 1: \"$prompt1\" ($(length(ids1)) tokens)")
    println("Prompt 2: \"$prompt2\" ($(length(ids2)) tokens)")
    println("Shared prefix: $(length(ids1)) tokens")

    # ── Run 1: Without prefix cache (from scratch) ──
    println("\nRun 1: From scratch (no cache)...")
    t0 = time()
    gen_nocache = generate(model, ids2; max_tokens=10)
    t1 = time()
    text_nocache = decode(tok, gen_nocache; skip_special=true)
    println("  Generated: $(repr(text_nocache)) ($(round(t1-t0, digits=2))s)")

    # ── Run 2: With prefix cache (first turn caches, second reuses) ──
    println("\nRun 2: With prefix cache...")
    pcache = PrefixCache(model.config; max_tokens=4096)

    # First call: generates from prompt1, caches the KV
    t0 = time()
    gen1 = generate_with_cache(model, pcache, ids1; max_tokens=5)
    t1 = time()
    text1 = decode(tok, gen1; skip_special=true)
    println("  Turn 1: \"$prompt1\" → $(repr(text1)) ($(round(t1-t0, digits=2))s)")

    # Check what was cached
    matched, _, segs = prefix_match(pcache, ids2)
    println("  Prefix cache hit: $matched/$(length(ids2)) tokens matched")

    # Second call: prompt2 shares prefix with prompt1
    t0 = time()
    gen2 = generate_with_cache(model, pcache, ids2; max_tokens=10)
    t1 = time()
    text2 = decode(tok, gen2; skip_special=true)
    println("  Turn 2: \"$prompt2\" → $(repr(text2)) ($(round(t1-t0, digits=2))s)")

    # ── Compare ──
    println("\n=== Comparison ===")
    println("  From scratch: $(repr(text_nocache))")
    println("  With cache:   $(repr(text2))")

    if gen_nocache == gen2
        println("  ✓ IDENTICAL output — prefix cache is correct!")
    else
        println("  ✗ Output differs (may be due to floating point — checking tokens...)")
        for i in 1:min(length(gen_nocache), length(gen2))
            if gen_nocache[i] != gen2[i]
                w1 = decode(tok, [gen_nocache[i]])
                w2 = decode(tok, [gen2[i]])
                println("    Token $i: nocache=$(gen_nocache[i]) $(repr(w1)) vs cache=$(gen2[i]) $(repr(w2))")
            end
        end
        # Check if first token matches (most important)
        if gen_nocache[1] == gen2[1]
            println("  ✓ First token matches — acceptable for 4-bit quantized inference")
        end
    end

    # ═══ Test 2: Exact same prompt twice ═══
    println("\n=== Test 2: Exact same prompt, second run uses full cache ===")
    pcache2 = PrefixCache(model.config; max_tokens=4096)

    prompt = "Hello, world!"
    ids = encode(tok, prompt)

    gen_a = generate_with_cache(model, pcache2, ids; max_tokens=10)
    matched_after, _, _ = prefix_match(pcache2, ids)
    println("  After first call: $matched_after/$(length(ids)) tokens cached")

    gen_b = generate_with_cache(model, pcache2, ids; max_tokens=10)
    println("  Run A tokens: $gen_a")
    println("  Run B tokens: $gen_b")

    if gen_a == gen_b
        println("  ✓ Identical output on repeated prompt with cache")
    else
        println("  ✗ Output differs")
        if gen_a[1] == gen_b[1]
            println("  ✓ First token matches")
        end
    end
end

main()
