"""Benchmark: fused 28-layer MPSGraph vs per-layer MPSGraph vs MLX."""

using Metal
using Statistics

include(joinpath(@__DIR__, "..", "src", "metal_graph.jl"))
include(joinpath(@__DIR__, "..", "src", "graph_forward.jl"))
using .MetalGraphModule
using .GraphForward

const HIDDEN = 3072
const HEAD_DIM = 128
const N_Q_HEADS = 24
const N_KV_HEADS = 8
const INTERMEDIATE = 8192
const N_LAYERS = 28
const EPS = 1f-5

function bench_fused_forward(seq_len::Int; n_warmup=10, n_iter=50)
    q_dim = N_Q_HEADS * HEAD_DIM
    kv_dim = N_KV_HEADS * HEAD_DIM
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = HEAD_DIM ÷ 2

    println("\n=== Fused $N_LAYERS-Layer MPSGraph (seq=$seq_len) ===")
    println("Building graph ($(N_LAYERS) layers in single graph)...")
    t0 = time_ns()
    fwd = build_full_forward(HIDDEN, HEAD_DIM, N_Q_HEADS, N_KV_HEADS,
                              INTERMEDIATE, seq_len, N_LAYERS, EPS)
    build_time = (time_ns() - t0) / 1e9
    println("  Graph built in $(round(build_time, digits=2))s")

    # Allocate feeds
    x = MtlArray(randn(Float16, HIDDEN, seq_len) * Float16(0.02))
    cos_t = MtlArray(rand(Float16, half_hd, seq_len))
    sin_t = MtlArray(rand(Float16, half_hd, seq_len))

    feeds = Dict{GraphTensor, MtlArray}(
        fwd.x_ph => x,
        fwd.cos_ph => cos_t,
        fwd.sin_ph => sin_t,
        fwd.final_norm_w_ph => MtlArray(ones(Float16, HIDDEN, 1)),
    )

    # Per-layer weights
    for l in 1:N_LAYERS
        feeds[fwd.input_norm_ws[l]] = MtlArray(ones(Float16, HIDDEN, 1))
        feeds[fwd.w_qkvs[l]] = MtlArray(randn(Float16, qkv_dim, HIDDEN) * Float16(0.02))
        feeds[fwd.w_os[l]] = MtlArray(randn(Float16, HIDDEN, q_dim) * Float16(0.02))
        feeds[fwd.post_norm_ws[l]] = MtlArray(ones(Float16, HIDDEN, 1))
        feeds[fwd.w_gate_ups[l]] = MtlArray(randn(Float16, 2*INTERMEDIATE, HIDDEN) * Float16(0.02))
        feeds[fwd.w_downs[l]] = MtlArray(randn(Float16, HIDDEN, INTERMEDIATE) * Float16(0.02))
    end

    out_buf = MtlArray(zeros(Float16, HIDDEN, seq_len))
    outputs = Dict{GraphTensor, MtlArray}(fwd.out => out_buf)

    println("Warming up ($n_warmup iterations)...")
    for _ in 1:n_warmup
        execute_gpu!(fwd.compiled, feeds, outputs)
        Metal.synchronize()
    end

    println("Benchmarking ($n_iter iterations)...")
    times = Float64[]
    for _ in 1:n_iter
        t0 = time_ns()
        execute_gpu!(fwd.compiled, feeds, outputs)
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end

    med = median(times)
    mn = minimum(times)
    tok_per_s = seq_len / (med / 1000)
    println("  Median: $(round(med, digits=2)) ms ($(round(Int, tok_per_s)) tok/s)")
    println("  Min:    $(round(mn, digits=2)) ms")
    return med
end

println("Fused MPSGraph Full Forward Pass Benchmark")
println("Model: Llama-3.2-3B ($N_LAYERS layers)")
println("Hidden=$HIDDEN, Heads=$N_Q_HEADS/$N_KV_HEADS, Inter=$INTERMEDIATE")

results = Dict{Int, Float64}()
for seq in [8, 32, 64]
    results[seq] = bench_fused_forward(seq)
end

# Compare with per-layer numbers from previous bench
println("\n" * "="^60)
println("Summary (28-layer forward, median ms):")
println("| seq | Fused Graph | Per-Layer Graph* | MLX* |")
println("|-----|-------------|-----------------|------|")
for seq in [8, 32, 64]
    println("|  $seq | $(round(results[seq], digits=1)) ms | — | — |")
end
println("* Fill in from bench_graph_layer.jl and bench_mlx_layer.py")
