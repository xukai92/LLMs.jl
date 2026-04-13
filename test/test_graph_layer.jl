"""Test full MPSGraph transformer layer against CPU reference."""

using Metal
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "metal_graph.jl"))
include(joinpath(@__DIR__, "..", "src", "graph_forward.jl"))
using .MetalGraphModule
using .GraphForward

# ── CPU reference implementations ──

function cpu_rmsnorm(x::Matrix{Float32}, w::Vector{Float32}, eps::Float32)
    ms = sum(x .^ 2, dims=1) ./ size(x, 1)
    inv = 1.0f0 ./ sqrt.(ms .+ eps)
    x .* inv .* w
end

function cpu_rope(x::Array{Float32, 3}, cos_t::Matrix{Float32}, sin_t::Matrix{Float32})
    hd, nh, seq = size(x)
    half = hd ÷ 2
    out = similar(x)
    for s in 1:seq, h in 1:nh
        for d in 1:half
            lo = x[d, h, s]; hi = x[half+d, h, s]
            c = cos_t[d, s]; sn = sin_t[d, s]
            out[d, h, s] = lo * c - hi * sn
            out[half+d, h, s] = hi * c + lo * sn
        end
    end
    out
end

function cpu_attention(q::Array{Float32,3}, k::Array{Float32,3}, v::Array{Float32,3},
                       scale::Float32, n_q::Int, n_kv::Int)
    hd, _, seq = size(q)
    gqa = n_q ÷ n_kv
    out = zeros(Float32, hd, n_q, seq)

    for qh in 1:n_q
        kvh = (qh - 1) ÷ gqa + 1
        # scores: seq × seq
        scores = zeros(Float32, seq, seq)
        for i in 1:seq, j in 1:seq
            for d in 1:hd
                scores[j, i] += q[d, qh, i] * k[d, kvh, j]
            end
            scores[j, i] *= scale
        end
        # Causal mask
        for i in 1:seq, j in 1:seq
            if j > i; scores[j, i] = -1f4; end
        end
        # Softmax along dim 1 (j dimension)
        for i in 1:seq
            mx = maximum(scores[:, i])
            scores[:, i] .= exp.(scores[:, i] .- mx)
            scores[:, i] ./= sum(scores[:, i])
        end
        # Value aggregation
        for i in 1:seq, d in 1:hd
            for j in 1:seq
                out[d, qh, i] += v[d, kvh, j] * scores[j, i]
            end
        end
    end
    out
end

function cpu_transformer_layer(x_normed, residual, w_qkv, w_o, cos_t, sin_t,
                                post_norm_w, w_gu, w_down, config)
    hidden = config.hidden; hd = config.head_dim
    n_q = config.n_q; n_kv = config.n_kv; inter = config.intermediate
    eps = config.eps; scale = config.scale
    q_dim = n_q * hd; kv_dim = n_kv * hd
    seq = size(x_normed, 2)

    # QKV
    qkv = w_qkv * x_normed
    q_flat = qkv[1:q_dim, :]
    k_flat = qkv[q_dim+1:q_dim+kv_dim, :]
    v_flat = qkv[q_dim+kv_dim+1:q_dim+2*kv_dim, :]

    q_3d = reshape(q_flat, hd, n_q, seq)
    k_3d = reshape(k_flat, hd, n_kv, seq)
    v_3d = reshape(v_flat, hd, n_kv, seq)

    # RoPE
    q_roped = cpu_rope(q_3d, cos_t, sin_t)
    k_roped = cpu_rope(k_3d, cos_t, sin_t)

    # Attention
    attn_out = cpu_attention(q_roped, k_roped, v_3d, scale, n_q, n_kv)
    attn_flat = reshape(attn_out, q_dim, seq)

    # O proj + residual
    o_out = w_o * attn_flat
    attn_res = residual .+ o_out

    # Post-attention norm + MLP
    normed_mlp = cpu_rmsnorm(attn_res, post_norm_w, eps)
    gu = w_gu * normed_mlp
    gate = gu[1:inter, :]
    up = gu[inter+1:2*inter, :]
    swi = gate .* (1 ./ (1 .+ exp.(-gate)))
    mlp_out = w_down * (swi .* up)

    out_res = attn_res .+ mlp_out
    return out_res
end

# ── Test ──

function test_transformer_layer()
    # Use small dims for testing
    hidden = 256; head_dim = 64; n_q = 4; n_kv = 2; intermediate = 512
    seq_len = 8; eps = 1f-5
    gqa = n_q ÷ n_kv
    q_dim = n_q * head_dim; kv_dim = n_kv * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    scale = Float32(1.0 / sqrt(Float64(head_dim)))

    config = (hidden=hidden, head_dim=head_dim, n_q=n_q, n_kv=n_kv,
              intermediate=intermediate, eps=eps, scale=scale)

    # Random data (Float32 for CPU ref, Float16 for GPU)
    x_normed_f32 = randn(Float32, hidden, seq_len) * 0.1f0
    residual_f32 = randn(Float32, hidden, seq_len) * 0.1f0
    w_qkv_f32 = randn(Float32, qkv_dim, hidden) * Float32(0.02)
    w_o_f32 = randn(Float32, hidden, q_dim) * Float32(0.02)
    cos_f32 = cos.(randn(Float32, half_hd, seq_len))
    sin_f32 = sin.(randn(Float32, half_hd, seq_len))
    post_norm_w_f32 = ones(Float32, hidden)
    w_gu_f32 = randn(Float32, 2*intermediate, hidden) * Float32(0.02)
    w_down_f32 = randn(Float32, hidden, intermediate) * Float32(0.02)

    # CPU reference
    ref = cpu_transformer_layer(x_normed_f32, residual_f32, w_qkv_f32, w_o_f32,
                                cos_f32, sin_f32, post_norm_w_f32, w_gu_f32, w_down_f32, config)

    # GPU graph
    println("Building transformer layer graph...")
    layer = build_transformer_layer(hidden, head_dim, n_q, n_kv, intermediate,
                                     seq_len, eps; emit_normed=false)

    # Execute
    feeds = Dict{GraphTensor, MtlArray}(
        layer.x_ph => MtlArray(Float16.(x_normed_f32)),
        layer.residual_ph => MtlArray(Float16.(residual_f32)),
        layer.w_qkv_ph => MtlArray(Float16.(w_qkv_f32)),
        layer.w_o_ph => MtlArray(Float16.(w_o_f32)),
        layer.cos_ph => MtlArray(Float16.(cos_f32)),
        layer.sin_ph => MtlArray(Float16.(sin_f32)),
        layer.post_norm_w_ph => MtlArray(Float16.(reshape(post_norm_w_f32, hidden, 1))),
        layer.w_gate_up_ph => MtlArray(Float16.(w_gu_f32)),
        layer.w_down_ph => MtlArray(Float16.(w_down_f32)),
    )

    println("Executing graph...")
    result = execute!(layer.compiled, feeds)

    gpu_out = result[layer.out_residual]
    ref16 = Float16.(ref)

    @testset "transformer layer" begin
        @test size(gpu_out) == size(ref16)

        # Per-element relative error
        err = maximum(abs.(Float32.(gpu_out) .- Float32.(ref16))) /
              (maximum(abs.(Float32.(ref16))) + 1f-8)
        println("  Max relative error: $(round(err, digits=5))")
        @test err < 0.1  # FP16 through full layer will accumulate error
    end
end

println("Testing full transformer layer graph...")
test_transformer_layer()
println("\nTransformer layer test complete!")
