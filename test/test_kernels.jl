using Test
using Metal
using LLMs

"""Helper: compute max relative error between two arrays."""
function max_rel_error(a, b; eps=1e-7)
    max_err = 0.0
    for (x, y) in zip(vec(a), vec(b))
        denom = max(abs(Float64(x)), abs(Float64(y)), eps)
        max_err = max(max_err, abs(Float64(x) - Float64(y)) / denom)
    end
    return max_err
end

@testset "Phase 1: Metal Kernels" begin

    # ═══════════════════════════════════════════
    @testset "RMSNorm" begin
        @testset "Small Float32" begin
            hidden, batch = 64, 4
            x = randn(Float32, hidden, batch)
            w = rand(Float32, hidden) .+ 0.5f0
            eps = 1.0f-5

            out_cpu = similar(x)
            rmsnorm_cpu!(out_cpu, x, w, eps)

            x_gpu = MtlArray(x)
            w_gpu = MtlArray(w)
            out_gpu = MtlArray(zeros(Float32, hidden, batch))
            metal_rmsnorm!(out_gpu, x_gpu, w_gpu, eps)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-5
        end

        @testset "Llama-sized Float16" begin
            hidden, batch = 3072, 8
            x = Float16.(randn(Float32, hidden, batch) * 0.1f0)
            w = Float16.(rand(Float32, hidden) .+ 0.5f0)
            eps = 1.0f-5

            out_cpu = similar(x)
            rmsnorm_cpu!(out_cpu, x, w, eps)

            x_gpu = MtlArray(x)
            w_gpu = MtlArray(w)
            out_gpu = MtlArray(zeros(Float16, hidden, batch))
            metal_rmsnorm!(out_gpu, x_gpu, w_gpu, eps)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-2  # fp16 tolerance
        end
    end

    # ═══════════════════════════════════════════
    @testset "RoPE" begin
        @testset "Basic rotation" begin
            head_dim, n_heads, seq_len = 128, 4, 16
            cos_table, sin_table = compute_rope_freqs(head_dim, seq_len + 10)

            x_cpu = randn(Float32, head_dim, n_heads, seq_len)
            x_gpu_data = copy(x_cpu)

            rope_cpu!(x_cpu, cos_table, sin_table, 1)

            x_mtl = MtlArray(x_gpu_data)
            cos_mtl = MtlArray(cos_table)
            sin_mtl = MtlArray(sin_table)
            metal_rope!(x_mtl, cos_mtl, sin_mtl, 1)

            err = max_rel_error(x_cpu, Array(x_mtl))
            @test err < 1e-5
        end

        @testset "With offset (KV cache scenario)" begin
            head_dim, n_heads, seq_len = 128, 4, 1
            start_pos = 50
            cos_table, sin_table = compute_rope_freqs(head_dim, 100)

            x_cpu = randn(Float32, head_dim, n_heads, seq_len)
            x_gpu_data = copy(x_cpu)

            rope_cpu!(x_cpu, cos_table, sin_table, start_pos)

            x_mtl = MtlArray(x_gpu_data)
            cos_mtl = MtlArray(cos_table)
            sin_mtl = MtlArray(sin_table)
            metal_rope!(x_mtl, cos_mtl, sin_mtl, start_pos)

            err = max_rel_error(x_cpu, Array(x_mtl))
            @test err < 1e-5
        end

        @testset "Float16" begin
            head_dim, n_heads, seq_len = 128, 24, 8
            cos_table, sin_table = compute_rope_freqs(head_dim, 32)

            x_cpu = Float16.(randn(Float32, head_dim, n_heads, seq_len) * 0.1f0)
            x_gpu_data = copy(x_cpu)

            rope_cpu!(x_cpu, cos_table, sin_table, 1)

            x_mtl = MtlArray(x_gpu_data)
            cos_mtl = MtlArray(cos_table)
            sin_mtl = MtlArray(sin_table)
            metal_rope!(x_mtl, cos_mtl, sin_mtl, 1)

            err = max_rel_error(x_cpu, Array(x_mtl))
            @test err < 1e-2
        end

        @testset "Llama3 frequency scaling" begin
            scaling = Dict(
                "factor" => 32.0,
                "high_freq_factor" => 4.0,
                "low_freq_factor" => 1.0,
                "original_max_position_embeddings" => 8192
            )
            cos_table, sin_table = compute_rope_freqs(
                128, 100; theta=500000.0, scaling_config=scaling)
            @test size(cos_table) == (64, 100)
            @test size(sin_table) == (64, 100)
            # Verify at position 0 all cos=1, sin=0
            @test all(abs.(cos_table[:, 1] .- 1.0f0) .< 1e-6)
            @test all(abs.(sin_table[:, 1]) .< 1e-6)
        end
    end

    # ═══════════════════════════════════════════
    @testset "SwiGLU" begin
        @testset "Float32" begin
            n = 8192
            gate = randn(Float32, n)
            up = randn(Float32, n)

            out_cpu = similar(gate)
            swiglu_cpu!(out_cpu, gate, up)

            gate_gpu = MtlArray(gate)
            up_gpu = MtlArray(up)
            out_gpu = MtlArray(zeros(Float32, n))
            metal_swiglu!(out_gpu, gate_gpu, up_gpu)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-5
        end

        @testset "Float16 batched" begin
            size_ff, batch = 8192, 4
            gate = Float16.(randn(Float32, size_ff, batch) * 0.5f0)
            up = Float16.(randn(Float32, size_ff, batch) * 0.5f0)

            out_cpu = similar(gate)
            swiglu_cpu!(out_cpu, gate, up)

            out_gpu = MtlArray(zeros(Float16, size_ff, batch))
            metal_swiglu!(out_gpu, MtlArray(gate), MtlArray(up))

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-2
        end
    end

    # ═══════════════════════════════════════════
    @testset "Softmax" begin
        @testset "Small exact" begin
            x = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]  # 2 rows, 3 cols
            out_cpu = similar(x)
            softmax_cpu!(out_cpu, x)

            # Manual check: row 1 softmax
            e = exp.(Float32[1, 2, 3] .- 3)
            expected_row1 = e ./ sum(e)
            @test out_cpu[1, :] ≈ expected_row1 atol=1e-6

            out_gpu = MtlArray(zeros(Float32, 2, 3))
            metal_softmax!(out_gpu, MtlArray(x))

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-5
        end

        @testset "Large Float16" begin
            rows, cols = 24, 512  # n_heads rows, seq_len cols
            x = Float16.(randn(Float32, rows, cols))

            out_cpu = similar(x)
            softmax_cpu!(out_cpu, x)

            out_gpu = MtlArray(zeros(Float16, rows, cols))
            metal_softmax!(out_gpu, MtlArray(x))

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 5e-2  # fp16 softmax has some numerical drift
        end

        @testset "Rows sum to 1" begin
            rows, cols = 8, 256
            x = MtlArray(randn(Float32, rows, cols))
            out = MtlArray(zeros(Float32, rows, cols))
            metal_softmax!(out, x)
            result = Array(out)
            for r in 1:rows
                @test abs(sum(result[r, :]) - 1.0f0) < 1e-5
            end
        end
    end

    # ═══════════════════════════════════════════
    @testset "Attention (GQA)" begin
        @testset "No GQA (equal heads)" begin
            head_dim, n_heads, seq_len = 64, 4, 16
            scale = 1.0f0 / sqrt(Float32(head_dim))

            Q = randn(Float32, head_dim, n_heads, seq_len)
            K = randn(Float32, head_dim, n_heads, seq_len)
            V = randn(Float32, head_dim, n_heads, seq_len)

            out_cpu = zeros(Float32, head_dim, n_heads, seq_len)
            attention_cpu!(out_cpu, Q, K, V, scale)

            out_gpu = MtlArray(zeros(Float32, head_dim, n_heads, seq_len))
            metal_attention!(out_gpu, MtlArray(Q), MtlArray(K), MtlArray(V), scale;
                            causal=false)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 5e-3  # fp32 softmax + matmul accumulation
        end

        @testset "GQA ratio=3 (Llama-style)" begin
            head_dim = 128
            n_q_heads, n_kv_heads = 24, 8
            seq_q, seq_kv = 4, 32
            scale = 1.0f0 / sqrt(Float32(head_dim))

            Q = randn(Float32, head_dim, n_q_heads, seq_q)
            K = randn(Float32, head_dim, n_kv_heads, seq_kv)
            V = randn(Float32, head_dim, n_kv_heads, seq_kv)

            out_cpu = zeros(Float32, head_dim, n_q_heads, seq_q)
            attention_cpu!(out_cpu, Q, K, V, scale)

            out_gpu = MtlArray(zeros(Float32, head_dim, n_q_heads, seq_q))
            metal_attention!(out_gpu, MtlArray(Q), MtlArray(K), MtlArray(V), scale;
                            causal=false)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 5e-3  # fp32 accumulation over 128 head_dim * 32 seq_kv
        end

        @testset "Causal masking" begin
            head_dim, n_heads = 64, 2
            seq_len = 8
            scale = 1.0f0 / sqrt(Float32(head_dim))

            Q = randn(Float32, head_dim, n_heads, seq_len)
            K = randn(Float32, head_dim, n_heads, seq_len)
            V = randn(Float32, head_dim, n_heads, seq_len)

            # Build causal mask: mask[sq, sk] = true means MASKED
            mask = falses(seq_len, seq_len)
            for sq in 1:seq_len, sk in 1:seq_len
                if sk > sq
                    mask[sq, sk] = true
                end
            end

            out_cpu = zeros(Float32, head_dim, n_heads, seq_len)
            attention_cpu!(out_cpu, Q, K, V, scale; mask=mask)

            # Metal attention uses built-in causal masking via causal_offset=0
            out_gpu = MtlArray(zeros(Float32, head_dim, n_heads, seq_len))
            metal_attention!(out_gpu, MtlArray(Q), MtlArray(K), MtlArray(V), scale;
                            causal_offset=0)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-4
        end

        @testset "Float16" begin
            head_dim, n_heads, seq_len = 128, 8, 16
            scale = 1.0f0 / sqrt(Float32(head_dim))

            Q = Float16.(randn(Float32, head_dim, n_heads, seq_len) * 0.1f0)
            K = Float16.(randn(Float32, head_dim, n_heads, seq_len) * 0.1f0)
            V = Float16.(randn(Float32, head_dim, n_heads, seq_len) * 0.1f0)

            out_cpu = zeros(Float16, head_dim, n_heads, seq_len)
            attention_cpu!(out_cpu, Q, K, V, scale)

            out_gpu = MtlArray(zeros(Float16, head_dim, n_heads, seq_len))
            metal_attention!(out_gpu, MtlArray(Q), MtlArray(K), MtlArray(V), scale;
                            causal=false)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 5e-2  # fp16 attention accumulates error
        end
    end

    # ═══════════════════════════════════════════
    @testset "Quantized Matmul" begin
        @testset "Dequantize correctness" begin
            # Create a simple quantized weight: 2 rows, 16 cols (2 packed cols)
            # group_size=8 for this small test
            packed = UInt32[
                0x76543210 0xFEDCBA98;  # row 1: values 0,1,2,...,F
            ]  # shape: (1, 2) — but Julia is col-major so we handle carefully

            # Actually let's be more careful with the layout.
            # packed[row, packed_col]
            # For row=1, packed_col=1: 0x76543210 → unpacks to [0,1,2,3,4,5,6,7]
            # For row=1, packed_col=2: 0xFEDCBA98 → unpacks to [8,9,10,11,12,13,14,15]
            packed = reshape(UInt32[0x76543210, 0xFEDCBA98], 1, 2)
            scales = reshape(Float16[1.0, 1.0], 1, 2)
            biases = reshape(Float16[0.0, 0.0], 1, 2)

            W = dequantize_cpu(packed, scales, biases; bits=4, group_size=8)
            @test size(W) == (1, 16)
            @test W[1, 1:8] ≈ Float32[0, 1, 2, 3, 4, 5, 6, 7]
            @test W[1, 9:16] ≈ Float32[8, 9, 10, 11, 12, 13, 14, 15]
        end

        @testset "Dequantize with scale and bias" begin
            packed = reshape(UInt32[0x76543210], 1, 1)
            scales = reshape(Float16[2.0], 1, 1)
            biases = reshape(Float16[0.5], 1, 1)

            W = dequantize_cpu(packed, scales, biases; bits=4, group_size=8)
            expected = Float32[2.0 * i + 0.5 for i in 0:7]
            @test W[1, :] ≈ expected
        end

        @testset "Matmul CPU vs Metal - small" begin
            O, I, B = 4, 64, 2
            group_size = 64

            # Create random quantized weights
            packed_cols = I ÷ 8
            n_groups = I ÷ group_size
            packed = rand(UInt32, O, packed_cols)
            scales = Float16.(rand(Float32, O, n_groups) * 0.1f0)
            biases = Float16.(rand(Float32, O, n_groups) * 0.01f0 .- 0.005f0)
            x = Float16.(randn(Float32, I, B) * 0.1f0)

            out_cpu = zeros(Float16, O, B)
            quantized_matmul_cpu!(out_cpu, x, packed, scales, biases;
                                  bits=4, group_size=group_size)

            out_gpu = MtlArray(zeros(Float16, O, B))
            metal_quantized_matmul!(out_gpu, MtlArray(x),
                                    MtlArray(packed), MtlArray(scales), MtlArray(biases);
                                    group_size=group_size)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 1e-2
        end

        @testset "Matmul CPU vs Metal - Llama q_proj sized" begin
            O, I, B = 3072, 3072, 1
            group_size = 64

            packed_cols = I ÷ 8
            n_groups = I ÷ group_size
            packed = rand(UInt32, O, packed_cols)
            scales = Float16.(rand(Float32, O, n_groups) * 0.1f0)
            biases = Float16.(rand(Float32, O, n_groups) * 0.01f0 .- 0.005f0)
            x = Float16.(randn(Float32, I, B) * 0.1f0)

            out_cpu = zeros(Float16, O, B)
            quantized_matmul_cpu!(out_cpu, x, packed, scales, biases;
                                  bits=4, group_size=group_size)

            out_gpu = MtlArray(zeros(Float16, O, B))
            metal_quantized_matmul!(out_gpu, MtlArray(x),
                                    MtlArray(packed), MtlArray(scales), MtlArray(biases);
                                    group_size=group_size)

            err = max_rel_error(out_cpu, Array(out_gpu))
            @test err < 5e-2  # fp16 accumulation over 3072 elements
        end
    end
end
