"""
    graph_forward.jl — Full transformer layer as a single MPSGraph

Builds the entire forward pass (attention + MLP) for a transformer layer
as a fused computation graph, eliminating intermediate memory round-trips.
"""
module GraphForward

using Metal
using ..MetalGraphModule

# ── Graph-based MLP block ──

struct GraphMLP
    """Pre-compiled MPSGraph for: rmsnorm → gate_up matmul → silu(gate)*up → down matmul → residual add"""
    compiled::CompiledGraph

    # Placeholder handles for feeding data
    x_ph::GraphTensor       # input hidden state (hidden, seq)
    residual_ph::GraphTensor # residual stream (hidden, seq)
    norm_w_ph::GraphTensor  # rmsnorm weight (hidden, 1)
    w_gate_up_ph::GraphTensor # fused gate+up weight (2*inter, hidden)
    w_down_ph::GraphTensor  # down projection weight (hidden, inter)

    # Output handle
    out::GraphTensor        # residual + down(silu(gate(normed)) * up(normed))
    normed_out::GraphTensor # optional: normed output (for next layer's attention)

    hidden::Int
    intermediate::Int
end

"""Build MLP graph for given dimensions."""
function build_mlp_graph(hidden::Int, intermediate::Int, seq_len::Int, eps::Float32;
                         fuse_next_rmsnorm::Bool=false, next_norm_w_ph=nothing)
    g = MetalGraphBuilder()

    # Placeholders
    x_ph = placeholder!(g, (hidden, seq_len), Float16)
    residual_ph = placeholder!(g, (hidden, seq_len), Float16)
    norm_w_ph = placeholder!(g, (hidden, 1), Float16)
    w_gu_ph = placeholder!(g, (2 * intermediate, hidden), Float16)
    w_down_ph = placeholder!(g, (hidden, intermediate), Float16)

    # RMSNorm
    normed = rmsnorm!(g, x_ph, norm_w_ph, eps)

    # Fused gate+up matmul
    gu = matmul!(g, w_gu_ph, normed)

    # Split gate and up
    gate = slice!(g, gu, 1, 0, intermediate)
    up = slice!(g, gu, 1, intermediate, intermediate)

    # SwiGLU: silu(gate) * up
    swi = silu!(g, gate)
    fused = mul!(g, swi, up)

    # Down projection
    down_out = matmul!(g, w_down_ph, fused)

    # Residual add: out = residual + down_out
    out = add!(g, residual_ph, down_out)

    # Optionally fuse the next layer's input rmsnorm
    normed_out_tensor = if fuse_next_rmsnorm && next_norm_w_ph !== nothing
        rmsnorm!(g, out, next_norm_w_ph, eps)
    else
        out  # dummy — won't be used
    end

    targets = fuse_next_rmsnorm ? [out, normed_out_tensor] : [out]
    compiled = compile!(g, targets)

    GraphMLP(compiled, x_ph, residual_ph, norm_w_ph, w_gu_ph, w_down_ph,
             out, normed_out_tensor, hidden, intermediate)
end

"""Execute MLP graph with pre-allocated GPU buffers."""
function execute_mlp!(mlp::GraphMLP, x::MtlMatrix{Float16}, residual::MtlMatrix{Float16},
                      norm_w::MtlMatrix{Float16}, w_gate_up::MtlMatrix{Float16},
                      w_down::MtlMatrix{Float16}, out::MtlMatrix{Float16};
                      normed_out::Union{MtlMatrix{Float16}, Nothing}=nothing)
    feeds = Dict{GraphTensor, MtlArray}(
        mlp.x_ph => x,
        mlp.residual_ph => residual,
        mlp.norm_w_ph => norm_w,
        mlp.w_gate_up_ph => w_gate_up,
        mlp.w_down_ph => w_down,
    )
    outputs = Dict{GraphTensor, MtlArray}(mlp.out => out)
    if normed_out !== nothing
        outputs[mlp.normed_out] = normed_out
    end
    execute_gpu!(mlp.compiled, feeds, outputs)
end


# ── Graph-based Attention block ──

struct GraphAttention
    """Pre-compiled MPSGraph for: QKV matmul → split → rope → attention → O proj"""
    compiled::CompiledGraph

    # Placeholders
    normed_ph::GraphTensor    # normed input (hidden, seq)
    w_qkv_ph::GraphTensor     # fused QKV weight (q+k+v, hidden)
    w_o_ph::GraphTensor       # output projection (hidden, q_dim)
    cos_ph::GraphTensor       # rope cos table (half_hd, seq)
    sin_ph::GraphTensor       # rope sin table (half_hd, seq)

    # Output
    out::GraphTensor          # attention output (hidden, seq)

    hidden::Int
    head_dim::Int
    n_q_heads::Int
    n_kv_heads::Int
end

"""Build attention graph for prefill (no KV cache, self-attention only).
For decode with KV cache, we'll need a separate graph or hybrid approach."""
function build_attention_graph(hidden::Int, head_dim::Int, n_q_heads::Int, n_kv_heads::Int,
                               seq_len::Int, eps::Float32)
    g = MetalGraphBuilder()
    q_dim = n_q_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    gqa_ratio = n_q_heads ÷ n_kv_heads
    scale = Float16(1.0 / sqrt(Float64(head_dim)))

    # Placeholders
    normed_ph = placeholder!(g, (hidden, seq_len), Float16)
    w_qkv_ph = placeholder!(g, (qkv_dim, hidden), Float16)
    w_o_ph = placeholder!(g, (hidden, q_dim), Float16)
    cos_ph = placeholder!(g, (half_hd, seq_len), Float16)
    sin_ph = placeholder!(g, (half_hd, seq_len), Float16)

    # QKV matmul
    qkv = matmul!(g, w_qkv_ph, normed_ph)  # (qkv_dim, seq)

    # Split Q, K, V
    q_flat = slice!(g, qkv, 1, 0, q_dim)           # (q_dim, seq)
    k_flat = slice!(g, qkv, 1, q_dim, kv_dim)      # (kv_dim, seq)
    v_flat = slice!(g, qkv, 1, q_dim + kv_dim, kv_dim)  # (kv_dim, seq)

    # Reshape to (head_dim, n_heads, seq) for RoPE and attention
    q_3d = reshape!(g, q_flat, (head_dim, n_q_heads, seq_len))
    k_3d = reshape!(g, k_flat, (head_dim, n_kv_heads, seq_len))
    v_3d = reshape!(g, v_flat, (head_dim, n_kv_heads, seq_len))

    # RoPE on Q and K
    # Need cos/sin as (half_hd, 1, seq) for broadcasting over heads
    cos_3d = reshape!(g, cos_ph, (half_hd, 1, seq_len))
    sin_3d = reshape!(g, sin_ph, (half_hd, 1, seq_len))

    q_roped = apply_rope!(g, q_3d, cos_3d, sin_3d, half_hd)
    k_roped = apply_rope!(g, k_3d, cos_3d, sin_3d, half_hd)

    # Attention: for each query head, find its KV head (GQA)
    # Reshape Q: (head_dim, n_kv_heads, gqa_ratio, seq) and K/V stay (head_dim, n_kv_heads, seq)
    # Then attention scores: Q^T @ K for each head group

    # For GQA: reshape Q to (head_dim, gqa_ratio, n_kv_heads, seq)
    # Then transpose to get batch dims right for batched matmul
    # scores[h,i,j] = sum_d Q[d,h,i] * K[d,h,j] / sqrt(d)

    # Approach: transpose to (n_kv_heads, head_dim, seq) for K and V
    # Then for each KV head, Q has gqa_ratio query heads

    # Simple approach: reshape Q to group by KV heads
    # Q: (head_dim, n_q_heads, seq) → (head_dim, gqa_ratio, n_kv_heads, seq)
    q_grouped = reshape!(g, q_roped, (head_dim, gqa_ratio, n_kv_heads, seq_len))

    # Transpose for batched matmul: we want scores = Q^T @ K per head
    # Q: (head_dim, gqa_ratio, n_kv_heads, seq) → (n_kv_heads, gqa_ratio, seq, head_dim) via transposes
    # K: (head_dim, n_kv_heads, seq) → (n_kv_heads, head_dim, seq)

    # Q: swap dim1↔dim3 to get (seq, gqa_ratio, n_kv_heads, head_dim)
    # then swap dim3↔dim4 nah this is getting complex. Let me think about MPSGraph's batched matmul.

    # MPSGraph matmul broadcasts over batch dims. For 3D tensors (B, M, K) @ (B, K, N) = (B, M, N)
    # Remember: we swap args for row/col major, so matmul!(g, a, b) = Julia a@b

    # K transposed: (n_kv_heads, seq, head_dim) — K^T in the (seq, head_dim) sense
    k_t = graph_transpose!(g, k_roped, 1, 3)  # (head_dim, n_kv, seq) → (seq, n_kv, head_dim)
    k_t = graph_transpose!(g, k_t, 2, 3)      # → (seq, head_dim, n_kv)
    # Hmm, this isn't right. Let me think step by step.

    # What we want:
    # For each kv_head h, for each gqa q_head within that group:
    #   scores[h, g, i, j] = sum_d Q[d, g, h, i] * K[d, h, j] / sqrt(d)
    # This is a batched matmul: (h*g, seq, head_dim) @ (h, head_dim, seq) with broadcasting

    # Let me use a simpler layout:
    # Merge gqa into batch: Q (head_dim, n_q_heads, seq) with n_q_heads = n_kv_heads * gqa_ratio
    # Repeat K for each gqa group: K (head_dim, n_q_heads, seq) — wasteful but simple

    # Actually, MPSGraph should handle broadcasting: if K is (head_dim, n_kv_heads, seq)
    # and we reshape Q as (head_dim, gqa_ratio, n_kv_heads, seq), the matmul should broadcast K
    # over the gqa_ratio dimension. But matmul only operates on the last 2 dims.

    # Let me try: Q as 4D (head_dim, gqa_ratio, n_kv_heads, seq)
    # For scores, we need (seq, seq) per head. So:
    # Q_t: (gqa_ratio, n_kv_heads, seq, head_dim) — transpose dim1↔dim4 of Q
    # K: (n_kv_heads, head_dim, seq) — need to broadcast over gqa_ratio

    # Simpler: just use n_q_heads as batch dim, tile K
    # Q: (n_q_heads, seq, head_dim) — from (head_dim, n_q_heads, seq) via transposes
    # K: (n_q_heads, head_dim, seq) — tile n_kv_heads → n_q_heads

    # Let's do this in the straightforward way:
    # Step 1: Rearrange Q to (n_q_heads, seq, head_dim)
    q_perm = graph_transpose!(g, q_roped, 1, 2)   # (head_dim, n_q, seq) → (n_q, head_dim, seq)
    q_perm = graph_transpose!(g, q_perm, 2, 3)     # → (n_q, seq, head_dim)

    # Step 2: Rearrange K to (n_kv_heads, head_dim, seq) — already (head_dim, n_kv, seq)
    # We need (n_q, head_dim, seq) by repeating KV heads gqa_ratio times
    # First transpose to (n_kv, head_dim, seq)
    k_perm = graph_transpose!(g, k_roped, 1, 2)   # → (n_kv, head_dim, seq)

    # Tile KV heads: reshape (n_kv, 1, head_dim, seq) then broadcast with (n_kv, gqa_ratio, head_dim, seq)
    # MPSGraph concat/tile isn't straightforward. Let's just tile via reshape + expand + reshape
    if gqa_ratio > 1
        # (n_kv, head_dim, seq) → (n_kv, 1, head_dim, seq) → broadcast to (n_kv, gqa, head_dim, seq)
        # → reshape to (n_q, head_dim, seq)
        k_exp = expanddims!(g, k_perm, 2)  # (n_kv, 1, head_dim, seq)
        # Use concat with gqa_ratio copies — this physically tiles
        k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 2)  # (n_kv, gqa, head_dim, seq)
        k_final = reshape!(g, k_tiled, (n_q_heads, head_dim, seq_len))  # (n_q, head_dim, seq)

        v_perm = graph_transpose!(g, v_3d, 1, 2)  # (n_kv, head_dim, seq)
        v_exp = expanddims!(g, v_perm, 2)
        v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 2)
        v_final = reshape!(g, v_tiled, (n_q_heads, head_dim, seq_len))
    else
        k_final = k_perm
        v_final = graph_transpose!(g, v_3d, 1, 2)  # (n_kv, head_dim, seq)
    end

    # Step 3: Batched matmul for scores
    # scores = Q @ K^T: (n_q, seq, head_dim) @ (n_q, head_dim, seq) → (n_q, seq, seq)
    # In our wrapper, matmul!(g, a, b) = a @ b in Julia convention
    # But for 3D batch matmul, the batch dim is dim 3 (last) in Julia
    # Wait — shapes are Julia convention: q_perm is (n_q, seq, head_dim) in Julia
    # MPSGraph sees it as (head_dim, seq, n_q) in row-major
    # MPSGraph batched matmul: last 2 dims are the matrix dims, earlier dims are batch
    # So (head_dim, seq, n_q) × (seq, head_dim, n_q) matmul in MPSGraph
    # = for each batch n_q: (head_dim, seq) × (seq, head_dim) = (head_dim, head_dim) — WRONG

    # I need to think about this more carefully.
    # In Julia col-major convention, matmul on 2D is: (M, K) @ (K, N) = (M, N)
    # In our wrapper, matmul!(g, a, b) does this correctly for 2D by swapping args to MPSGraph.

    # For 3D batched matmul in Julia convention: (M, K, B) @ (K, N, B) = (M, N, B)
    # which maps to MPSGraph row-major: (B, K, M) @ (B, N, K) = (B, N, M) — that IS matmul(b,a) in MPSGraph!
    # So our existing matmul! (which does MPSGraph matmul(b, a)) should work for 3D too!

    # Let's verify the shapes:
    # Q: Julia (n_q, seq, head_dim) = MPSGraph (head_dim, seq, n_q)
    # K_final: Julia (n_q, head_dim, seq) = MPSGraph (seq, head_dim, n_q)
    # scores = matmul!(g, Q, K_final): Julia (n_q, seq, head_dim) @ (n_q, head_dim, seq)
    # Hmm this doesn't match the (M,K,B) @ (K,N,B) pattern since batch B is dim 1 not dim 3.

    # I think the issue is that our Q is (n_q, seq, head_dim) in Julia,
    # but for batched matmul we want batch as the LAST dim in Julia (which becomes first in MPSGraph).

    # Let me rearrange: Q_t = (seq, head_dim, n_q) so batch=n_q is last dim
    # Hmm wait, that means the matrix dims are (seq, head_dim) and we want scores as (seq, seq).
    # So: Q_t (seq, head_dim, n_q) @ K_t (head_dim, seq, n_q) = (seq, seq, n_q) ← scores!

    # MPSGraph: (n_q, head_dim, seq) matmul(b,a) (n_q, seq, head_dim) = (n_q, seq, seq)
    # reversed to Julia: (seq, seq, n_q) ← YES!

    # So Q needs to be (seq, head_dim, n_q) and K needs to be (head_dim, seq, n_q):
    # q_roped is (head_dim, n_q, seq) → transpose(2,3) → (head_dim, seq, n_q) — that's K format
    # We need Q as (seq, head_dim, n_q) → transpose(1,2) of (head_dim, seq, n_q)

    # Let me redo this cleanly:
    # q_roped: (head_dim, n_q, seq)
    # k_roped: (head_dim, n_kv, seq)

    # For scores = Q^T K / sqrt(d):
    # Q_t: transpose to (seq, head_dim, n_q) — swap dim 1,3 then dim 1,2? No...
    # (head_dim, n_q, seq) →swap(2,3)→ (head_dim, seq, n_q) →swap(1,2)→ (seq, head_dim, n_q) ✓

    q_for_attn = graph_transpose!(g, q_roped, 2, 3)  # (head_dim, seq, n_q)
    q_for_attn = graph_transpose!(g, q_for_attn, 1, 2)  # (seq, head_dim, n_q)

    # K: (head_dim, n_kv, seq) — need (head_dim, seq, n_q) with GQA tiling
    k_for_attn = graph_transpose!(g, k_roped, 2, 3)  # (head_dim, seq, n_kv)

    # V: (head_dim, n_kv, seq) — need (seq, head_dim, n_q) for value aggregation later
    # Actually V: need (head_dim, seq, n_q) for attn_weights @ V
    v_for_attn = graph_transpose!(g, v_3d, 2, 3)  # (head_dim, seq, n_kv)

    # GQA tiling
    if gqa_ratio > 1
        # (head_dim, seq, n_kv) → (head_dim, seq, 1, n_kv) → tile → (head_dim, seq, gqa, n_kv)
        # → reshape to (head_dim, seq, n_q)
        k_exp = expanddims!(g, k_for_attn, 3)  # (head_dim, seq, 1, n_kv)
        k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 3)  # (head_dim, seq, gqa, n_kv)
        k_for_attn = reshape!(g, k_tiled, (head_dim, seq_len, n_q_heads))

        v_exp = expanddims!(g, v_for_attn, 3)
        v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 3)
        v_for_attn = reshape!(g, v_tiled, (head_dim, seq_len, n_q_heads))
    end

    # Batched scores: (seq, head_dim, n_q) @ (head_dim, seq, n_q) → (seq, seq, n_q)
    scores = matmul!(g, q_for_attn, k_for_attn)

    # Scale
    scale_t = constant_scalar!(g, Float64(scale), Float16)
    scores = mul!(g, scores, scale_t)

    # Causal mask: lower triangular
    # Create ones matrix (seq, seq) and use bandpart to make it lower triangular
    ones_mat = constant_fill!(g, 1.0, (seq_len, seq_len), Float16)
    mask = bandpart!(g, ones_mat, -1, 0)  # lower triangular (including diagonal)
    # Where mask == 0, set scores to -inf
    neg_inf = constant_scalar!(g, -1e4, Float16)  # -inf for fp16
    inv_mask = sub!(g, constant_scalar!(g, 1.0, Float16), mask)  # 1 - mask = upper triangular
    mask_penalty = mul!(g, inv_mask, neg_inf)
    scores = add!(g, scores, mask_penalty)

    # Softmax along dim 1 (the K sequence dimension — first seq in (seq, seq, n_q))
    attn_weights = softmax!(g, scores, 1)

    # Value aggregation: attn_weights @ V
    # attn_weights: (seq, seq, n_q), V: (head_dim, seq, n_q)
    # Want: (head_dim, seq, n_q) = V @ attn_weights^T per batch
    # i.e., out[d, i, h] = sum_j V[d, j, h] * attn[i, j, h]
    # = V @ attn^T in the (head_dim, seq) dims with batch n_q

    # (head_dim, seq, n_q) @ (seq, seq, n_q) — matmul dims are (head_dim, seq) @ (seq, seq) = (head_dim, seq)
    # But matmul!(a, b) = a @ b where a is (M,K,B) b is (K,N,B) → (M,N,B)
    # v_for_attn: (head_dim, seq, n_q) — M=head_dim, K=seq
    # attn_weights: (seq, seq, n_q) — K=seq, N=seq
    # So matmul!(v_for_attn, attn_weights) = (head_dim, seq, n_q) ✓
    attn_out = matmul!(g, v_for_attn, attn_weights)

    # Transpose back: (head_dim, seq, n_q) → (head_dim, n_q, seq)
    attn_out = graph_transpose!(g, attn_out, 2, 3)

    # Flatten heads: (head_dim, n_q, seq) → (q_dim, seq)
    attn_flat = reshape!(g, attn_out, (q_dim, seq_len))

    # O projection
    out = matmul!(g, w_o_ph, attn_flat)  # (hidden, seq)

    compiled = compile!(g, [out])

    GraphAttention(compiled, normed_ph, w_qkv_ph, w_o_ph, cos_ph, sin_ph,
                   out, hidden, head_dim, n_q_heads, n_kv_heads)
end

"""Apply RoPE within graph: split → rotate → concat."""
function apply_rope!(g::MetalGraphBuilder, x::GraphTensor, cos_t::GraphTensor,
                     sin_t::GraphTensor, half_hd::Int)
    x_lo = slice!(g, x, 1, 0, half_hd)
    x_hi = slice!(g, x, 1, half_hd, half_hd)
    out_lo = sub!(g, mul!(g, x_lo, cos_t), mul!(g, x_hi, sin_t))
    out_hi = add!(g, mul!(g, x_hi, cos_t), mul!(g, x_lo, sin_t))
    concat!(g, [out_lo, out_hi], 1)
end


# ── Full transformer layer graph ──

struct GraphTransformerLayer
    compiled::CompiledGraph

    # Placeholders
    x_ph::GraphTensor           # input (hidden, seq) — already normed for this layer
    residual_ph::GraphTensor    # residual stream (hidden, seq)
    w_qkv_ph::GraphTensor       # fused QKV weights
    w_o_ph::GraphTensor         # O projection weights
    cos_ph::GraphTensor         # RoPE cos
    sin_ph::GraphTensor         # RoPE sin
    post_norm_w_ph::GraphTensor # post-attention norm weights
    w_gate_up_ph::GraphTensor   # fused gate+up weights
    w_down_ph::GraphTensor      # down projection weights

    # Outputs
    out_residual::GraphTensor   # updated residual (hidden, seq)
    out_normed::GraphTensor     # normed output for next layer (hidden, seq)

    hidden::Int
    seq_len::Int
end

"""Build a full transformer layer as a single MPSGraph (prefill only, no KV cache)."""
function build_transformer_layer(hidden::Int, head_dim::Int, n_q_heads::Int, n_kv_heads::Int,
                                 intermediate::Int, seq_len::Int, eps::Float32;
                                 emit_normed::Bool=true)
    g = MetalGraphBuilder()
    q_dim = n_q_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    gqa_ratio = n_q_heads ÷ n_kv_heads
    scale = Float16(1.0 / sqrt(Float64(head_dim)))

    # ── Placeholders ──
    x_ph = placeholder!(g, (hidden, seq_len), Float16)            # normed input
    residual_ph = placeholder!(g, (hidden, seq_len), Float16)     # residual stream
    w_qkv_ph = placeholder!(g, (qkv_dim, hidden), Float16)
    w_o_ph = placeholder!(g, (hidden, q_dim), Float16)
    cos_ph = placeholder!(g, (half_hd, seq_len), Float16)
    sin_ph = placeholder!(g, (half_hd, seq_len), Float16)
    post_norm_w_ph = placeholder!(g, (hidden, 1), Float16)
    w_gu_ph = placeholder!(g, (2 * intermediate, hidden), Float16)
    w_down_ph = placeholder!(g, (hidden, intermediate), Float16)

    # ── Attention ──
    qkv = matmul!(g, w_qkv_ph, x_ph)
    q_flat = slice!(g, qkv, 1, 0, q_dim)
    k_flat = slice!(g, qkv, 1, q_dim, kv_dim)
    v_flat = slice!(g, qkv, 1, q_dim + kv_dim, kv_dim)

    q_3d = reshape!(g, q_flat, (head_dim, n_q_heads, seq_len))
    k_3d = reshape!(g, k_flat, (head_dim, n_kv_heads, seq_len))
    v_3d = reshape!(g, v_flat, (head_dim, n_kv_heads, seq_len))

    cos_3d = reshape!(g, cos_ph, (half_hd, 1, seq_len))
    sin_3d = reshape!(g, sin_ph, (half_hd, 1, seq_len))

    q_roped = apply_rope!(g, q_3d, cos_3d, sin_3d, half_hd)
    k_roped = apply_rope!(g, k_3d, cos_3d, sin_3d, half_hd)

    # Rearrange for batched matmul: batch dim is last in Julia (first in MPSGraph)
    q_t = graph_transpose!(g, q_roped, 2, 3)   # (hd, seq, n_q)
    q_t = graph_transpose!(g, q_t, 1, 2)        # (seq, hd, n_q)

    k_t = graph_transpose!(g, k_roped, 2, 3)    # (hd, seq, n_kv)
    v_t = graph_transpose!(g, v_3d, 2, 3)        # (hd, seq, n_kv)

    # GQA tiling
    if gqa_ratio > 1
        k_exp = expanddims!(g, k_t, 3)
        k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 3)
        k_t = reshape!(g, k_tiled, (head_dim, seq_len, n_q_heads))

        v_exp = expanddims!(g, v_t, 3)
        v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 3)
        v_t = reshape!(g, v_tiled, (head_dim, seq_len, n_q_heads))
    end

    # Scores: (seq, hd, n_q) @ (hd, seq, n_q) → (seq, seq, n_q)
    scores = matmul!(g, q_t, k_t)
    scores = mul!(g, scores, constant_scalar!(g, Float64(scale), Float16))

    # Causal mask
    ones_mat = constant_fill!(g, 1.0, (seq_len, seq_len), Float16)
    mask = bandpart!(g, ones_mat, -1, 0)
    inv_mask = sub!(g, constant_scalar!(g, 1.0, Float16), mask)
    scores = add!(g, scores, mul!(g, inv_mask, constant_scalar!(g, -1e4, Float16)))

    attn_w = softmax!(g, scores, 1)

    # Value aggregation: (hd, seq, n_q) @ (seq, seq, n_q) → (hd, seq, n_q)
    attn_out = matmul!(g, v_t, attn_w)

    # Back to (hd, n_q, seq) → flatten to (q_dim, seq)
    attn_out = graph_transpose!(g, attn_out, 2, 3)
    attn_flat = reshape!(g, attn_out, (q_dim, seq_len))

    # O projection + residual
    o_out = matmul!(g, w_o_ph, attn_flat)
    attn_residual = add!(g, residual_ph, o_out)

    # ── MLP ──
    normed_mlp = rmsnorm!(g, attn_residual, post_norm_w_ph, eps)
    gu = matmul!(g, w_gu_ph, normed_mlp)
    gate_out = slice!(g, gu, 1, 0, intermediate)
    up_out = slice!(g, gu, 1, intermediate, intermediate)
    swi = silu!(g, gate_out)
    mlp_hidden = mul!(g, swi, up_out)
    mlp_out = matmul!(g, w_down_ph, mlp_hidden)
    out_residual = add!(g, attn_residual, mlp_out)

    # Optionally compute normed output for next layer
    # Use a dummy norm weight placeholder if needed
    out_normed = if emit_normed
        next_norm_w_ph = placeholder!(g, (hidden, 1), Float16)
        rmsnorm!(g, out_residual, next_norm_w_ph, eps)
    else
        out_residual  # placeholder
    end

    targets = emit_normed ? [out_residual, out_normed] : [out_residual]
    compiled = compile!(g, targets)

    # Note: if emit_normed, the last placeholder is next_norm_w
    GraphTransformerLayer(compiled,
        x_ph, residual_ph, w_qkv_ph, w_o_ph, cos_ph, sin_ph,
        post_norm_w_ph, w_gu_ph, w_down_ph,
        out_residual, out_normed, hidden, seq_len)
end

# ── Multi-layer fused graph ──

struct GraphFullForward
    compiled::CompiledGraph

    # Per-layer placeholders (indexed by layer)
    input_norm_ws::Vector{GraphTensor}     # input rmsnorm weights
    w_qkvs::Vector{GraphTensor}
    w_os::Vector{GraphTensor}
    post_norm_ws::Vector{GraphTensor}
    w_gate_ups::Vector{GraphTensor}
    w_downs::Vector{GraphTensor}

    # Shared placeholders
    x_ph::GraphTensor          # initial input (hidden, seq) — embedding output
    cos_ph::GraphTensor
    sin_ph::GraphTensor
    final_norm_w_ph::GraphTensor

    # Outputs
    out::GraphTensor           # final normed output (hidden, seq)

    n_layers::Int
    hidden::Int
    seq_len::Int
end

"""Build a full N-layer transformer forward pass as a SINGLE MPSGraph.
Eliminates all per-layer Julia→ObjC dispatch overhead."""
function build_full_forward(hidden::Int, head_dim::Int, n_q_heads::Int, n_kv_heads::Int,
                            intermediate::Int, seq_len::Int, n_layers::Int, eps::Float32)
    g = MetalGraphBuilder()
    q_dim = n_q_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    gqa_ratio = n_q_heads ÷ n_kv_heads
    scale = Float16(1.0 / sqrt(Float64(head_dim)))

    # Shared placeholders
    x_ph = placeholder!(g, (hidden, seq_len), Float16)
    cos_ph = placeholder!(g, (half_hd, seq_len), Float16)
    sin_ph = placeholder!(g, (half_hd, seq_len), Float16)
    cos_3d = reshape!(g, cos_ph, (half_hd, 1, seq_len))
    sin_3d = reshape!(g, sin_ph, (half_hd, 1, seq_len))

    # Per-layer placeholders
    input_norm_ws = GraphTensor[]
    w_qkvs = GraphTensor[]
    w_os = GraphTensor[]
    post_norm_ws = GraphTensor[]
    w_gate_ups = GraphTensor[]
    w_downs = GraphTensor[]

    residual = x_ph
    for layer_idx in 1:n_layers
        # Layer norm weights
        in_nw = placeholder!(g, (hidden, 1), Float16)
        post_nw = placeholder!(g, (hidden, 1), Float16)
        push!(input_norm_ws, in_nw)
        push!(post_norm_ws, post_nw)

        # Weight placeholders
        wqkv = placeholder!(g, (qkv_dim, hidden), Float16)
        wo = placeholder!(g, (hidden, q_dim), Float16)
        wgu = placeholder!(g, (2 * intermediate, hidden), Float16)
        wd = placeholder!(g, (hidden, intermediate), Float16)
        push!(w_qkvs, wqkv); push!(w_os, wo)
        push!(w_gate_ups, wgu); push!(w_downs, wd)

        # ── Input RMSNorm ──
        normed = rmsnorm!(g, residual, in_nw, eps)

        # ── Attention ──
        qkv = matmul!(g, wqkv, normed)
        q_flat = slice!(g, qkv, 1, 0, q_dim)
        k_flat = slice!(g, qkv, 1, q_dim, kv_dim)
        v_flat = slice!(g, qkv, 1, q_dim + kv_dim, kv_dim)

        q_3d = reshape!(g, q_flat, (head_dim, n_q_heads, seq_len))
        k_3d = reshape!(g, k_flat, (head_dim, n_kv_heads, seq_len))
        v_3d = reshape!(g, v_flat, (head_dim, n_kv_heads, seq_len))

        q_roped = apply_rope!(g, q_3d, cos_3d, sin_3d, half_hd)
        k_roped = apply_rope!(g, k_3d, cos_3d, sin_3d, half_hd)

        # Rearrange for batched matmul
        q_t = graph_transpose!(g, q_roped, 2, 3)
        q_t = graph_transpose!(g, q_t, 1, 2)
        k_t = graph_transpose!(g, k_roped, 2, 3)
        v_t = graph_transpose!(g, v_3d, 2, 3)

        if gqa_ratio > 1
            k_exp = expanddims!(g, k_t, 3)
            k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 3)
            k_t = reshape!(g, k_tiled, (head_dim, seq_len, n_q_heads))
            v_exp = expanddims!(g, v_t, 3)
            v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 3)
            v_t = reshape!(g, v_tiled, (head_dim, seq_len, n_q_heads))
        end

        scores = matmul!(g, q_t, k_t)
        scores = mul!(g, scores, constant_scalar!(g, Float64(scale), Float16))

        ones_mat = constant_fill!(g, 1.0, (seq_len, seq_len), Float16)
        mask = bandpart!(g, ones_mat, -1, 0)
        inv_mask = sub!(g, constant_scalar!(g, 1.0, Float16), mask)
        scores = add!(g, scores, mul!(g, inv_mask, constant_scalar!(g, -1e4, Float16)))

        attn_w = softmax!(g, scores, 1)
        attn_out = matmul!(g, v_t, attn_w)
        attn_out = graph_transpose!(g, attn_out, 2, 3)
        attn_flat = reshape!(g, attn_out, (q_dim, seq_len))

        o_out = matmul!(g, wo, attn_flat)
        residual = add!(g, residual, o_out)

        # ── MLP ──
        normed_mlp = rmsnorm!(g, residual, post_nw, eps)
        gu = matmul!(g, wgu, normed_mlp)
        gate_out = slice!(g, gu, 1, 0, intermediate)
        up_out = slice!(g, gu, 1, intermediate, intermediate)
        mlp_out = matmul!(g, wd, mul!(g, silu!(g, gate_out), up_out))
        residual = add!(g, residual, mlp_out)
    end

    # Final norm
    final_nw = placeholder!(g, (hidden, 1), Float16)
    final_out = rmsnorm!(g, residual, final_nw, eps)

    compiled = compile!(g, [final_out])

    GraphFullForward(compiled,
        input_norm_ws, w_qkvs, w_os, post_norm_ws, w_gate_ups, w_downs,
        x_ph, cos_ph, sin_ph, final_nw, final_out,
        n_layers, hidden, seq_len)
end

# ── Decode graph with KV cache ──

struct GraphDecodeLayer
    """Single-layer decode graph: takes KV cache as input, outputs layer result + new K/V."""
    compiled::CompiledGraph

    # Placeholders
    x_ph::GraphTensor           # input (hidden, 1) — already normed
    residual_ph::GraphTensor    # residual (hidden, 1)
    w_qkv_ph::GraphTensor
    w_o_ph::GraphTensor
    cos_ph::GraphTensor         # (half_hd, 1) for current position
    sin_ph::GraphTensor
    k_cache_ph::GraphTensor     # (head_dim, n_kv, max_len)
    v_cache_ph::GraphTensor     # (head_dim, n_kv, max_len)
    attn_mask_ph::GraphTensor   # (1, max_len) — 0 for valid, -1e4 for invalid
    post_norm_w_ph::GraphTensor
    w_gate_up_ph::GraphTensor
    w_down_ph::GraphTensor

    # Outputs
    out_residual::GraphTensor   # (hidden, 1)
    new_k::GraphTensor          # (head_dim, n_kv, 1) — to append to cache
    new_v::GraphTensor          # (head_dim, n_kv, 1)

    max_len::Int
end

"""Build a decode (seq=1) transformer layer with KV cache attention.
The KV cache is passed as a placeholder of fixed max_len; a mask hides unused positions."""
function build_decode_layer(hidden::Int, head_dim::Int, n_q_heads::Int, n_kv_heads::Int,
                            intermediate::Int, max_len::Int, eps::Float32)
    g = MetalGraphBuilder()
    q_dim = n_q_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    gqa_ratio = n_q_heads ÷ n_kv_heads
    scale = Float16(1.0 / sqrt(Float64(head_dim)))

    # Placeholders
    x_ph = placeholder!(g, (hidden, 1), Float16)
    residual_ph = placeholder!(g, (hidden, 1), Float16)
    w_qkv_ph = placeholder!(g, (qkv_dim, hidden), Float16)
    w_o_ph = placeholder!(g, (hidden, q_dim), Float16)
    cos_ph = placeholder!(g, (half_hd, 1), Float16)
    sin_ph = placeholder!(g, (half_hd, 1), Float16)
    k_cache_ph = placeholder!(g, (head_dim, n_kv_heads, max_len), Float16)
    v_cache_ph = placeholder!(g, (head_dim, n_kv_heads, max_len), Float16)
    attn_mask_ph = placeholder!(g, (1, max_len), Float16)  # 0 or -1e4
    post_norm_w_ph = placeholder!(g, (hidden, 1), Float16)
    w_gu_ph = placeholder!(g, (2 * intermediate, hidden), Float16)
    w_down_ph = placeholder!(g, (hidden, intermediate), Float16)

    # ── QKV ──
    qkv = matmul!(g, w_qkv_ph, x_ph)  # (qkv_dim, 1)
    q_flat = slice!(g, qkv, 1, 0, q_dim)
    k_flat = slice!(g, qkv, 1, q_dim, kv_dim)
    v_flat = slice!(g, qkv, 1, q_dim + kv_dim, kv_dim)

    # Reshape to 3D: (head_dim, n_heads, 1)
    q_3d = reshape!(g, q_flat, (head_dim, n_q_heads, 1))
    k_3d = reshape!(g, k_flat, (head_dim, n_kv_heads, 1))
    v_3d = reshape!(g, v_flat, (head_dim, n_kv_heads, 1))

    # RoPE
    cos_3d = reshape!(g, cos_ph, (half_hd, 1, 1))
    sin_3d = reshape!(g, sin_ph, (half_hd, 1, 1))
    q_roped = apply_rope!(g, q_3d, cos_3d, sin_3d, half_hd)
    k_roped = apply_rope!(g, k_3d, cos_3d, sin_3d, half_hd)

    # New K/V to output (for cache update)
    new_k = k_roped  # (head_dim, n_kv, 1)
    new_v = v_3d     # (head_dim, n_kv, 1) — V doesn't get RoPE

    # ── Attention with KV cache ──
    # K cache: (head_dim, n_kv, max_len) — already includes all past K
    # For the current step, the caller should have already written new_k into the cache
    # at the right position. So k_cache_ph is the FULL cache including this step.
    # (We output new_k/new_v for the caller to update cache BEFORE the next call.)

    # Q: (head_dim, n_q, 1) → need (1, head_dim, n_q) for batched matmul
    q_t = graph_transpose!(g, q_roped, 2, 3)    # (head_dim, 1, n_q)
    q_t = graph_transpose!(g, q_t, 1, 2)         # (1, head_dim, n_q)

    # K cache: (head_dim, n_kv, max_len) → (head_dim, max_len, n_kv) for matmul
    k_t = graph_transpose!(g, k_cache_ph, 2, 3)  # (head_dim, max_len, n_kv)

    # V cache: same rearrangement
    v_t = graph_transpose!(g, v_cache_ph, 2, 3)  # (head_dim, max_len, n_kv)

    # GQA tiling for K and V
    if gqa_ratio > 1
        k_exp = expanddims!(g, k_t, 3)
        k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 3)
        k_t = reshape!(g, k_tiled, (head_dim, max_len, n_q_heads))

        v_exp = expanddims!(g, v_t, 3)
        v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 3)
        v_t = reshape!(g, v_tiled, (head_dim, max_len, n_q_heads))
    end

    # Scores: (1, head_dim, n_q) @ (head_dim, max_len, n_q) → (1, max_len, n_q)
    scores = matmul!(g, q_t, k_t)
    scores = mul!(g, scores, constant_scalar!(g, Float64(scale), Float16))

    # Apply mask: attn_mask is (1, max_len), broadcast over n_q
    scores = add!(g, scores, attn_mask_ph)

    # Softmax along dim 2 (the max_len dimension: (1, max_len, n_q) → softmax over max_len)
    attn_w = softmax!(g, scores, 2)

    # Value aggregation: (head_dim, max_len, n_q) @ (max_len, 1, n_q)
    # We need attn_w transposed: (1, max_len, n_q) → (max_len, 1, n_q)
    attn_w_t = graph_transpose!(g, attn_w, 1, 2)  # (max_len, 1, n_q)
    # Actually: matmul!(v_t, attn_w_t) = (head_dim, max_len, n_q) @ (max_len, 1, n_q) → (head_dim, 1, n_q)
    attn_out = matmul!(g, v_t, attn_w_t)

    # Reshape back: (head_dim, 1, n_q) → (head_dim, n_q, 1) → (q_dim, 1)
    attn_out = graph_transpose!(g, attn_out, 2, 3)  # (head_dim, n_q, 1)
    attn_flat = reshape!(g, attn_out, (q_dim, 1))

    # O projection + residual
    o_out = matmul!(g, w_o_ph, attn_flat)
    attn_res = add!(g, residual_ph, o_out)

    # ── MLP ──
    normed_mlp = rmsnorm!(g, attn_res, post_norm_w_ph, eps)
    gu = matmul!(g, w_gu_ph, normed_mlp)
    gate_out = slice!(g, gu, 1, 0, intermediate)
    up_out = slice!(g, gu, 1, intermediate, intermediate)
    mlp_out = matmul!(g, w_down_ph, mul!(g, silu!(g, gate_out), up_out))
    out_residual = add!(g, attn_res, mlp_out)

    compiled = compile!(g, [out_residual, new_k, new_v])

    GraphDecodeLayer(compiled,
        x_ph, residual_ph, w_qkv_ph, w_o_ph, cos_ph, sin_ph,
        k_cache_ph, v_cache_ph, attn_mask_ph,
        post_norm_w_ph, w_gu_ph, w_down_ph,
        out_residual, new_k, new_v, max_len)
end


# ── Fused multi-layer decode ──

struct GraphDecodeFull
    """Fused N-layer decode graph with KV cache."""
    compiled::CompiledGraph

    # Shared
    x_ph::GraphTensor
    cos_ph::GraphTensor
    sin_ph::GraphTensor
    attn_mask_ph::GraphTensor

    # Per-layer
    input_norm_ws::Vector{GraphTensor}
    w_qkvs::Vector{GraphTensor}
    w_os::Vector{GraphTensor}
    k_caches::Vector{GraphTensor}
    v_caches::Vector{GraphTensor}
    post_norm_ws::Vector{GraphTensor}
    w_gate_ups::Vector{GraphTensor}
    w_downs::Vector{GraphTensor}

    # Final norm
    final_norm_w_ph::GraphTensor

    # Outputs
    out::GraphTensor
    new_ks::Vector{GraphTensor}
    new_vs::Vector{GraphTensor}

    n_layers::Int
    max_len::Int
end

"""Build fused N-layer decode graph with KV cache attention."""
function build_decode_full(hidden::Int, head_dim::Int, n_q_heads::Int, n_kv_heads::Int,
                           intermediate::Int, max_len::Int, n_layers::Int, eps::Float32)
    g = MetalGraphBuilder()
    q_dim = n_q_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = head_dim ÷ 2
    gqa_ratio = n_q_heads ÷ n_kv_heads
    scale = Float16(1.0 / sqrt(Float64(head_dim)))

    # Shared placeholders
    x_ph = placeholder!(g, (hidden, 1), Float16)
    cos_ph = placeholder!(g, (half_hd, 1), Float16)
    sin_ph = placeholder!(g, (half_hd, 1), Float16)
    attn_mask_ph = placeholder!(g, (1, max_len), Float16)

    cos_3d = reshape!(g, cos_ph, (half_hd, 1, 1))
    sin_3d = reshape!(g, sin_ph, (half_hd, 1, 1))

    input_norm_ws = GraphTensor[]
    w_qkvs = GraphTensor[]
    w_os = GraphTensor[]
    k_caches = GraphTensor[]
    v_caches = GraphTensor[]
    post_norm_ws = GraphTensor[]
    w_gate_ups = GraphTensor[]
    w_downs = GraphTensor[]
    new_ks = GraphTensor[]
    new_vs = GraphTensor[]

    residual = x_ph

    for _ in 1:n_layers
        in_nw = placeholder!(g, (hidden, 1), Float16)
        wqkv = placeholder!(g, (qkv_dim, hidden), Float16)
        wo = placeholder!(g, (hidden, q_dim), Float16)
        kc = placeholder!(g, (head_dim, n_kv_heads, max_len), Float16)
        vc = placeholder!(g, (head_dim, n_kv_heads, max_len), Float16)
        post_nw = placeholder!(g, (hidden, 1), Float16)
        wgu = placeholder!(g, (2 * intermediate, hidden), Float16)
        wd = placeholder!(g, (hidden, intermediate), Float16)

        push!(input_norm_ws, in_nw); push!(w_qkvs, wqkv); push!(w_os, wo)
        push!(k_caches, kc); push!(v_caches, vc)
        push!(post_norm_ws, post_nw); push!(w_gate_ups, wgu); push!(w_downs, wd)

        # RMSNorm + QKV
        normed = rmsnorm!(g, residual, in_nw, eps)
        qkv = matmul!(g, wqkv, normed)
        q_flat = slice!(g, qkv, 1, 0, q_dim)
        k_flat = slice!(g, qkv, 1, q_dim, kv_dim)
        v_flat = slice!(g, qkv, 1, q_dim + kv_dim, kv_dim)

        q_3d = reshape!(g, q_flat, (head_dim, n_q_heads, 1))
        k_3d = reshape!(g, k_flat, (head_dim, n_kv_heads, 1))
        v_3d = reshape!(g, v_flat, (head_dim, n_kv_heads, 1))

        q_roped = apply_rope!(g, q_3d, cos_3d, sin_3d, half_hd)
        k_roped = apply_rope!(g, k_3d, cos_3d, sin_3d, half_hd)
        push!(new_ks, k_roped)
        push!(new_vs, v_3d)

        # Attention with KV cache
        q_t = graph_transpose!(g, q_roped, 2, 3)
        q_t = graph_transpose!(g, q_t, 1, 2)     # (1, head_dim, n_q)
        k_t = graph_transpose!(g, kc, 2, 3)       # (head_dim, max_len, n_kv)
        v_t = graph_transpose!(g, vc, 2, 3)       # (head_dim, max_len, n_kv)

        if gqa_ratio > 1
            k_exp = expanddims!(g, k_t, 3)
            k_tiled = concat!(g, [k_exp for _ in 1:gqa_ratio], 3)
            k_t = reshape!(g, k_tiled, (head_dim, max_len, n_q_heads))
            v_exp = expanddims!(g, v_t, 3)
            v_tiled = concat!(g, [v_exp for _ in 1:gqa_ratio], 3)
            v_t = reshape!(g, v_tiled, (head_dim, max_len, n_q_heads))
        end

        scores = matmul!(g, q_t, k_t)  # (1, max_len, n_q)
        scores = mul!(g, scores, constant_scalar!(g, Float64(scale), Float16))
        scores = add!(g, scores, attn_mask_ph)
        attn_w = softmax!(g, scores, 2)

        attn_w_t = graph_transpose!(g, attn_w, 1, 2)  # (max_len, 1, n_q)
        attn_out = matmul!(g, v_t, attn_w_t)           # (head_dim, 1, n_q)
        attn_out = graph_transpose!(g, attn_out, 2, 3)  # (head_dim, n_q, 1)
        attn_flat = reshape!(g, attn_out, (q_dim, 1))

        o_out = matmul!(g, wo, attn_flat)
        residual = add!(g, residual, o_out)

        # MLP
        normed_mlp = rmsnorm!(g, residual, post_nw, eps)
        gu = matmul!(g, wgu, normed_mlp)
        gate_out = slice!(g, gu, 1, 0, intermediate)
        up_out = slice!(g, gu, 1, intermediate, intermediate)
        mlp_out = matmul!(g, wd, mul!(g, silu!(g, gate_out), up_out))
        residual = add!(g, residual, mlp_out)
    end

    final_nw = placeholder!(g, (hidden, 1), Float16)
    final_out = rmsnorm!(g, residual, final_nw, eps)

    all_outputs = GraphTensor[final_out]
    append!(all_outputs, new_ks)
    append!(all_outputs, new_vs)

    compiled = compile!(g, all_outputs)

    GraphDecodeFull(compiled,
        x_ph, cos_ph, sin_ph, attn_mask_ph,
        input_norm_ws, w_qkvs, w_os, k_caches, v_caches,
        post_norm_ws, w_gate_ups, w_downs,
        final_nw, final_out, new_ks, new_vs,
        n_layers, max_len)
end

export GraphMLP, build_mlp_graph, execute_mlp!
export GraphAttention, build_attention_graph, apply_rope!
export GraphTransformerLayer, build_transformer_layer
export GraphFullForward, build_full_forward
export GraphDecodeLayer, build_decode_layer
export GraphDecodeFull, build_decode_full

end # module
