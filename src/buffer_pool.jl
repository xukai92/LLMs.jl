"""
Pre-allocated buffer pool for inference.

Eliminates per-forward-pass MtlArray allocations by pre-allocating all
intermediate tensors at model load time and reusing them.
"""

struct BufferPool
    # Per-layer reusable buffers (all Float16)
    # Attention projections
    q::MtlMatrix{Float16}         # (n_heads * head_dim, max_batch)
    k::MtlMatrix{Float16}         # (n_kv_heads * head_dim, max_batch)
    v::MtlMatrix{Float16}         # (n_kv_heads * head_dim, max_batch)
    attn_out_flat::MtlMatrix{Float16}  # (hidden, max_batch)

    # MLP buffers
    gate::MtlMatrix{Float16}      # (intermediate, max_batch)
    up::MtlMatrix{Float16}        # (intermediate, max_batch)
    swiglu_out::MtlMatrix{Float16} # (intermediate, max_batch)
    mlp_out::MtlMatrix{Float16}   # (hidden, max_batch)

    # Norm output (shared across pre-attn and post-attn norm)
    normed::MtlMatrix{Float16}    # (hidden, max_batch)

    # Attention internals
    scores::MtlArray{Float32, 3}  # (n_heads, max_batch, max_seq)

    # Output projection (o_proj)
    o_out::MtlMatrix{Float16}     # (hidden, max_batch)

    # K/V contiguous copies for attention
    k_contig::MtlArray{Float16, 3}  # (head_dim, n_kv_heads, max_seq)
    v_contig::MtlArray{Float16, 3}  # (head_dim, n_kv_heads, max_seq)

    # Attention output 3D
    attn_out_3d::MtlArray{Float16, 3}  # (head_dim, n_heads, max_batch)

    # Logits buffer for lm_head
    logits::MtlMatrix{Float16}    # (vocab, max_batch)

    max_batch::Int
    max_seq::Int
end

function BufferPool(config::LlamaConfig; max_batch::Int=512, max_seq::Int=4096)
    h = config.hidden_size
    inter = config.intermediate_size
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim

    BufferPool(
        MtlArray(zeros(Float16, n_q * hd, max_batch)),          # q
        MtlArray(zeros(Float16, n_kv * hd, max_batch)),         # k
        MtlArray(zeros(Float16, n_kv * hd, max_batch)),         # v
        MtlArray(zeros(Float16, h, max_batch)),                  # attn_out_flat
        MtlArray(zeros(Float16, inter, max_batch)),              # gate
        MtlArray(zeros(Float16, inter, max_batch)),              # up
        MtlArray(zeros(Float16, inter, max_batch)),              # swiglu_out
        MtlArray(zeros(Float16, h, max_batch)),                  # mlp_out
        MtlArray(zeros(Float16, h, max_batch)),                  # normed
        MtlArray(zeros(Float32, n_q, max_batch, max_seq)),       # scores
        MtlArray(zeros(Float16, h, max_batch)),                  # o_out
        MtlArray(zeros(Float16, hd, n_kv, max_seq)),             # k_contig
        MtlArray(zeros(Float16, hd, n_kv, max_seq)),             # v_contig
        MtlArray(zeros(Float16, hd, n_q, max_batch)),            # attn_out_3d
        MtlArray(zeros(Float16, config.vocab_size, max_batch)),  # logits
        max_batch,
        max_seq,
    )
end

"""Get a view of the buffer sized to the current batch/seq dimensions."""
function sized(buf::MtlMatrix{T}, rows::Int, cols::Int) where T
    # Use : for dimensions that match the buffer to avoid SubArray
    # (view(:, 1:N) returns MtlMatrix, view(1:M, 1:N) returns SubArray)
    r = rows == size(buf, 1) ? Colon() : (1:rows)
    c = cols == size(buf, 2) ? Colon() : (1:cols)
    return view(buf, r, c)
end

function sized(buf::MtlArray{T, 3}, d1::Int, d2::Int, d3::Int) where T
    r1 = d1 == size(buf, 1) ? Colon() : (1:d1)
    r2 = d2 == size(buf, 2) ? Colon() : (1:d2)
    r3 = d3 == size(buf, 3) ? Colon() : (1:d3)
    return view(buf, r1, r2, r3)
end
