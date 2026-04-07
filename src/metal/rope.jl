"""
Rotary Position Embedding (RoPE).

Applies rotation to Q and K vectors using precomputed cos/sin tables.

For each pair of elements (x[2i-1], x[2i]) at position pos:
    x_rot[2i-1] = x[2i-1] * cos[i, pos] - x[2i] * sin[i, pos]
    x_rot[2i]   = x[2i-1] * sin[i, pos] + x[2i] * cos[i, pos]

Llama-3.2: head_dim=128, rope_theta=500000.0, with frequency scaling.
"""

# ── Frequency table precomputation (CPU) ──

"""
    compute_rope_freqs(head_dim, max_seq_len; theta, scaling_config)

Precompute cos/sin tables for RoPE. Returns (cos_table, sin_table) each of
shape (head_dim÷2, max_seq_len) in Float32.

For Llama-3.2, `scaling_config` applies the llama3-style frequency scaling:
  - Frequencies below low_freq_factor * (old_context/factor) are unchanged
  - Frequencies above high_freq_factor * (old_context/factor) are divided by factor
  - In between: linear interpolation
"""
function compute_rope_freqs(head_dim::Int, max_seq_len::Int;
                            theta::Float64=500000.0,
                            scaling_config::Union{Nothing, Dict}=nothing)
    half = head_dim ÷ 2
    # Base frequencies: theta^(-2i/d) for i in 0:half-1
    freqs = Float64[theta^(-2.0 * i / head_dim) for i in 0:half-1]

    # Apply Llama3-style frequency scaling if configured
    if scaling_config !== nothing
        factor = get(scaling_config, "factor", 1.0)
        low_freq_factor = get(scaling_config, "low_freq_factor", 1.0)
        high_freq_factor = get(scaling_config, "high_freq_factor", 1.0)
        old_context = get(scaling_config, "original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context / low_freq_factor
        high_freq_wavelen = old_context / high_freq_factor

        for i in 1:half
            wavelen = 2π / freqs[i]
            if wavelen > low_freq_wavelen
                # Scale down (low frequency — extend context)
                freqs[i] /= factor
            elseif wavelen > high_freq_wavelen
                # Smooth interpolation
                smooth = (old_context / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor)
                freqs[i] = (1 - smooth) * freqs[i] / factor + smooth * freqs[i]
            end
            # else: high frequency — keep unchanged
        end
    end

    # Build position-indexed tables: cos(pos * freq), sin(pos * freq)
    cos_table = zeros(Float32, half, max_seq_len)
    sin_table = zeros(Float32, half, max_seq_len)
    for pos in 1:max_seq_len
        for i in 1:half
            angle = Float64(pos - 1) * freqs[i]  # 0-indexed position
            cos_table[i, pos] = Float32(cos(angle))
            sin_table[i, pos] = Float32(sin(angle))
        end
    end

    return cos_table, sin_table
end

# ── CPU reference ──

"""
Apply RoPE in-place. x has shape (head_dim, n_heads, seq_len).
cos_table, sin_table have shape (head_dim÷2, max_seq_len).
`start_pos` is the starting position (1-indexed) for the sequence.
"""
function rope_cpu!(x::AbstractArray{T, 3}, cos_table::AbstractMatrix{Float32},
                   sin_table::AbstractMatrix{Float32}, start_pos::Int) where T
    head_dim, n_heads, seq_len = size(x)
    half = head_dim ÷ 2

    for s in 1:seq_len
        pos = start_pos + s - 1
        for h in 1:n_heads
            @inbounds for i in 1:half
                x1 = Float32(x[2i-1, h, s])
                x2 = Float32(x[2i, h, s])
                c = cos_table[i, pos]
                sn = sin_table[i, pos]
                x[2i-1, h, s] = T(x1 * c - x2 * sn)
                x[2i, h, s]   = T(x1 * sn + x2 * c)
            end
        end
    end
    return x
end

# ── Metal kernel ──

# Each thread handles one (pair_idx, head, seq_pos) triple.
# Grid: (half_dim, n_heads, seq_len)

function rope_kernel!(x, cos_table, sin_table, start_pos::Int32, half_dim::Int32)
    pos_in_grid = thread_position_in_grid()
    i = pos_in_grid.x     # pair index (1..half_dim)
    h = pos_in_grid.y     # head index
    s = pos_in_grid.z     # sequence position in this chunk

    if i > half_dim
        return nothing
    end

    pos = start_pos + Int32(s) - Int32(1)  # actual position for RoPE table

    @inbounds begin
        x1 = Float32(x[2*i-1, h, s])
        x2 = Float32(x[2*i, h, s])
        c  = cos_table[i, pos]
        sn = sin_table[i, pos]
        x[2*i-1, h, s] = typeof(x[1,1,1])(x1 * c - x2 * sn)
        x[2*i, h, s]   = typeof(x[1,1,1])(x1 * sn + x2 * c)
    end

    return nothing
end

function metal_rope!(x, cos_table, sin_table, start_pos::Int)
    head_dim, n_heads, seq_len = size(x)
    half = head_dim ÷ 2

    # 3D grid: each thread handles one pair for one head at one position
    threads_per_group = (min(half, 64), 1, 1)
    groups = (cld(half, threads_per_group[1]), n_heads, seq_len)

    @metal threads=threads_per_group groups=groups rope_kernel!(
        x, cos_table, sin_table, Int32(start_pos), Int32(half))
    return x
end
