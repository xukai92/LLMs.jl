"""
KV Cache for autoregressive decoding.

Pre-allocates contiguous buffers per layer for K and V tensors.
Appends new K/V on each forward step.
"""

mutable struct KVCache
    # K and V buffers per layer: (head_dim, n_kv_heads, max_seq_len)
    k_cache::Vector{MtlArray{Float16, 3}}
    v_cache::Vector{MtlArray{Float16, 3}}
    # Current sequence length (how many positions have been filled)
    seq_len::Int
    max_seq_len::Int
end

function KVCache(config::LlamaConfig; max_seq_len::Int=4096)
    k_cache = MtlArray{Float16, 3}[]
    v_cache = MtlArray{Float16, 3}[]

    for _ in 1:config.num_hidden_layers
        push!(k_cache, MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, max_seq_len)))
        push!(v_cache, MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, max_seq_len)))
    end

    return KVCache(k_cache, v_cache, 0, max_seq_len)
end

"""
Metal kernel to copy new K/V into the cache at the right position.
"""
function kv_append_kernel!(cache, new_kv, start_pos::Int32, head_dim::Int32, n_heads::Int32)
    tid = thread_position_in_grid_1d()
    new_seq = Int32(size(new_kv, 3))
    total = head_dim * n_heads * new_seq

    if tid <= total
        # Decompose linear index into (d, h, s)
        idx = tid - Int32(1)
        d = (idx % head_dim) + Int32(1)
        idx = idx ÷ head_dim
        h = (idx % n_heads) + Int32(1)
        s = (idx ÷ n_heads) + Int32(1)

        cache_pos = start_pos + s
        @inbounds cache[d, h, cache_pos] = new_kv[d, h, s]
    end
    return nothing
end

"""
    append_kv!(cache, layer_idx, new_k, new_v)

Append new K and V tensors to the cache for the given layer.
new_k, new_v: (head_dim, n_kv_heads, seq_len_new)
"""
function append_kv!(cache::KVCache, layer_idx::Int, new_k, new_v)
    head_dim, n_heads, new_seq = size(new_k)
    total = head_dim * n_heads * new_seq
    tg = 256

    start_pos = Int32(cache.seq_len)

    @metal threads=tg groups=cld(total, tg) kv_append_kernel!(
        cache.k_cache[layer_idx], new_k, start_pos, Int32(head_dim), Int32(n_heads))
    @metal threads=tg groups=cld(total, tg) kv_append_kernel!(
        cache.v_cache[layer_idx], new_v, start_pos, Int32(head_dim), Int32(n_heads))

    return nothing
end

"""
    get_kv(cache, layer_idx) -> (k, v)

Get the filled portion of K and V for attention computation.
Returns views into the cache covering [1:seq_len].
"""
function get_kv(cache::KVCache, layer_idx::Int)
    k = @view cache.k_cache[layer_idx][:, :, 1:cache.seq_len]
    v = @view cache.v_cache[layer_idx][:, :, 1:cache.seq_len]
    return k, v
end

function reset!(cache::KVCache)
    cache.seq_len = 0
end
