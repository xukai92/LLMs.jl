"""
Radix tree prefix cache for KV reuse (SGLang-style RadixAttention).

The radix tree maps token sequences to KV cache states. When a new request
shares a prefix with a cached sequence, we reuse the cached KV and only
compute the new tokens.

Design:
- CPU-side radix tree manages prefix matching and eviction
- KV data lives in a shared GPU pool (one per layer)
- Each tree node stores an offset+length into the pool
- LRU eviction on leaf nodes when pool exceeds token budget
"""

# ── Radix Tree Node ──

mutable struct RadixNode
    # Token sequence stored at this edge (from parent to this node)
    tokens::Vector{Int}
    # Children keyed by first token of their edge
    children::Dict{Int, RadixNode}
    # KV pool reference: this node's KV data covers pool positions
    # [pool_offset+1 : pool_offset+pool_length] (1-indexed)
    pool_offset::Int
    pool_length::Int  # number of tokens this node represents in the pool
    # LRU tracking
    last_access::Float64
    # Reference count (number of active sequences using this prefix)
    ref_count::Int
    # Parent pointer for eviction traversal
    parent::Union{Nothing, RadixNode}
end

function RadixNode(; tokens=Int[], pool_offset=0, pool_length=0, parent=nothing)
    RadixNode(tokens, Dict{Int, RadixNode}(), pool_offset, pool_length,
              time(), 0, parent)
end

# ── KV Pool ──

"""
Shared KV pool: pre-allocated GPU buffers that the radix tree indexes into.
"""
mutable struct KVPool
    # Per-layer K and V buffers: (head_dim, n_kv_heads, max_tokens)
    k_pool::Vector{MtlArray{Float16, 3}}
    v_pool::Vector{MtlArray{Float16, 3}}
    max_tokens::Int
    # Next free position in the pool (0-indexed for consistency with offsets)
    next_free::Int
    # Free list: ranges of freed slots for reuse
    free_ranges::Vector{UnitRange{Int}}
end

function KVPool(config::LlamaConfig; max_tokens::Int=8192)
    k_pool = [MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, max_tokens))
              for _ in 1:config.num_hidden_layers]
    v_pool = [MtlArray(zeros(Float16, config.head_dim, config.num_key_value_heads, max_tokens))
              for _ in 1:config.num_hidden_layers]
    KVPool(k_pool, v_pool, max_tokens, 0, UnitRange{Int}[])
end

"""Allocate a contiguous range of `n` positions from the pool."""
function pool_alloc!(pool::KVPool, n::Int)::Int
    # Try free list first
    for (i, r) in enumerate(pool.free_ranges)
        if length(r) >= n
            offset = first(r)
            if length(r) == n
                deleteat!(pool.free_ranges, i)
            else
                pool.free_ranges[i] = (first(r) + n):last(r)
            end
            return offset
        end
    end

    # Allocate from the end
    if pool.next_free + n > pool.max_tokens
        return -1  # pool full, need eviction
    end
    offset = pool.next_free
    pool.next_free += n
    return offset
end

"""Free a range of positions back to the pool."""
function pool_free!(pool::KVPool, offset::Int, length::Int)
    if length > 0
        push!(pool.free_ranges, offset:(offset + length - 1))
        # Merge adjacent ranges (simple approach)
        sort!(pool.free_ranges; by=first)
        merged = UnitRange{Int}[]
        for r in pool.free_ranges
            if !isempty(merged) && first(r) <= last(merged[end]) + 1
                merged[end] = first(merged[end]):max(last(merged[end]), last(r))
            else
                push!(merged, r)
            end
        end
        pool.free_ranges = merged
    end
end

"""Used tokens in the pool."""
function pool_used(pool::KVPool)
    free_count = sum(length, pool.free_ranges; init=0)
    return pool.next_free - free_count
end

# ── Prefix Cache ──

mutable struct PrefixCache
    root::RadixNode
    pool::KVPool
    config::LlamaConfig
    token_budget::Int  # max tokens to cache before eviction
end

function PrefixCache(config::LlamaConfig; max_tokens::Int=8192)
    pool = KVPool(config; max_tokens=max_tokens)
    root = RadixNode()
    PrefixCache(root, pool, config, max_tokens)
end

"""
    prefix_match(cache, token_ids) -> (matched_length, node, kv_positions)

Walk the radix tree to find the longest matching prefix.
Returns:
- matched_length: number of tokens matched
- node: the deepest matching node
- kv_positions: pool offsets for the matched KV data (list of (offset, length) per node)
"""
function prefix_match(cache::PrefixCache, token_ids::Vector{Int})
    node = cache.root
    pos = 1
    kv_segments = Tuple{Int, Int}[]  # (pool_offset, pool_length) for each matched node

    while pos <= length(token_ids)
        tok = token_ids[pos]
        if !haskey(node.children, tok)
            break
        end
        child = node.children[tok]

        # Check if the child's edge tokens match
        edge_len = length(child.tokens)
        remaining = length(token_ids) - pos + 1

        if remaining < edge_len
            # Partial match of edge — can't use this node
            break
        end

        # Check all tokens on the edge match
        matched = true
        for j in 1:edge_len
            if token_ids[pos + j - 1] != child.tokens[j]
                matched = false
                break
            end
        end

        if !matched
            break
        end

        # Full edge match
        pos += edge_len
        node = child
        node.last_access = time()
        if node.pool_length > 0
            push!(kv_segments, (node.pool_offset, node.pool_length))
        end
    end

    matched_length = pos - 1
    return matched_length, node, kv_segments
end

"""
    insert_prefix!(cache, token_ids, kv_cache, start_from)

Insert a token sequence into the radix tree, copying KV data from the
given KV cache into the shared pool.

`start_from`: number of tokens already in the tree (only copy new KV data).
`kv_cache`: the KVCache that has been filled during inference.
"""
function insert_prefix!(cache::PrefixCache, token_ids::Vector{Int},
                        kv_cache::KVCache, start_from::Int)
    if start_from >= length(token_ids)
        return  # nothing new to insert
    end

    new_tokens = token_ids[start_from+1:end]
    n_new = length(new_tokens)

    # Allocate pool space for the new tokens
    offset = pool_alloc!(cache.pool, n_new)
    if offset == -1
        # Pool full — evict LRU leaves
        evict_lru!(cache, n_new)
        offset = pool_alloc!(cache.pool, n_new)
        if offset == -1
            @warn "Prefix cache pool full even after eviction, skipping insert"
            return
        end
    end

    # Copy KV data from the inference cache to the pool
    # The new tokens' KV data is at positions [start_from+1 : start_from+n_new] in kv_cache
    for layer in 1:cache.config.num_hidden_layers
        src_k = kv_cache.k_cache[layer][:, :, start_from+1:start_from+n_new]
        src_v = kv_cache.v_cache[layer][:, :, start_from+1:start_from+n_new]
        # Copy to pool at [offset+1 : offset+n_new]
        copyto!(view(cache.pool.k_pool[layer], :, :, offset+1:offset+n_new), src_k)
        copyto!(view(cache.pool.v_pool[layer], :, :, offset+1:offset+n_new), src_v)
    end

    # Navigate to the insertion point in the tree, splitting edges as needed
    node = cache.root
    pos = 1
    while pos <= start_from
        tok = token_ids[pos]
        if !haskey(node.children, tok)
            break
        end
        child = node.children[tok]
        edge_len = length(child.tokens)

        if pos + edge_len - 1 <= start_from
            # This edge fits entirely within the already-cached prefix
            pos += edge_len
            node = child
        else
            # start_from falls within this edge — need to split
            split_at = start_from - pos + 1  # how many tokens of the edge are cached
            # Create intermediate node for the first split_at tokens
            split = RadixNode(;
                tokens=child.tokens[1:split_at],
                pool_offset=child.pool_offset,
                pool_length=split_at,
                parent=node
            )
            # Modify old child to represent the suffix
            child.tokens = child.tokens[split_at+1:end]
            child.pool_offset += split_at
            child.pool_length -= split_at
            child.parent = split
            split.children[child.tokens[1]] = child
            # Replace in parent
            node.children[tok] = split
            node = split
            pos = start_from + 1
            break
        end
    end

    # Insert new_tokens as a new child (or extend existing path)
    _insert_edge!(node, new_tokens, offset, n_new)
end

function _insert_edge!(parent::RadixNode, tokens::Vector{Int}, pool_offset::Int, pool_length::Int)
    if isempty(tokens)
        return
    end

    first_tok = tokens[1]

    if haskey(parent.children, first_tok)
        child = parent.children[first_tok]
        # Find common prefix between tokens and child.tokens
        common_len = 0
        for i in 1:min(length(tokens), length(child.tokens))
            if tokens[i] == child.tokens[i]
                common_len += 1
            else
                break
            end
        end

        if common_len == length(child.tokens)
            # Child edge fully matched, continue deeper
            remaining = tokens[common_len+1:end]
            if !isempty(remaining)
                # The pool_offset for the remaining part
                new_offset = pool_offset + common_len
                new_length = pool_length - common_len
                _insert_edge!(child, remaining, new_offset, new_length)
            end
        else
            # Need to split the child edge
            # Create intermediate node for the common prefix
            split = RadixNode(;
                tokens=child.tokens[1:common_len],
                pool_offset=child.pool_offset,
                pool_length=common_len,
                parent=parent
            )

            # Modify old child to represent the suffix
            child.tokens = child.tokens[common_len+1:end]
            child.pool_offset += common_len
            child.pool_length -= common_len
            child.parent = split
            split.children[child.tokens[1]] = child

            # Insert the new branch
            remaining = tokens[common_len+1:end]
            if !isempty(remaining)
                new_node = RadixNode(;
                    tokens=remaining,
                    pool_offset=pool_offset + common_len,
                    pool_length=pool_length - common_len,
                    parent=split
                )
                split.children[remaining[1]] = new_node
            end

            # Replace child with split in parent
            parent.children[first_tok] = split
        end
    else
        # No existing child — create new leaf
        new_node = RadixNode(;
            tokens=tokens,
            pool_offset=pool_offset,
            pool_length=pool_length,
            parent=parent
        )
        parent.children[first_tok] = new_node
    end
end

"""
    restore_kv!(kv_cache, cache, kv_segments)

Copy cached KV data from the pool back into the working KV cache.
"""
function restore_kv!(kv_cache::KVCache, pcache::PrefixCache,
                     kv_segments::Vector{Tuple{Int, Int}})
    dest_pos = 0
    for (pool_offset, pool_length) in kv_segments
        for layer in 1:pcache.config.num_hidden_layers
            src_k = pcache.pool.k_pool[layer][:, :, pool_offset+1:pool_offset+pool_length]
            src_v = pcache.pool.v_pool[layer][:, :, pool_offset+1:pool_offset+pool_length]
            copyto!(view(kv_cache.k_cache[layer], :, :, dest_pos+1:dest_pos+pool_length), src_k)
            copyto!(view(kv_cache.v_cache[layer], :, :, dest_pos+1:dest_pos+pool_length), src_v)
        end
        dest_pos += pool_length
    end
    kv_cache.seq_len = dest_pos
end

# ── LRU Eviction ──

"""Collect all leaf nodes in the tree."""
function collect_leaves(node::RadixNode, leaves::Vector{RadixNode}=RadixNode[])
    if isempty(node.children)
        push!(leaves, node)
    else
        for child in values(node.children)
            collect_leaves(child, leaves)
        end
    end
    return leaves
end

"""Evict LRU leaf nodes until `needed` tokens are freed."""
function evict_lru!(cache::PrefixCache, needed::Int)
    freed = 0
    while freed < needed
        leaves = collect_leaves(cache.root)
        # Filter to evictable leaves (ref_count == 0)
        evictable = filter(l -> l.ref_count == 0, leaves)
        if isempty(evictable)
            @warn "No evictable leaves in prefix cache"
            return
        end

        # Sort by last_access (oldest first)
        sort!(evictable; by=l -> l.last_access)

        # Evict the oldest leaf
        leaf = evictable[1]
        pool_free!(cache.pool, leaf.pool_offset, leaf.pool_length)
        freed += leaf.pool_length

        # Remove from parent
        if leaf.parent !== nothing
            parent = leaf.parent
            for (k, v) in parent.children
                if v === leaf
                    delete!(parent.children, k)
                    break
                end
            end

            # If parent now has exactly one child, merge them
            if length(parent.children) == 1 && parent.parent !== nothing
                only_child = first(values(parent.children))
                parent.tokens = vcat(parent.tokens, only_child.tokens)
                parent.pool_length += only_child.pool_length
                parent.children = only_child.children
                # Update children's parent pointers
                for child in values(parent.children)
                    child.parent = parent
                end
                parent.last_access = only_child.last_access
            end
        end
    end
end

# ── High-level API ──

"""
    generate_with_cache(model, pcache, token_ids; max_tokens=50)

Generate tokens using the prefix cache for KV reuse.
"""
function generate_with_cache(model::LlamaModel, pcache::PrefixCache,
                             prompt_ids::Vector{Int};
                             max_tokens::Int=50)
    config = model.config
    total_len = length(prompt_ids) + max_tokens + 16
    cache = KVCache(config; max_seq_len=total_len)

    # Check prefix cache for a match
    matched_len, matched_node, kv_segments = prefix_match(pcache, prompt_ids)

    if matched_len > 0
        # Restore cached KV
        restore_kv!(cache, pcache, kv_segments)
        # Only need to process the unmatched suffix
        remaining_ids = prompt_ids[matched_len+1:end]
    else
        remaining_ids = prompt_ids
    end

    generated = Int[]

    # Prefill the remaining prompt tokens
    if !isempty(remaining_ids)
        prompt_gpu = MtlArray(Int32.(remaining_ids))
        logits = forward(model, prompt_gpu, cache)
        Metal.synchronize()
    else
        # All tokens were cached, but we need to run one forward pass
        # to get the logits for the last cached token.
        # Actually, we need the logits from the last token. Let's re-run
        # just the last token to get its logits.
        last_tok = MtlArray(Int32[prompt_ids[end]])
        # Back up cache.seq_len by 1 to recompute last position
        cache.seq_len -= 1
        logits = forward(model, last_tok, cache)
        Metal.synchronize()
    end

    # Get next token
    next_token = argmax_last_col_cpu(logits)
    push!(generated, Int(next_token))

    # Decode loop
    for step in 1:max_tokens-1
        if Int(next_token) in config.eos_token_ids
            break
        end

        token_gpu = MtlArray(Int32[next_token])
        logits = forward(model, token_gpu, cache)
        Metal.synchronize()

        next_token = argmax_last_col_cpu(logits)
        push!(generated, Int(next_token))
    end

    # Insert the full sequence (prompt + generated) into the prefix cache
    full_sequence = vcat(prompt_ids, generated)
    insert_prefix!(pcache, prompt_ids, cache, matched_len)

    return generated
end
