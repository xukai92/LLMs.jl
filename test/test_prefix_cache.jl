using Test
using Metal
using LLMs

# Access internals for testing
import LLMs: RadixNode, KVPool, PrefixCache, LlamaConfig
import LLMs: pool_alloc!, pool_free!, pool_used
import LLMs: prefix_match, insert_prefix!, restore_kv!, collect_leaves, evict_lru!

@testset "Phase 3: Prefix Cache" begin

    # ═══ Pool allocator ═══
    @testset "KVPool allocation" begin
        # Minimal config for testing
        config = LlamaConfig(
            64, 128, 2, 4, 4, 16, 100, 512,
            1f-5, 10000.0, nothing, true, 1, [2],
            4, 64
        )
        pool = KVPool(config; max_tokens=100)

        # Allocate some slots
        o1 = pool_alloc!(pool, 20)
        @test o1 == 0
        @test pool.next_free == 20

        o2 = pool_alloc!(pool, 30)
        @test o2 == 20
        @test pool.next_free == 50

        # Free first allocation
        pool_free!(pool, 0, 20)
        @test length(pool.free_ranges) == 1

        # Reallocate from free list
        o3 = pool_alloc!(pool, 15)
        @test o3 == 0  # reuses freed slot

        # Remaining free space
        o4 = pool_alloc!(pool, 5)
        @test o4 == 15  # remainder of freed range

        # Allocate beyond pool
        o5 = pool_alloc!(pool, 60)
        @test o5 == -1  # pool full (50 + 60 > 100)

        o6 = pool_alloc!(pool, 50)
        @test o6 == 50  # fits exactly
    end

    # ═══ Radix tree operations ═══
    @testset "Prefix matching" begin
        config = LlamaConfig(
            64, 128, 2, 4, 4, 16, 100, 512,
            1f-5, 10000.0, nothing, true, 1, [2],
            4, 64
        )
        cache = PrefixCache(config; max_tokens=1000)

        # Empty tree — no match
        matched, node, segs = prefix_match(cache, [10, 20, 30, 40])
        @test matched == 0
        @test isempty(segs)

        # Insert a sequence manually
        child = RadixNode(; tokens=[10, 20, 30], pool_offset=0, pool_length=3, parent=cache.root)
        cache.root.children[10] = child
        cache.pool.next_free = 3

        # Full prefix match
        matched, node, segs = prefix_match(cache, [10, 20, 30, 40, 50])
        @test matched == 3
        @test length(segs) == 1
        @test segs[1] == (0, 3)

        # Partial match (only first 2 of 3 edge tokens match — no match)
        matched, _, _ = prefix_match(cache, [10, 20, 99])
        @test matched == 0  # Can't partially match an edge

        # Exact match
        matched, _, segs = prefix_match(cache, [10, 20, 30])
        @test matched == 3

        # No match at all
        matched, _, _ = prefix_match(cache, [99, 100])
        @test matched == 0
    end

    @testset "Insert and match multi-level" begin
        config = LlamaConfig(
            64, 128, 2, 4, 4, 16, 100, 512,
            1f-5, 10000.0, nothing, true, 1, [2],
            4, 64
        )

        # Build a mock KVCache with known data
        kv = KVCache(config; max_seq_len=100)
        # Fill with distinguishable data
        for layer in 1:2
            for pos in 1:10
                fill!(view(Array(kv.k_cache[layer]), :, :, pos:pos), Float16(pos))
                copyto!(view(kv.k_cache[layer], :, :, pos:pos),
                       MtlArray(fill(Float16(pos), 16, 4, 1)))
                copyto!(view(kv.v_cache[layer], :, :, pos:pos),
                       MtlArray(fill(Float16(pos * 10), 16, 4, 1)))
            end
        end
        kv.seq_len = 10

        pcache = PrefixCache(config; max_tokens=1000)

        # Insert first sequence [1,2,3,4,5]
        insert_prefix!(pcache, [1, 2, 3, 4, 5], kv, 0)

        # Verify pool was used
        @test pcache.pool.next_free == 5

        # Match against it
        matched, _, segs = prefix_match(pcache, [1, 2, 3, 4, 5, 6, 7])
        @test matched == 5

        # Insert overlapping sequence [1,2,3,10,11] (shares first 3 tokens)
        kv2 = KVCache(config; max_seq_len=100)
        for layer in 1:2
            for pos in 1:5
                copyto!(view(kv2.k_cache[layer], :, :, pos:pos),
                       MtlArray(fill(Float16(pos + 100), 16, 4, 1)))
                copyto!(view(kv2.v_cache[layer], :, :, pos:pos),
                       MtlArray(fill(Float16(pos * 10 + 100), 16, 4, 1)))
            end
        end
        kv2.seq_len = 5

        # Insert from position 3 (first 3 tokens [1,2,3] already cached)
        insert_prefix!(pcache, [1, 2, 3, 10, 11], kv2, 3)

        # Verify the tree now has a split
        # Root -> [1,2,3] -> [4,5]  (original)
        #                 -> [10,11] (new)
        @test haskey(pcache.root.children, 1)
        split_node = pcache.root.children[1]
        @test split_node.tokens == [1, 2, 3]
        @test length(split_node.children) == 2
        @test haskey(split_node.children, 4)
        @test haskey(split_node.children, 10)

        # Match the new branch
        matched, _, _ = prefix_match(pcache, [1, 2, 3, 10, 11, 12])
        @test matched == 5
    end

    @testset "KV restore correctness" begin
        config = LlamaConfig(
            64, 128, 2, 4, 4, 16, 100, 512,
            1f-5, 10000.0, nothing, true, 1, [2],
            4, 64
        )

        # Create source KV with known values
        kv_src = KVCache(config; max_seq_len=100)
        test_val_k = Float16(42.0)
        test_val_v = Float16(84.0)
        for layer in 1:2
            copyto!(view(kv_src.k_cache[layer], :, :, 1:5),
                   MtlArray(fill(test_val_k, 16, 4, 5)))
            copyto!(view(kv_src.v_cache[layer], :, :, 1:5),
                   MtlArray(fill(test_val_v, 16, 4, 5)))
        end
        kv_src.seq_len = 5

        pcache = PrefixCache(config; max_tokens=1000)
        insert_prefix!(pcache, [1, 2, 3, 4, 5], kv_src, 0)

        # Restore into a fresh KV cache
        kv_dst = KVCache(config; max_seq_len=100)
        _, _, segs = prefix_match(pcache, [1, 2, 3, 4, 5])
        restore_kv!(kv_dst, pcache, segs)

        @test kv_dst.seq_len == 5

        # Verify data was copied correctly
        k_data = Array(kv_dst.k_cache[1])[:, :, 1:5]
        v_data = Array(kv_dst.v_cache[1])[:, :, 1:5]
        @test all(k_data .== test_val_k)
        @test all(v_data .== test_val_v)
    end

    @testset "LRU eviction" begin
        config = LlamaConfig(
            64, 128, 2, 4, 4, 16, 100, 512,
            1f-5, 10000.0, nothing, true, 1, [2],
            4, 64
        )
        # Pool of 30: can hold 10+5+10=25, but inserting a 4th forces eviction
        pcache = PrefixCache(config; max_tokens=30)

        kv = KVCache(config; max_seq_len=100)
        kv.seq_len = 15

        # Insert first sequence (uses 10 tokens)
        insert_prefix!(pcache, collect(1:10), kv, 0)
        @test pcache.pool.next_free == 10

        # Insert second sequence (uses 5 tokens)
        kv2 = KVCache(config; max_seq_len=100)
        kv2.seq_len = 5
        insert_prefix!(pcache, collect(101:105), kv2, 0)
        @test pcache.pool.next_free == 15

        # Touch the first sequence to make it more recent
        sleep(0.01)
        prefix_match(pcache, collect(1:10))

        # Insert third sequence (10 tokens) — fits at next_free=15, total=25 < 30
        kv3 = KVCache(config; max_seq_len=100)
        kv3.seq_len = 10
        insert_prefix!(pcache, collect(201:210), kv3, 0)
        @test pcache.pool.next_free == 25

        # Now insert 10 more — pool has 25 used, needs 10, total 35 > 30
        # Should evict the oldest untouched leaf: [101..105] (5 tokens)
        # Then the oldest remaining: [201..210] (10 tokens from kv3, inserted last but not touched)
        # Actually [201..210] was just inserted — let's touch [1..10] again
        sleep(0.01)
        prefix_match(pcache, collect(1:10))

        kv4 = KVCache(config; max_seq_len=100)
        kv4.seq_len = 10
        insert_prefix!(pcache, collect(301:310), kv4, 0)

        # Second sequence [101..105] was oldest, should be evicted
        matched, _, _ = prefix_match(pcache, collect(101:105))
        @test matched == 0  # evicted

        # First sequence should still be there (was touched most recently)
        matched, _, _ = prefix_match(pcache, collect(1:10))
        @test matched == 10  # still cached
    end
end
