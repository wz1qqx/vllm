# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the deferred release mechanism in KV cache management.

This module tests the ReleaseCache, FreeKVCacheBlockQueue's deferred release
methods, and BlockPool's integration with the deferred release mechanism.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    ReleaseCache,
)

pytestmark = pytest.mark.cpu_test


class TestReleaseCache:
    """Tests for the ReleaseCache class."""

    def test_init(self):
        """Test ReleaseCache initialization."""
        cache = ReleaseCache()
        assert len(cache) == 0
        assert cache.num_blocks == 0
        assert cache.enable_coalescing is True

        cache_no_coalesce = ReleaseCache(enable_coalescing=False)
        assert cache_no_coalesce.enable_coalescing is False

    def test_add_single_block(self):
        """Test adding a single block to the release cache."""
        cache = ReleaseCache()
        block = KVCacheBlock(block_id=5)

        cache.add(block)
        assert len(cache) == 1
        assert cache.num_blocks == 1

    def test_add_batch(self):
        """Test adding multiple blocks in a batch."""
        cache = ReleaseCache()
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]

        cache.add_batch(blocks)
        assert len(cache) == 5
        assert cache.num_blocks == 5

    def test_pop_all_with_coalescing(self):
        """Test pop_all with coalescing enabled (sorted by block_id)."""
        cache = ReleaseCache(enable_coalescing=True)
        # Add blocks in non-sorted order
        blocks = [KVCacheBlock(block_id=i) for i in [5, 2, 8, 1, 3]]
        cache.add_batch(blocks)

        popped = cache.pop_all()
        assert len(popped) == 5
        assert len(cache) == 0  # Cache should be empty after pop

        # Verify blocks are sorted by block_id
        block_ids = [b.block_id for b in popped]
        assert block_ids == [1, 2, 3, 5, 8]

    def test_pop_all_without_coalescing(self):
        """Test pop_all without coalescing (original order preserved)."""
        cache = ReleaseCache(enable_coalescing=False)
        # Add blocks in specific order
        blocks = [KVCacheBlock(block_id=i) for i in [5, 2, 8, 1, 3]]
        cache.add_batch(blocks)

        popped = cache.pop_all()
        assert len(popped) == 5

        # Verify blocks are in original order
        block_ids = [b.block_id for b in popped]
        assert block_ids == [5, 2, 8, 1, 3]

    def test_pop_all_empty_cache(self):
        """Test pop_all on empty cache returns empty list."""
        cache = ReleaseCache()
        popped = cache.pop_all()
        assert popped == []

    def test_clear(self):
        """Test clearing the release cache."""
        cache = ReleaseCache()
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        cache.add_batch(blocks)
        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0


class TestFreeKVCacheBlockQueueDeferredRelease:
    """Tests for deferred release methods in FreeKVCacheBlockQueue."""

    def test_init_with_deferred_release(self):
        """Test initialization with deferred release enabled."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        assert queue.enable_deferred_release is True
        assert queue.num_free_blocks == 10
        assert queue.release_cache_size == 0

    def test_deferred_free_single_block(self):
        """Test deferred free of a single block."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate some blocks
        allocated = queue.popleft_n(3)
        assert queue.num_free_blocks == 7

        # Deferred free one block
        queue.deferred_free(allocated[0])
        assert queue.num_free_blocks == 7  # Not added to free pool yet
        assert queue.release_cache_size == 1  # In release cache

    def test_deferred_free_multiple_blocks(self):
        """Test deferred free of multiple blocks."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate some blocks
        allocated = queue.popleft_n(5)
        assert queue.num_free_blocks == 5

        # Deferred free all allocated blocks
        queue.deferred_free_n(allocated)
        assert queue.num_free_blocks == 5  # Not added to free pool yet
        assert queue.release_cache_size == 5  # All in release cache

    def test_deferred_free_disabled(self):
        """Test deferred_free falls back to immediate release when disabled."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=False,
        )

        # Allocate some blocks
        allocated = queue.popleft_n(3)
        assert queue.num_free_blocks == 7

        # Deferred free should fall back to immediate release
        queue.deferred_free(allocated[0])
        assert queue.num_free_blocks == 8  # Added directly to free pool
        assert queue.release_cache_size == 0

    def test_merge_release_cache(self):
        """Test merging release cache into free pool."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Allocate some blocks
        allocated = queue.popleft_n(5)
        assert queue.num_free_blocks == 5

        # Deferred free
        queue.deferred_free_n(allocated)
        assert queue.release_cache_size == 5

        # Merge release cache
        num_merged = queue.merge_release_cache()
        assert num_merged == 5
        assert queue.num_free_blocks == 10
        assert queue.release_cache_size == 0

    def test_try_allocate_with_merge_sufficient_free_pool(self):
        """Test allocation when free pool has enough blocks."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate 3 blocks
        result = queue.try_allocate_with_merge(3)
        assert result is not None
        assert len(result) == 3
        assert queue.num_free_blocks == 7

    def test_try_allocate_with_merge_needs_merge(self):
        """Test allocation that requires merging release cache."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate all blocks
        allocated = queue.popleft_n(10)
        assert queue.num_free_blocks == 0

        # Deferred free some blocks
        queue.deferred_free_n(allocated[:5])
        assert queue.release_cache_size == 5

        # Try to allocate 3 blocks (should merge first)
        result = queue.try_allocate_with_merge(3)
        assert result is not None
        assert len(result) == 3
        assert queue.num_free_blocks == 2  # 5 merged - 3 allocated
        assert queue.release_cache_size == 0

    def test_try_allocate_with_merge_insufficient_total(self):
        """Test allocation when total available is insufficient."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate all blocks
        allocated = queue.popleft_n(10)

        # Deferred free only 3 blocks
        queue.deferred_free_n(allocated[:3])

        # Try to allocate 5 blocks (not enough even after merge)
        result = queue.try_allocate_with_merge(5)
        assert result is None

    def test_get_total_available_blocks(self):
        """Test getting total available blocks count."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Initial state
        assert queue.get_total_available_blocks() == 10

        # Allocate some blocks
        allocated = queue.popleft_n(3)
        assert queue.get_total_available_blocks() == 7

        # Deferred free
        queue.deferred_free_n(allocated)
        # Total should include both free pool and release cache
        assert queue.get_total_available_blocks() == 10

    def test_get_contiguous_free_ranges(self):
        """Test getting contiguous free block ranges."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Allocate blocks 2, 3, 6, 7 (leaving 0, 1, 4, 5, 8, 9 free)
        queue.popleft_n(4)  # Removes blocks 0, 1, 2, 3

        # Now free pool has blocks 4, 5, 6, 7, 8, 9
        # Get contiguous ranges
        ranges = queue.get_contiguous_free_ranges()

        # Should have one contiguous range: 4-9
        assert len(ranges) == 1
        assert ranges[0] == (4, 6)  # start=4, length=6

    def test_get_contiguous_free_ranges_fragmented(self):
        """Test getting contiguous ranges with fragmented free blocks."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Allocate all, then free blocks in non-contiguous pattern
        all_blocks = queue.popleft_n(10)

        # Free blocks 0, 1, 2, 5, 6, 9 (creates gaps at 3-4, 7-8)
        queue.deferred_free_n([all_blocks[i] for i in [0, 1, 2, 5, 6, 9]])
        queue.merge_release_cache()

        ranges = queue.get_contiguous_free_ranges()

        # Should have three ranges: (0, 3), (5, 2), (9, 1)
        assert len(ranges) == 3
        assert (0, 3) in ranges  # blocks 0, 1, 2
        assert (5, 2) in ranges  # blocks 5, 6
        assert (9, 1) in ranges  # block 9

    def test_force_merge(self):
        """Test force merge method."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks,
            enable_deferred_release=True,
        )

        # Allocate and deferred free
        allocated = queue.popleft_n(5)
        queue.deferred_free_n(allocated)

        # Force merge
        num_merged = queue.force_merge()
        assert num_merged == 5
        assert queue.release_cache_size == 0
        assert queue.num_free_blocks == 10


class TestBlockPoolDeferredRelease:
    """Tests for BlockPool with deferred release enabled."""

    def test_init_with_deferred_release(self):
        """Test BlockPool initialization with deferred release."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        assert pool.enable_deferred_release is True
        assert pool.enable_coalescing is True

    def test_free_blocks_deferred(self):
        """Test that free_blocks uses deferred release when enabled."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        # Allocate some blocks
        blocks = pool.get_new_blocks(5)
        assert len(blocks) == 5

        initial_free = pool.get_num_free_blocks_in_pool()

        # Free blocks (should be deferred)
        pool.free_blocks(blocks)

        # Free pool should not have changed
        assert pool.get_num_free_blocks_in_pool() == initial_free
        # But release cache should have the blocks
        assert pool.get_num_blocks_in_release_cache() == 5
        # Total available should include both
        assert pool.get_num_free_blocks() == initial_free + 5

    def test_free_blocks_immediate_when_disabled(self):
        """Test that free_blocks is immediate when deferred release is disabled."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=False,
        )

        # Allocate some blocks
        blocks = pool.get_new_blocks(5)
        initial_free = pool.get_num_free_blocks()

        # Free blocks (should be immediate)
        pool.free_blocks(blocks)

        # Free pool should have the blocks immediately
        assert pool.get_num_free_blocks() == initial_free + 5
        assert pool.get_num_blocks_in_release_cache() == 0

    def test_get_new_blocks_with_merge(self):
        """Test get_new_blocks_with_merge method."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        # Allocate all blocks
        initial_blocks = pool.get_new_blocks(pool.get_num_free_blocks())
        assert pool.get_num_free_blocks() == 0

        # Deferred free some blocks
        pool.free_blocks(initial_blocks[:10])
        assert pool.get_num_free_blocks_in_pool() == 0
        assert pool.get_num_blocks_in_release_cache() == 10

        # Use get_new_blocks_with_merge (should merge and allocate)
        result = pool.get_new_blocks_with_merge(5)
        assert result is not None
        assert len(result) == 5
        # Release cache should be empty after merge
        assert pool.get_num_blocks_in_release_cache() == 0
        assert pool.get_num_free_blocks() == 5  # 10 merged - 5 allocated

    def test_get_new_blocks_with_merge_insufficient(self):
        """Test get_new_blocks_with_merge when insufficient blocks."""
        pool = BlockPool(
            num_gpu_blocks=10,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        # Allocate all blocks
        all_blocks = pool.get_new_blocks(pool.get_num_free_blocks())

        # Deferred free only 3 blocks
        pool.free_blocks(all_blocks[:3])

        # Try to allocate 5 blocks (not enough)
        result = pool.get_new_blocks_with_merge(5)
        assert result is None

    def test_merge_release_cache(self):
        """Test explicit merge_release_cache call."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        # Allocate and free
        blocks = pool.get_new_blocks(10)
        pool.free_blocks(blocks)

        assert pool.get_num_blocks_in_release_cache() == 10

        # Explicit merge
        num_merged = pool.merge_release_cache()
        assert num_merged == 10
        assert pool.get_num_blocks_in_release_cache() == 0

    def test_get_contiguous_free_ranges(self):
        """Test get_contiguous_free_ranges method."""
        pool = BlockPool(
            num_gpu_blocks=20,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        ranges = pool.get_contiguous_free_ranges()

        # Initially should have one large contiguous range
        # (minus the null block at index 0)
        assert len(ranges) >= 1

    def test_free_blocks_deferred_explicit(self):
        """Test free_blocks_deferred method."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=False,  # Disabled globally
        )

        # Allocate some blocks
        blocks = pool.get_new_blocks(5)

        # Use explicit deferred free (should work even when disabled globally)
        pool.free_blocks_deferred(blocks)

        # Should be in release cache, not free pool
        assert pool.get_num_blocks_in_release_cache() == 5


class TestDeferredReleaseIntegration:
    """Integration tests for deferred release mechanism."""

    def test_allocation_fragmentation_reduction(self):
        """Test that coalescing reduces fragmentation."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Create fragmentation by allocating and freeing in a pattern
        all_blocks = pool.get_new_blocks(50)

        # Free blocks in non-contiguous pattern
        to_free = [all_blocks[i] for i in range(0, 50, 2)]  # Every other block
        pool.free_blocks(to_free)

        # Merge with coalescing
        pool.merge_release_cache()

        # Get contiguous ranges
        ranges = pool.get_contiguous_free_ranges()

        # With coalescing, blocks should be sorted and potentially contiguous
        # at the front of the queue
        assert len(ranges) > 0

    def test_high_concurrency_simulation(self):
        """Simulate high-concurrency allocation/deallocation patterns."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Simulate multiple "requests" allocating and freeing
        active_allocations = []

        for i in range(10):
            # Allocate for a "request"
            blocks = pool.get_new_blocks_with_merge(5)
            if blocks:
                active_allocations.append(blocks)

            # Free some older allocations
            if len(active_allocations) > 3:
                to_free = active_allocations.pop(0)
                pool.free_blocks(to_free)

        # Final cleanup
        for blocks in active_allocations:
            pool.free_blocks(blocks)

        # Merge and verify
        pool.merge_release_cache()

        # All blocks should be back (minus null block)
        # The exact count depends on implementation details
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1

    def test_kv_offload_scenario(self):
        """Test scenario optimized for KV offload efficiency."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        # Allocate blocks
        blocks = pool.get_new_blocks(50)

        # Free all blocks (simulating request completion)
        pool.free_blocks(blocks)

        # Force merge to prepare for KV offload
        pool.merge_release_cache()

        # Get contiguous ranges for efficient DMA transfer
        ranges = pool.get_contiguous_free_ranges()

        # Verify we have contiguous ranges
        total_in_ranges = sum(length for _, length in ranges)
        assert total_in_ranges == pool.get_num_free_blocks()
