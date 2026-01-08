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
        assert cache.enable_coalescing is True

        cache_no_coalesce = ReleaseCache(enable_coalescing=False)
        assert cache_no_coalesce.enable_coalescing is False

    def test_add_and_add_batch(self):
        """Test adding blocks to the release cache."""
        cache = ReleaseCache()

        # Test add single block
        cache.add(KVCacheBlock(block_id=5))
        assert len(cache) == 1

        # Test add batch
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        cache.add_batch(blocks)
        assert len(cache) == 4

    def test_pop_all_with_coalescing(self):
        """Test pop_all with coalescing enabled (sorted by block_id)."""
        cache = ReleaseCache(enable_coalescing=True)
        blocks = [KVCacheBlock(block_id=i) for i in [5, 2, 8, 1, 3]]
        cache.add_batch(blocks)

        popped = cache.pop_all()
        assert len(cache) == 0
        assert [b.block_id for b in popped] == [1, 2, 3, 5, 8]

    def test_pop_all_without_coalescing(self):
        """Test pop_all without coalescing (original order preserved)."""
        cache = ReleaseCache(enable_coalescing=False)
        blocks = [KVCacheBlock(block_id=i) for i in [5, 2, 8, 1, 3]]
        cache.add_batch(blocks)

        popped = cache.pop_all()
        assert [b.block_id for b in popped] == [5, 2, 8, 1, 3]

    def test_clear(self):
        """Test clearing the release cache."""
        cache = ReleaseCache()
        cache.add_batch([KVCacheBlock(block_id=i) for i in range(5)])
        cache.clear()
        assert len(cache) == 0


class TestFreeKVCacheBlockQueueDeferredRelease:
    """Tests for deferred release methods in FreeKVCacheBlockQueue."""

    def test_deferred_free_enabled(self):
        """Test deferred free when enabled."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(blocks, enable_deferred_release=True)

        allocated = queue.popleft_n(5)
        assert queue.num_free_blocks == 5

        # Single block deferred free
        queue.deferred_free(allocated[0])
        assert queue.num_free_blocks == 5  # Not added to free pool yet
        assert queue.release_cache_size == 1

        # Multiple blocks deferred free
        queue.deferred_free_n(allocated[1:])
        assert queue.num_free_blocks == 5
        assert queue.release_cache_size == 5

    def test_deferred_free_disabled(self):
        """Test deferred_free falls back to immediate release when disabled."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(blocks, enable_deferred_release=False)

        allocated = queue.popleft_n(3)
        assert queue.num_free_blocks == 7

        queue.deferred_free(allocated[0])
        assert queue.num_free_blocks == 8  # Added directly to free pool
        assert queue.release_cache_size == 0

    def test_merge_release_cache(self):
        """Test merging release cache into free pool."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks, enable_deferred_release=True, enable_coalescing=True
        )

        allocated = queue.popleft_n(5)
        queue.deferred_free_n(allocated)

        num_merged = queue.merge_release_cache()
        assert num_merged == 5
        assert queue.num_free_blocks == 10
        assert queue.release_cache_size == 0

    def test_try_allocate_with_merge(self):
        """Test allocation that triggers merge when free pool is insufficient."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(blocks, enable_deferred_release=True)

        # Allocate all, then deferred free some
        allocated = queue.popleft_n(10)
        queue.deferred_free_n(allocated[:5])
        assert queue.num_free_blocks == 0
        assert queue.release_cache_size == 5

        # Should merge and allocate
        result = queue.try_allocate_with_merge(3)
        assert result is not None
        assert len(result) == 3
        assert queue.num_free_blocks == 2
        assert queue.release_cache_size == 0

        # Insufficient even after merge
        result = queue.try_allocate_with_merge(5)
        assert result is None

    def test_get_total_available_blocks(self):
        """Test getting total available blocks count."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(blocks, enable_deferred_release=True)

        allocated = queue.popleft_n(3)
        assert queue.get_total_available_blocks() == 7

        queue.deferred_free_n(allocated)
        assert queue.get_total_available_blocks() == 10

    def test_get_contiguous_free_ranges(self):
        """Test getting contiguous free block ranges."""
        blocks = [KVCacheBlock(block_id=i) for i in range(10)]
        queue = FreeKVCacheBlockQueue(
            blocks, enable_deferred_release=True, enable_coalescing=True
        )

        # Allocate all, then free non-contiguous blocks: 0, 1, 2, 5, 6, 9
        all_blocks = queue.popleft_n(10)
        queue.deferred_free_n([all_blocks[i] for i in [0, 1, 2, 5, 6, 9]])
        queue.merge_release_cache()

        ranges = queue.get_contiguous_free_ranges()
        assert len(ranges) == 3
        assert (0, 3) in ranges  # blocks 0, 1, 2
        assert (5, 2) in ranges  # blocks 5, 6
        assert (9, 1) in ranges  # block 9


class TestBlockPoolDeferredRelease:
    """Tests for BlockPool with deferred release enabled."""

    def test_free_blocks_deferred_release(self):
        """Test free_blocks uses deferred release when enabled."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        blocks = pool.get_new_blocks(5)
        initial_free = pool.get_num_free_blocks_in_pool()

        pool.free_blocks(blocks)

        assert pool.get_num_free_blocks_in_pool() == initial_free
        assert pool.get_num_blocks_in_release_cache() == 5
        assert pool.get_num_free_blocks() == initial_free + 5

    def test_free_blocks_immediate_when_disabled(self):
        """Test free_blocks is immediate when deferred release is disabled."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=False,
        )

        blocks = pool.get_new_blocks(5)
        initial_free = pool.get_num_free_blocks()

        pool.free_blocks(blocks)

        assert pool.get_num_free_blocks() == initial_free + 5
        assert pool.get_num_blocks_in_release_cache() == 0

    def test_get_new_blocks_with_merge(self):
        """Test get_new_blocks_with_merge triggers merge when needed."""
        pool = BlockPool(
            num_gpu_blocks=20,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
        )

        # Allocate all, deferred free some
        initial_blocks = pool.get_new_blocks(pool.get_num_free_blocks())
        pool.free_blocks(initial_blocks[:10])
        assert pool.get_num_free_blocks_in_pool() == 0
        assert pool.get_num_blocks_in_release_cache() == 10

        # Should merge and allocate
        result = pool.get_new_blocks_with_merge(5)
        assert result is not None
        assert len(result) == 5
        assert pool.get_num_blocks_in_release_cache() == 0

        # Insufficient blocks
        result = pool.get_new_blocks_with_merge(100)
        assert result is None


class TestDeferredReleaseIntegration:
    """Integration tests for deferred release mechanism."""

    def test_high_concurrency_simulation(self):
        """Simulate high-concurrency allocation/deallocation patterns."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=False,
            hash_block_size=16,
            enable_deferred_release=True,
            enable_coalescing=True,
        )

        active_allocations = []
        for _ in range(10):
            blocks = pool.get_new_blocks_with_merge(5)
            if blocks:
                active_allocations.append(blocks)
            if len(active_allocations) > 3:
                pool.free_blocks(active_allocations.pop(0))

        # Cleanup
        for blocks in active_allocations:
            pool.free_blocks(blocks)
        pool.merge_release_cache()

        # All blocks should be back (minus null block)
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

        blocks = pool.get_new_blocks(50)
        pool.free_blocks(blocks)
        pool.merge_release_cache()

        ranges = pool.get_contiguous_free_ranges()
        total_in_ranges = sum(length for _, length in ranges)
        assert total_in_ranges == pool.get_num_free_blocks()
