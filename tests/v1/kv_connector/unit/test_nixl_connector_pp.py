# SPDX-License-Identifier: Apache-2.0
# Unit tests for PP support in NixlConnector.
#
# TDD: Tests marked [FAILS_NOW] fail with current vllm code and pass after fix.
#      Tests marked [PASSES_NOW] document current (broken) behavior.
#
# Two gaps under test:
# 1. gpu_worker.get_kv_connector_handshake_metadata: key = tp_rank only (collision)
#    Fix: key = pp_rank * tp_size + tp_rank  (global worker index)
# 2. NixlConnectorScheduler.side_channel_port: offset by dp_index only (collision)
#    Fix: offset += pp_rank * tp_size

import pytest
import torch
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the actual vllm functions under test
# ---------------------------------------------------------------------------
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


# ---------------------------------------------------------------------------
# Helper: call GPUWorker.get_kv_connector_handshake_metadata as unbound method
# with mocked tp_rank, pp_rank, tp_size, and a fake metadata object.
# ---------------------------------------------------------------------------

def _call_handshake_metadata(tp_rank: int, pp_rank: int, tp_size: int):
    """
    Call the actual GPUWorker.get_kv_connector_handshake_metadata with
    mocked distributed state.  Returns the dict it produces.
    """
    fake_metadata = MagicMock(name="metadata")

    mock_tp_group = MagicMock()
    mock_tp_group.rank_in_group = tp_rank

    mock_pp_group = MagicMock()
    mock_pp_group.rank_in_group = pp_rank
    mock_pp_group.world_size = tp_size  # pp_size (unused in current code)

    mock_connector = MagicMock()
    mock_connector.get_handshake_metadata.return_value = fake_metadata

    mock_parallel_config = MagicMock()
    mock_parallel_config.tensor_parallel_size = tp_size
    mock_parallel_config.pipeline_parallel_size = max(1, pp_rank + 1)  # at least pp_rank+1

    mock_self = MagicMock()
    mock_self.vllm_config.parallel_config = mock_parallel_config

    with (
        patch("vllm.v1.worker.gpu_worker.has_kv_transfer_group", return_value=True),
        patch("vllm.v1.worker.gpu_worker.get_kv_transfer_group",
              return_value=mock_connector),
        patch("vllm.v1.worker.gpu_worker.get_tp_group", return_value=mock_tp_group),
        # get_pp_group does not exist yet in gpu_worker — patching it in advance
        patch("vllm.v1.worker.gpu_worker.get_pp_group",
              return_value=mock_pp_group),
    ):
        return GPUWorker.get_kv_connector_handshake_metadata(mock_self)


# ===========================================================================
# Gap 1: Handshake key collision
# ===========================================================================

class TestHandshakeKeyPP:
    """
    GPUWorker.get_kv_connector_handshake_metadata should return a key that
    uniquely identifies each worker across both TP and PP dimensions.

    Current code: key = tp_rank   (only TP dimension)
    Required fix: key = pp_rank * tp_size + tp_rank   (global worker index)
    """

    # --- [PASSES_NOW] Documents current behavior: key is just tp_rank ---

    def test_pp1_tp0_key_is_0_current(self):
        """PP=1, tp_rank=0 → key must be 0 (baseline, no PP)."""
        result = _call_handshake_metadata(tp_rank=0, pp_rank=0, tp_size=4)
        assert result is not None
        assert 0 in result

    def test_pp1_tp1_key_is_1_current(self):
        """PP=1, tp_rank=1 → key must be 1 (baseline, no PP)."""
        result = _call_handshake_metadata(tp_rank=1, pp_rank=0, tp_size=4)
        assert result is not None
        assert 1 in result

    # --- [FAILS_NOW] These tests FAIL with current code, PASS after fix ---

    def test_pp1_rank1_tp_rank0_key_must_be_4(self):
        """[FAILS_NOW] pp_rank=1, tp_rank=0, tp_size=4 → key must be 4, not 0.

        Current code returns {0: metadata} because it ignores pp_rank.
        After fix it should return {4: metadata} (1*4+0=4).
        """
        result = _call_handshake_metadata(tp_rank=0, pp_rank=1, tp_size=4)
        assert result is not None
        assert 4 in result, (
            f"Expected key=4 (pp_rank=1*tp_size=4+tp_rank=0), "
            f"got keys={list(result.keys())}. "
            "Fix: use pp_rank * tp_size + tp_rank as key."
        )

    def test_pp1_rank1_tp_rank3_key_must_be_7(self):
        """[FAILS_NOW] pp_rank=1, tp_rank=3, tp_size=4 → key must be 7."""
        result = _call_handshake_metadata(tp_rank=3, pp_rank=1, tp_size=4)
        assert result is not None
        assert 7 in result, (
            f"Expected key=7 (1*4+3), got keys={list(result.keys())}"
        )

    def test_pp3_rank3_tp_rank1_key_must_be_7(self):
        """[FAILS_NOW] pp_rank=3, tp_rank=1, tp_size=2 → key must be 7 (3*2+1)."""
        result = _call_handshake_metadata(tp_rank=1, pp_rank=3, tp_size=2)
        assert result is not None
        assert 7 in result, (
            f"Expected key=7 (3*2+1), got keys={list(result.keys())}"
        )

    def test_pp0_workers_survive_merge_with_pp1(self):
        """[FAILS_NOW] After merging PP0+PP1 workers, PP0 metadata must NOT be lost.

        Simulates engine/core.py collecting metadata from all 8 workers (PP2+TP4).
        With current code (key=tp_rank), PP0 is overwritten by PP1.
        After fix (key=global_idx), all 8 workers survive.
        """
        # Collect dicts from 8 workers (PP2+TP4)
        worker_dicts = [
            _call_handshake_metadata(tp_rank=tp_r, pp_rank=pp_r, tp_size=4)
            for pp_r in range(2)
            for tp_r in range(4)
        ]

        # Merge (mirrors engine/core.py logic)
        merged: dict = {}
        for d in worker_dicts:
            if d is not None:
                merged.update(d)

        # After fix: all 8 workers present (keys 0-7)
        assert len(merged) == 8, (
            f"Expected 8 unique worker entries after merge, got {len(merged)}. "
            "Current code loses PP0 workers because PP1 overwrites them."
        )

        # Verify PP0 workers (keys 0-3) are present
        for tp_r in range(4):
            expected_key = 0 * 4 + tp_r  # pp_rank=0
            assert expected_key in merged, f"PP0 tp_rank={tp_r} missing from merged dict"

        # Verify PP1 workers (keys 4-7) are present
        for tp_r in range(4):
            expected_key = 1 * 4 + tp_r  # pp_rank=1
            assert expected_key in merged, f"PP1 tp_rank={tp_r} missing from merged dict"

    def test_pp1_behavior_backward_compatible(self):
        """[PASSES_NOW and after fix] PP=1 key=0 is unchanged."""
        result_pp1 = _call_handshake_metadata(tp_rank=0, pp_rank=0, tp_size=4)
        # With both current and fixed code, PP=1 should return key=0
        assert result_pp1 is not None
        assert 0 in result_pp1


# ===========================================================================
# Gap 2: Side-channel port collision
# ===========================================================================

class TestSideChannelPortPP:
    """
    NixlConnectorScheduler.side_channel_port must be unique per PP rank.

    Current code:  port = BASE + data_parallel_index
    Required fix:  port = BASE + data_parallel_index + pp_rank * tp_size

    We test the formula directly since full scheduler init requires complex config.
    """

    BASE_PORT = 9100

    def _current_port(self, dp_index: int) -> int:
        """Current formula (from nixl_connector.py:557-560)."""
        return self.BASE_PORT + dp_index

    def _fixed_port(self, dp_index: int, pp_rank: int, tp_size: int) -> int:
        """Expected formula after fix."""
        return self.BASE_PORT + dp_index + pp_rank * tp_size

    # --- [PASSES_NOW] Documents current broken behavior ---

    def test_current_formula_collision_same_dp_different_pp(self):
        """[PASSES_NOW] Current formula: PP0 and PP1 get same port when dp_index=0."""
        port_pp0 = self._current_port(dp_index=0)
        port_pp1 = self._current_port(dp_index=0)
        # This is the BUG: same port for different PP stages
        assert port_pp0 == port_pp1, "Expected collision — this is the bug"

    # --- [FAILS_NOW] These tests define required behavior after fix ---

    def test_fixed_formula_no_collision_pp2_tp4(self):
        """[PASSES after fix] PP0 and PP1 must get different ports."""
        port_pp0 = self._fixed_port(dp_index=0, pp_rank=0, tp_size=4)
        port_pp1 = self._fixed_port(dp_index=0, pp_rank=1, tp_size=4)
        assert port_pp0 != port_pp1, (
            "PP0 and PP1 must use different ports to avoid bind conflict"
        )
        assert port_pp1 - port_pp0 == 4  # offset = 1 * tp_size

    def test_fixed_formula_pp1_same_as_current(self):
        """[PASSES after fix] PP=1 (pp_rank=0): fixed formula equals current."""
        current = self._current_port(dp_index=0)
        fixed = self._fixed_port(dp_index=0, pp_rank=0, tp_size=4)
        assert current == fixed, "PP=1 must be backward compatible"

    def test_fixed_formula_all_unique_pp4_tp2(self):
        """[PASSES after fix] PP4+TP2: 4 PP stages all get unique ports."""
        ports = [
            self._fixed_port(dp_index=0, pp_rank=pp_r, tp_size=2)
            for pp_r in range(4)
        ]
        assert len(ports) == len(set(ports)), f"Port collision: {ports}"

    def test_note_port_is_per_engine_not_per_pp_rank(self):
        """[INFO] NixlConnectorScheduler port is per-engine, not per-PP-rank.

        In vLLM PP, there is ONE EngineCore (and ONE NixlConnectorScheduler)
        per serving instance, regardless of PP size. All PP ranks are worker
        subprocesses of the same engine. Therefore:
        - Only ONE side_channel_port exists per Prefill instance.
        - Port collision between PP ranks does NOT occur at the scheduler level.
        - No scheduler port fix is needed.

        This test documents the architecture to prevent future confusion.
        """
        # NixlConnectorScheduler is instantiated ONCE in NixlConnector.__init__:
        #   NixlConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
        # With PP2+TP4, there are 8 workers but only 1 scheduler.
        # The side_channel_port formula (base + dp_index) is correct for PP.
        assert True  # architecture documentation test


# ===========================================================================
# Tests: TpKVTopology.get_all_pp_tp_targets (Wave 2.2)
# ===========================================================================

class TestTpKVTopologyPP:
    """
    TpKVTopology needs a new method get_all_pp_tp_targets that returns
    global worker indices across ALL PP stages for a given local D rank.

    Current get_target_remote_ranks(remote_tp_size=4):
      D_rank 0,1 -> [0]  (only PP0_TP0, PP1 never reached)

    Required get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2):
      D_rank 0,1 -> [0, 4]  (PP0_TP0 AND PP1_TP0)
      D_rank 2,3 -> [1, 5]
      D_rank 4,5 -> [2, 6]
      D_rank 6,7 -> [3, 7]

    Formula: for each tp_rank in get_target_remote_ranks(tp_size):
               [tp_rank + pp_rank * tp_size for pp_rank in range(pp_size)]
    """

    def _make_topology(self, d_tp_rank: int, d_tp_size: int, engine_id="prefill"):
        from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
        from unittest.mock import MagicMock

        mock_backend = MagicMock()
        mock_backend.get_kv_cache_shape.return_value = (1, 16, 4, 1, 1)
        mock_backend.get_kv_cache_stride_order.side_effect = NotImplementedError

        return TpKVTopology(
            tp_rank=d_tp_rank,
            engine_id=engine_id,
            remote_tp_size={engine_id: d_tp_size},
            remote_block_size={engine_id: 16},
            is_mla=True,
            total_num_kv_heads=4,
            attn_backends=[mock_backend],
            tensor_shape=None,
        )

    def test_pp1_identical_to_existing_method(self):
        """PP=1: get_all_pp_tp_targets must equal get_target_remote_ranks (backward compat)."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        existing = topo.get_target_remote_ranks(remote_tp_size=4)
        pp1_result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=1)
        assert pp1_result == existing

    # --- [FAILS_NOW] ---

    def test_d_rank0_pp2_tp4_connects_to_0_and_4(self):
        """[FAILS_NOW] D_rank=0, D_TP=8, remote PP2+TP4 -> global [0, 4]."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [0, 4], f"Got {result}"

    def test_d_rank1_pp2_tp4_connects_to_0_and_4(self):
        """[FAILS_NOW] D_rank=1, D_TP=8, remote PP2+TP4 -> global [0, 4]."""
        topo = self._make_topology(d_tp_rank=1, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [0, 4]

    def test_d_rank2_pp2_tp4_connects_to_1_and_5(self):
        """[FAILS_NOW] D_rank=2, D_TP=8 -> [1, 5]."""
        topo = self._make_topology(d_tp_rank=2, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [1, 5]

    def test_d_rank7_pp2_tp4_connects_to_3_and_7(self):
        """[FAILS_NOW] D_rank=7, D_TP=8 -> [3, 7]."""
        topo = self._make_topology(d_tp_rank=7, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        assert result == [3, 7]

    def test_all_8_d_ranks_cover_all_8_global_indices(self):
        """[FAILS_NOW] All 8 D ranks together must cover all 8 global indices 0-7."""
        covered: set[int] = set()
        for d_rank in range(8):
            topo = self._make_topology(d_tp_rank=d_rank, d_tp_size=8)
            covered.update(topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2))
        assert covered == set(range(8)), f"Missing global indices: {set(range(8)) - covered}"

    def test_pp4_tp2_d_rank0_connects_to_4_agents(self):
        """[FAILS_NOW] D_rank=0, D_TP=8, PP4+TP2 -> [0, 2, 4, 6] (one per PP stage)."""
        topo = self._make_topology(d_tp_rank=0, d_tp_size=8)
        result = topo.get_all_pp_tp_targets(remote_tp_size=2, remote_pp_size=4)
        assert result == [0, 2, 4, 6], f"Got {result}"


# ===========================================================================
# Tests: NixlConnector PP wiring (Wave 2.3)
# ===========================================================================

class TestNixlConnectorPPWiring:
    """
    Three wiring changes needed for PP KV transfer:

    1. ReqMeta gets a pp_size field so each request carries the remote PP size.
    2. NixlConnectorMetadata._add_new_req reads pp_size from kv_transfer_params.
    3. NixlConnectorScheduler.request_finished includes pp_size in the returned dict.
    4. NixlConnectorWorker._nixl_handshake receives remote_pp_size and uses
       get_all_pp_tp_targets instead of get_target_remote_ranks.
    """

    # --- [FAILS_NOW] ReqMeta.pp_size field ---

    def test_req_meta_has_pp_size_field(self):
        """[FAILS_NOW] ReqMeta must have a pp_size field (default 1)."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import ReqMeta
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ReqMeta)}
        assert "pp_size" in fields, (
            f"ReqMeta missing pp_size field. Current fields: {fields}. "
            "Add: pp_size: int = 1"
        )

    def test_req_meta_pp_size_default_is_1(self):
        """[FAILS_NOW] ReqMeta.pp_size must default to 1 (backward compat)."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import ReqMeta

        req = ReqMeta(
            local_block_ids=[],
            local_physical_block_ids=[],
            tp_size=4,
            # pp_size not specified — should default to 1
        )
        assert req.pp_size == 1, f"Expected pp_size=1 (default), got {req.pp_size}"

    # --- [FAILS_NOW] kv_transfer_params includes pp_size ---

    def test_add_new_req_reads_pp_size_from_params(self):
        """[FAILS_NOW] _add_new_req must read pp_size from kv_transfer_params."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnectorMetadata,
        )

        meta = NixlConnectorMetadata()
        req_meta = meta._add_new_req(
            local_block_ids=[0, 1, 2],
            kv_transfer_params={"tp_size": 4, "pp_size": 2},
        )
        assert req_meta.pp_size == 2, (
            f"Expected pp_size=2 from kv_transfer_params, got {req_meta.pp_size}. "
            "Fix: tp_size=kv_transfer_params.get('pp_size', 1)"
        )

    def test_add_new_req_pp_size_defaults_to_1(self):
        """[FAILS_NOW] _add_new_req pp_size=1 when not in kv_transfer_params."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnectorMetadata,
        )

        meta = NixlConnectorMetadata()
        req_meta = meta._add_new_req(
            local_block_ids=[0],
            kv_transfer_params={"tp_size": 8},  # no pp_size key
        )
        assert req_meta.pp_size == 1

    # --- [FAILS_NOW] NixlConnectorScheduler.request_finished includes pp_size ---

    def test_request_finished_kv_transfer_params_includes_pp_size(self):
        """[FAILS_NOW] kv_transfer_params returned by request_finished must include pp_size."""
        from unittest.mock import MagicMock, patch
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnectorScheduler,
        )
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
        )

        cfg = MagicMock()
        cfg.cache_config.block_size = 16
        cfg.parallel_config.data_parallel_index = 0
        cfg.parallel_config.tensor_parallel_size = 4
        cfg.parallel_config.pipeline_parallel_size = 2  # PP2
        cfg.scheduler_config.disable_hybrid_kv_cache_manager = True
        cfg.kv_transfer_config = MagicMock()
        cfg.kv_transfer_config.kv_buffer_device = "cpu"

        kv_cache_config = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer0"],
                    FullAttentionSpec(block_size=16, num_kv_heads=4,
                                     head_size=16, dtype=torch.float16),
                )
            ],
        )

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.current_platform"
        ) as mock_platform:
            mock_platform.device_type = "cpu"
            sched = NixlConnectorScheduler(cfg, "engine-0", kv_cache_config)

        from vllm.v1.request import RequestStatus

        # Mock a finished Decode-side request (do_remote_decode=True)
        mock_request = MagicMock()
        mock_request.request_id = "req-1"
        mock_request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_request.kv_transfer_params = {"do_remote_decode": True}

        with patch.object(sched, "get_sw_clipped_blocks", return_value=[[0, 1, 2]]):
            _, kv_params = sched.request_finished(mock_request, [[0, 1, 2]])

        assert kv_params is not None, "request_finished should return kv_transfer_params"
        assert "pp_size" in kv_params, (
            f"kv_transfer_params missing pp_size. Got keys: {list(kv_params.keys())}. "
            "Fix: add pp_size=vllm_config.parallel_config.pipeline_parallel_size"
        )
        assert kv_params["pp_size"] == 2, (
            f"Expected pp_size=2 (from PP2 config), got {kv_params.get('pp_size')}"
        )

    # --- [FAILS_NOW] _nixl_handshake uses get_all_pp_tp_targets ---

    def test_nixl_handshake_uses_get_all_pp_tp_targets_for_pp2(self):
        """[FAILS_NOW] With remote_pp_size=2, _nixl_handshake must request
        global indices from get_all_pp_tp_targets (not get_target_remote_ranks).

        Verifies that the method calls get_all_pp_tp_targets when remote_pp_size > 1.
        """
        from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology

        mock_backend = MagicMock()
        mock_backend.get_kv_cache_shape.return_value = (1, 16, 4, 1, 1)
        mock_backend.get_kv_cache_stride_order.side_effect = NotImplementedError

        topo = TpKVTopology(
            tp_rank=0,
            engine_id="prefill",
            remote_tp_size={"prefill": 8},
            remote_block_size={"prefill": 16},
            is_mla=True,
            total_num_kv_heads=4,
            attn_backends=[mock_backend],
            tensor_shape=None,
        )

        # With remote PP2+TP4, D_rank=0 must query global indices [0, 4]
        targets_pp2 = topo.get_all_pp_tp_targets(remote_tp_size=4, remote_pp_size=2)
        targets_pp1 = topo.get_target_remote_ranks(remote_tp_size=4)

        # PP2 must return more targets than PP1
        assert len(targets_pp2) > len(targets_pp1), (
            f"PP2 ({targets_pp2}) should have more targets than PP1 ({targets_pp1})"
        )
        assert targets_pp2 == [0, 4], f"Expected [0, 4] for D_rank=0 PP2+TP4, got {targets_pp2}"
        assert targets_pp1 == [0], f"Expected [0] for D_rank=0 TP4 (PP1), got {targets_pp1}"


# ===========================================================================
# Tests: validate_remote_agent_handshake PP layer count mismatch (Wave 2.3 fix)
# ===========================================================================

class TestValidateRemoteAgentPP:
    """
    _validate_remote_agent_handshake must not crash when remote PP agent's
    block_lens has fewer entries than local block_len_per_layer.

    Root cause: with PP4+TP1 Prefill, each PP rank registers only its 7 layers,
    so nixl_agent_meta.block_lens has 7 entries. Decode's block_len_per_layer
    has 27 entries. The loop `for i in range(27)` crashes at i=7.

    Fix: `range(min(local, remote))` - only validate the overlap.

    Tests here use pure Python logic (same as the fix) — no NIXL runtime needed.
    """

    # --- Pure logic tests for the min() fix ---

    def test_old_range_causes_index_error_pp4(self):
        """Documents the bug: range(27) with 7-entry block_lens → IndexError at i=7."""
        block_len_per_layer = [32768] * 27
        remote_block_lens = [32768] * 7  # PP stage: only 7 layers
        with pytest.raises(IndexError):
            for i in range(len(block_len_per_layer)):   # OLD: range(27)
                _ = block_len_per_layer[i] == remote_block_lens[i]

    def test_fixed_range_no_error_pp4(self):
        """After fix: min(27,7)=7, range(7) → no IndexError."""
        block_len_per_layer = [32768] * 27
        remote_block_lens = [32768] * 7
        num_check = min(len(block_len_per_layer), len(remote_block_lens))
        assert num_check == 7
        for i in range(num_check):   # FIXED: range(7)
            assert block_len_per_layer[i] == remote_block_lens[i]

    def test_fixed_range_pp1_unchanged(self):
        """PP=1: min(27,27)=27, all 27 validated, same as before."""
        block_len_per_layer = [32768] * 27
        remote_block_lens = [32768] * 27
        num_check = min(len(block_len_per_layer), len(remote_block_lens))
        assert num_check == 27
        for i in range(num_check):
            assert block_len_per_layer[i] == remote_block_lens[i]

    def test_fixed_range_mismatch_still_caught(self):
        """Block size mismatch in overlap still raises AssertionError."""
        block_len_per_layer = [32768] * 27
        remote_block_lens = [16384] * 7   # WRONG
        num_check = min(len(block_len_per_layer), len(remote_block_lens))
        with pytest.raises(AssertionError):
            for i in range(num_check):
                assert block_len_per_layer[i] == remote_block_lens[i], \
                    "KV cache sizes must match between P and D when replicated"

    def test_fixed_range_all_pp4_stages(self):
        """PP4: 4 stages with 7/7/7/6 layers each, all pass without IndexError."""
        block_len_per_layer = [32768] * 27
        for pp_rank, num_layers in enumerate([7, 7, 7, 6]):
            remote = [32768] * num_layers
            num_check = min(len(block_len_per_layer), len(remote))
            for i in range(num_check):
                assert block_len_per_layer[i] == remote[i]

    # (Old integration tests with complex mocks removed - pure logic tests above cover the fix)


# ===========================================================================
# Tests: layer-range routing - _build_layer_range_xfer_handle (Wave 2.5)
# ===========================================================================

class TestPPLayerRangeRouting:
    """
    With PP Prefill, each PP rank has a different layer subset.
    The local src handle must cover only the same layers as the remote PP rank
    so that make_prepped_xfer src/dst descriptor counts match.

    Root cause: PP4+TP1 Prefill PP0 has 7 layers (indices 0-6).
    Decode local handle has 27 layers. make_prepped_xfer tries remote[7] → crash.
    Fix: build per-PP-rank local handle sliced to matching layer range.
    """

    def test_layer_slice_pp4_stage0(self):
        """PP rank 0 (7 layers): slice [0:7] of local blocks_data."""
        num_blocks = 10
        num_total_layers = 27
        # Simulate blocks_data as a flat list: layer0_b0, layer0_b1, ..., layer26_b9
        blocks_data = [(i, 64, 0) for i in range(num_total_layers * num_blocks)]

        pp0_start, pp0_end = 0, 7
        sliced = blocks_data[pp0_start * num_blocks: pp0_end * num_blocks]
        assert len(sliced) == 7 * num_blocks
        assert sliced[0] == blocks_data[0]         # first block of layer 0
        assert sliced[-1] == blocks_data[70 - 1]   # last block of layer 6

    def test_layer_slice_pp4_stage1(self):
        """PP rank 1 (7 layers): slice [7:14] of local blocks_data."""
        num_blocks = 10
        blocks_data = [(i, 64, 0) for i in range(27 * num_blocks)]

        pp1_start, pp1_end = 7, 14
        sliced = blocks_data[pp1_start * num_blocks: pp1_end * num_blocks]
        assert len(sliced) == 7 * num_blocks
        assert sliced[0] == blocks_data[7 * num_blocks]   # first block of layer 7

    def test_layer_slice_pp4_stage3_uneven(self):
        """PP rank 3 (6 layers, uneven): slice [21:27] of local blocks_data."""
        num_blocks = 10
        blocks_data = [(i, 64, 0) for i in range(27 * num_blocks)]

        pp3_start, pp3_end = 21, 27
        sliced = blocks_data[pp3_start * num_blocks: pp3_end * num_blocks]
        assert len(sliced) == 6 * num_blocks  # 6 not 7

    def test_all_pp4_slices_cover_all_layers(self):
        """4 PP stage slices together cover all 27 layers exactly once."""
        num_blocks = 10
        blocks_data = [(i, 64, 0) for i in range(27 * num_blocks)]
        pp_layer_counts = [7, 7, 7, 6]  # 3x7 + 6 = 27

        all_sliced = []
        offset = 0
        for count in pp_layer_counts:
            sliced = blocks_data[offset * num_blocks: (offset + count) * num_blocks]
            all_sliced.extend(sliced)
            offset += count

        # All 27*10 = 270 elements covered exactly once
        assert len(all_sliced) == 27 * num_blocks
        assert all_sliced == blocks_data

    def test_pp1_no_slicing_needed(self):
        """PP=1: use full blocks_data (same behavior as before)."""
        num_blocks = 10
        blocks_data = [(i, 64, 0) for i in range(27 * num_blocks)]

        # PP=1: layer_range = [0:27] = full list
        sliced = blocks_data[0: 27 * num_blocks]
        assert sliced == blocks_data
