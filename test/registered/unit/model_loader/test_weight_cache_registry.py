"""CPU-only tests for identity-keyed weight-cache discovery (registry.py).

Protocol/transport/quant-allowlist coverage lives in
test_weight_cache_protocol.py in this same directory.
"""

import fcntl
import os
import signal
import tempfile
import unittest
from unittest import mock

from sglang.srt.weight_cache import registry
from sglang.srt.weight_cache.protocol import CACHE_IDENTITY_FIELDS, CacheConfig
from sglang.srt.weight_cache.registry import (
    IdentityLock,
    identity_for,
    lock_holder_diagnostics,
    lock_path,
    socket_path,
    unlink_socket_for_owner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(**overrides) -> CacheConfig:
    values = {
        "model_path": "/models/demo",
        "model_arch": "LlamaForCausalLM",
        "tp_size": 2,
        "tp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "dp_size": 1,
        "ep_size": 1,
        "moe_dp_size": 1,
        "moe_dp_rank": 0,
        "moe_ep_rank": 0,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "attn_cp_size": 1,
        "moe_dense_tp_size": None,
        "moe_a2a_backend": "none",
        "quant_method": "",
        "quant_config_hash": "",
        "dtype": "torch.float16",
        "revision": "",
        "resolved_revision": "",
        "device_capability": "8.0",
        "torch_version": "2.5.1",
        "load_format": "auto",
        "model_loader_extra_config_hash": "",
        "trust_remote_code": False,
    }
    values.update(overrides)
    return CacheConfig(**values)


class TestCacheIdentity(CustomTestCase):
    def test_allowlist_covers_every_cacheconfig_field(self):
        # Every CacheConfig field must be a deliberate identity decision, not
        # silently included or omitted as fields are added later.
        self.assertEqual(set(CACHE_IDENTITY_FIELDS), set(CacheConfig.__struct_fields__))

    def test_allowlist_excludes_dp_rank(self):
        # DP replicas on one GPU must share a daemon; see registry.py's
        # CACHE_IDENTITY_FIELDS comment.
        self.assertNotIn("dp_rank", CACHE_IDENTITY_FIELDS)

    def test_allowlist_includes_tp_and_pp_topology(self):
        for field in ("tp_size", "tp_rank", "pp_size", "pp_rank"):
            self.assertIn(field, CACHE_IDENTITY_FIELDS)

    def test_paths_follow_uuid_and_fingerprint(self):
        first = identity_for(_config(), "GPU-a")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            self.assertEqual(lock_path(first).rsplit("/", 1)[-1], f"{first.key}.lock")
            self.assertEqual(socket_path(first).rsplit("/", 1)[-1], f"{first.key}.sock")
        self.assertNotEqual(first.key, identity_for(_config(), "GPU-b").key)
        self.assertNotEqual(first.key, identity_for(_config(tp_rank=1), "GPU-a").key)

    def test_parallel_topology_rank_fields_change_the_fingerprint(self):
        # Each of these ranks selects a distinct cached shard on a shared
        # GPU; conflating any of them would serve the wrong shard's weights.
        base = identity_for(_config(), "GPU-topology").key
        for overrides in (
            {"pp_rank": 1},
            {"moe_dp_rank": 1},
            {"moe_ep_rank": 1},
        ):
            self.assertNotEqual(
                base,
                identity_for(_config(**overrides), "GPU-topology").key,
                msg=f"{overrides} must change the identity key",
            )


class TestIdentityLock(CustomTestCase):
    def test_exclusive_claim_blocks_a_second_claimant(self):
        identity = identity_for(_config(), "GPU-lock")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            owner = IdentityLock(identity)
            contender = IdentityLock(identity)
            self.assertTrue(owner.acquire(fcntl.LOCK_EX))
            self.assertFalse(contender.acquire(fcntl.LOCK_EX))
            owner.release()
            self.assertTrue(contender.acquire(fcntl.LOCK_EX))
            contender.release()

    def test_multiple_shared_claimants_coexist(self):
        # Concurrent disk-fallback clients (or a fallback client racing a
        # not-yet-claimed daemon) must not exclude each other -- only an
        # exclusive claimant is exclusive.
        identity = identity_for(_config(), "GPU-multi-sh")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            first = IdentityLock(identity)
            second = IdentityLock(identity)
            third = IdentityLock(identity)
            self.assertTrue(first.acquire(fcntl.LOCK_SH))
            self.assertTrue(second.acquire(fcntl.LOCK_SH))
            self.assertTrue(third.acquire(fcntl.LOCK_SH))
            first.release()
            second.release()
            third.release()

    def test_different_gpu_uuid_same_config_do_not_contend(self):
        # Two physically distinct GPUs running the identical CacheConfig
        # (e.g. two replicas of the same shard) must each get their own
        # daemon -- neither's exclusive claim may block the other's.
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            gpu_a = IdentityLock(identity_for(_config(), "GPU-distinct-a"))
            gpu_b = IdentityLock(identity_for(_config(), "GPU-distinct-b"))
            self.assertTrue(gpu_a.acquire(fcntl.LOCK_EX))
            self.assertTrue(gpu_b.acquire(fcntl.LOCK_EX))
            gpu_a.release()
            gpu_b.release()

    def test_same_gpu_uuid_different_config_do_not_contend(self):
        # A differently-configured daemon on the same physical GPU (e.g. a
        # config change across a restart racing the old daemon's teardown)
        # is a distinct identity with its own lock -- see protocol.py's
        # CacheConfig docstring.
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            cfg_a = IdentityLock(identity_for(_config(), "GPU-shared"))
            cfg_b = IdentityLock(identity_for(_config(tp_rank=1), "GPU-shared"))
            self.assertTrue(cfg_a.acquire(fcntl.LOCK_EX))
            self.assertTrue(cfg_b.acquire(fcntl.LOCK_EX))
            cfg_a.release()
            cfg_b.release()

    def test_shared_claim_blocks_an_exclusive_claimant(self):
        """A daemon's EX claim attempt must fail while a client's disk-
        fallback SH claim is held for that same identity -- this is the
        flock property _claim_fallback_or_wait's fallback-lifetime fix and
        the daemon's own claim acquisition both depend on."""
        identity = identity_for(_config(), "GPU-sh-blocks-ex")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            fallback_client = IdentityLock(identity)
            daemon = IdentityLock(identity)
            self.assertTrue(fallback_client.acquire(fcntl.LOCK_SH))
            self.assertFalse(daemon.acquire(fcntl.LOCK_EX))
            fallback_client.release()
            self.assertTrue(daemon.acquire(fcntl.LOCK_EX))
            daemon.release()

    def test_socket_removal_requires_exclusive_claim(self):
        identity = identity_for(_config(), "GPU-socket")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            claim = IdentityLock(identity)
            with self.assertRaisesRegex(RuntimeError, "exclusive claim"):
                unlink_socket_for_owner(socket_path(identity), claim)
            self.assertTrue(claim.acquire(fcntl.LOCK_EX))
            unlink_socket_for_owner(socket_path(identity), claim)
            claim.release()

    def test_holder_diagnostics_never_raises(self):
        identity = identity_for(_config(), "GPU-diagnostics")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            claim = IdentityLock(identity)
            self.assertTrue(claim.acquire(fcntl.LOCK_EX))
            self.assertTrue(lock_holder_diagnostics(identity))
            claim.release()

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_killing_owner_releases_claim_held_before_fork(self):
        identity = identity_for(_config(), "GPU-kill")
        with tempfile.TemporaryDirectory(
            dir="/tmp", prefix="wc-"
        ) as runtime_dir, mock.patch.object(registry, "_RUNTIME_DIR", runtime_dir):
            read_fd, write_fd = os.pipe()
            owner = os.fork()
            if owner == 0:
                os.close(read_fd)
                claim = IdentityLock(identity)
                if not claim.acquire(fcntl.LOCK_EX):
                    os._exit(1)
                child = os.fork()
                if child == 0:
                    signal.pause()
                    os._exit(0)
                os.write(write_fd, str(child).encode())
                signal.pause()
                os._exit(0)

            os.close(write_fd)
            child = int(os.read(read_fd, 32))
            os.close(read_fd)
            try:
                os.kill(owner, signal.SIGKILL)
                os.waitpid(owner, 0)
                contender = IdentityLock(identity)
                self.assertTrue(contender.acquire(fcntl.LOCK_EX))
                contender.release()
            finally:
                os.kill(child, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
