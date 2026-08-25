"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig fingerprint matching / (de)serialization
  - quant-config hashing and method-name extraction
  - daemon spawn configuration forwarding
  - the IPC quantization allowlist (the gate that keeps silently-wrong
    quant methods off the zero-copy path)

They intentionally require no CUDA, no model download, and no daemon
process, so they run in the fast CPU suite and would catch a regression
in any of these branches before it reaches the expensive GPU path.

Identity/claim/lock coverage for weight_cache/registry.py lives in
test_weight_cache_registry.py in this same directory.
"""

import os
import socket
import struct
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.weight_cache.protocol import (
    IPC_QUANT_ALLOWLIST,
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    hash_quant_config,
    is_ipc_quant_supported,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
    get_client_transport_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_cache_config(**overrides) -> CacheConfig:
    base = dict(
        model_path="/models/demo",
        model_arch="LlamaForCausalLM",
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="torch.float16",
        revision="",
        resolved_revision="",
        device_capability="8.0",
        torch_version="2.5.1",
        load_format="auto",
        model_loader_extra_config_hash="",
        trust_remote_code=False,
    )
    base.update(overrides)
    return CacheConfig(**base)


class TestProtocolFraming(CustomTestCase):
    """Length-prefixed pickle framing over a real socket pair."""

    def test_round_trip(self):
        a, b = socket.socketpair()
        try:
            payload = {"handles": [1, 2, 3], "meta": ("x", 4.5), "flag": True}
            send_msg(a, payload)
            self.assertEqual(recv_msg(b), payload)
        finally:
            a.close()
            b.close()

    def test_multiple_messages_are_framed_independently(self):
        a, b = socket.socketpair()
        try:
            send_msg(a, {"n": 1})
            send_msg(a, {"n": 2})
            self.assertEqual(recv_msg(b), {"n": 1})
            self.assertEqual(recv_msg(b), {"n": 2})
        finally:
            a.close()
            b.close()

    def test_connection_closed_mid_header_raises(self):
        a, b = socket.socketpair()
        try:
            # Peer sends only a partial header, then hangs up.
            a.sendall(struct.pack("!I", 128)[:2])
            a.close()
            with self.assertRaises(ConnectionError):
                recv_msg(b)
        finally:
            b.close()

    def test_connection_closed_mid_body_raises(self):
        a, b = socket.socketpair()
        try:
            # Full header promising 128 bytes, but no body follows.
            a.sendall(struct.pack("!I", 128))
            a.close()
            with self.assertRaises(ConnectionError):
                recv_msg(b)
        finally:
            b.close()


class TestTransportBackend(CustomTestCase):
    def test_default_backend_is_torch_ipc(self):
        backend = get_client_transport_backend(None)
        self.assertEqual(backend.name, TORCH_IPC_BACKEND)

    def test_unknown_backend_raises(self):
        with self.assertRaises(RuntimeError):
            get_client_transport_backend("does_not_exist")

    def test_torch_ipc_backend_round_trip(self):
        backend = TorchIpcTransportBackend()
        state_tensors = {"x": (torch.arange(8, dtype=torch.float32), True)}
        entries = backend.prepare_export(state_tensors)

        a, b = socket.socketpair()
        try:
            backend.send_fetch_state_response(
                a,
                config={"k": "v"},
                entries=entries,
                daemon={
                    "device_uuid": "GPU-0000",
                    "config_fingerprint": "fp",
                    "pid": 123,
                    "process_start_time": 1.0,
                },
            )
            resp = recv_msg(b)
            resp = backend.recv_fetch_state_response(b, resp)
            imported = backend.import_tensor(resp["entries"]["x"])
            self.assertTrue(torch.equal(imported.cpu(), state_tensors["x"][0]))
            self.assertEqual(resp["transport_backend"], TORCH_IPC_BACKEND)
        finally:
            a.close()
            b.close()


class TestCacheConfig(CustomTestCase):
    def test_identical_configs_match(self):
        self.assertTrue(_make_cache_config().matches(_make_cache_config()))

    def test_any_field_difference_breaks_match(self):
        base = _make_cache_config()
        for field, value in (
            ("tp_rank", 1),
            ("moe_dp_rank", 1),
            ("moe_ep_rank", 1),
            ("enable_dp_attention", True),
            ("moe_dense_tp_size", 1),
            ("moe_a2a_backend", "mooncake"),
            ("dtype", "torch.bfloat16"),
            ("quant_method", "fp8"),
            ("model_path", "/models/other"),
            ("revision", "v2"),
            ("resolved_revision", "abc123"),
            ("load_format", "safetensors"),
            ("trust_remote_code", True),
            ("device_capability", "9.0"),
            ("torch_version", "2.4.0"),
        ):
            self.assertFalse(
                base.matches(_make_cache_config(**{field: value})),
                msg=f"{field} difference should break match",
            )

    def test_dict_round_trip(self):
        cfg = _make_cache_config(quant_method="fp8", quant_config_hash="abc123")
        restored = CacheConfig.from_dict(cfg.to_dict())
        self.assertTrue(cfg.matches(restored))
        self.assertEqual(cfg.to_dict(), restored.to_dict())


class TestQuantConfigHashing(CustomTestCase):
    def test_none_hashes_to_empty(self):
        self.assertEqual(hash_quant_config(None), "")

    def test_dict_hash_is_deterministic_and_order_insensitive(self):
        h1 = hash_quant_config({"bits": 8, "group_size": 128})
        h2 = hash_quant_config({"group_size": 128, "bits": 8})
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, hash_quant_config({"bits": 4, "group_size": 128}))

    def test_hash_is_not_truncated(self):
        # A correctness gate must use the full SHA-256 digest, not a 16-char prefix.
        self.assertEqual(len(hash_quant_config({"bits": 8})), 64)

    def test_hash_does_not_embed_object_address(self):
        # Two distinct instances with identical public attrs must hash equal,
        # otherwise configs would mismatch across processes (the bug the
        # docstring warns about).
        class _Q:
            def __init__(self):
                self.bits = 8
                self.method = "fp8"

        self.assertEqual(hash_quant_config(_Q()), hash_quant_config(_Q()))

    def test_get_quant_method_name_variants(self):
        self.assertEqual(get_quant_method_name(None), "")
        self.assertEqual(get_quant_method_name("fp8"), "fp8")

        class _WithGetName:
            def get_name(self):
                return "gptq_marlin"

        class _WithName:
            name = "awq"

        self.assertEqual(get_quant_method_name(_WithGetName()), "gptq_marlin")
        self.assertEqual(get_quant_method_name(_WithName()), "awq")


class TestGlobalRankAndGpuId(CustomTestCase):
    def test_compute_global_rank_formula(self):
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=0, tp_rank=3), 3)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=1, tp_rank=0), 4)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=2, tp_rank=1), 9)

    def test_compute_local_gpu_id_honors_base_and_step(self):
        # Single-node TP=4: identity mapping rank -> gpu.
        self.assertEqual(
            compute_local_gpu_id(0, 2, pp_size_per_node=1, tp_size_per_node=4),
            2,
        )
        # base_gpu_id offsets every rank; gpu_id_step strides between them.
        self.assertEqual(
            compute_local_gpu_id(
                0, 2, pp_size_per_node=1, tp_size_per_node=4, base_gpu_id=4
            ),
            6,
        )
        self.assertEqual(
            compute_local_gpu_id(
                0, 2, pp_size_per_node=1, tp_size_per_node=4, gpu_id_step=2
            ),
            4,
        )


class TestDaemonLaunchConfiguration(CustomTestCase):
    def test_spawn_forwards_complete_server_args_without_projection(self):
        from sglang.srt.weight_cache import daemon

        # The spawn helper receives Engine's already-resolved ServerArgs. A
        # minimal namespace keeps this projection test CPU-only and
        # model-independent; importantly, no EPLB configuration is involved.
        server_args = SimpleNamespace(
            model_path="/models/demo",
            tp_size=8,
            pp_size=1,
            dp_size=8,
            ep_size=8,
            moe_dp_size=2,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            attn_cp_size=2,
            moe_dense_tp_size=1,
            moe_a2a_backend="mooncake",
            load_format="safetensors",
            dtype="bfloat16",
            quantization="fp8",
            model_loader_extra_config='{"key": "value"}',
            trust_remote_code=True,
            revision="test-revision",
        )

        class FakeProcess:
            pid = 1234

            def start(self):
                pass

        class FakeContext:
            def Process(self, **kwargs):
                self.kwargs = kwargs
                return FakeProcess()

        fake_context = FakeContext()
        from unittest import mock

        with mock.patch.object(
            daemon.multiprocessing, "get_context", return_value=fake_context
        ) as get_context:
            result = daemon.spawn_weight_cache_daemon(
                server_args,
                gpu_id=3,
                tp_rank=3,
                pp_rank=0,
                dist_init_method="tcp://127.0.0.1:29500",
            )

        get_context.assert_called_once_with("spawn")
        self.assertIsInstance(result, FakeProcess)
        self.assertIs(fake_context.kwargs["target"], daemon.run_weight_cache_daemon)
        self.assertEqual(
            fake_context.kwargs["args"],
            (server_args, 3, 3, 0, "tcp://127.0.0.1:29500"),
        )


class TestIpcQuantAllowlist(CustomTestCase):
    def test_unquantized_is_supported(self):
        self.assertTrue(is_ipc_quant_supported("", None))

    def test_block_fp8_supported_but_per_tensor_fp8_rejected(self):
        self.assertTrue(
            is_ipc_quant_supported("fp8", {"weight_block_size": [128, 128]})
        )
        # Per-tensor FP8 (no weight_block_size) transposes the weight during
        # post-processing -> not reproducible by the meta-init client.
        self.assertFalse(is_ipc_quant_supported("fp8", {}))
        self.assertFalse(is_ipc_quant_supported("fp8", None))

    def test_unknown_method_rejected(self):
        self.assertFalse(is_ipc_quant_supported("gptq_marlin", None))
        self.assertFalse(is_ipc_quant_supported("awq", None))

    def test_check_raises_on_unsupported(self):
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("awq", None, where="client")
        # Per-tensor FP8 must also raise even though "fp8" is a known key.
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("fp8", {}, where="daemon")

    def test_check_passes_on_supported(self):
        # Should not raise.
        check_ipc_quant_support("", None, where="daemon")
        check_ipc_quant_support(
            "fp8", {"weight_block_size": [128, 128]}, where="daemon"
        )

    def test_allowlist_registry_shape(self):
        # Guard against accidentally widening the allowlist without review.
        self.assertEqual(set(IPC_QUANT_ALLOWLIST), {"", "fp8"})


class TestDaemonModeRefusesDiskLoad(CustomTestCase):
    """In daemon mode the engine and daemon share a GPU, so a missing daemon
    must be a hard error — NOT a silent disk-load that would OOM the shared
    device. This exercises that contract without a GPU or a live daemon by
    pointing the loader at an explicit socket path that does not exist (an
    explicit socket bypasses identity discovery entirely).
    """

    def _model_config(self):
        # Minimal stand-in: the loader only reads hf_config.quantization_config,
        # quantization, and (unreached here) hf_config.architectures.
        hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLM"], quantization_config=None
        )
        return SimpleNamespace(
            model_path="/models/demo",
            hf_config=hf_config,
            quantization=None,
            revision=None,
            dtype="torch.float16",
        )

    def test_daemon_mode_missing_daemon_raises_instead_of_disk_load(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        missing_socket = "/tmp/sglang-weight-cache-test-missing-daemon.sock"
        if os.path.exists(missing_socket):
            os.unlink(missing_socket)

        loader = IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=missing_socket,
            weight_cache_mode="daemon",
            fallback_load_format="auto",
        )

        with self.assertRaises(RuntimeError) as ctx:
            loader.load_model(model_config=self._model_config(), device_config=None)
        # The error must be about the missing daemon, proving we did not quietly
        # fall through to a disk load.
        self.assertIn("daemon", str(ctx.exception).lower())


class TestClaimFallbackOrWaitDaemonRace(CustomTestCase):
    """A co-terminal daemon may not have claimed its identity yet, so an
    uncontended shared-lock acquisition is not proof none is coming. Daemon
    mode must wait out the full bound rather than fast-exit on that lock."""

    def _loader(self, weight_cache_mode: str):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        return IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=None,  # identity-based discovery, not an explicit override
            weight_cache_mode=weight_cache_mode,
            fallback_load_format="auto",
        )

    def test_daemon_mode_waits_out_the_bound_then_raises(self):
        from unittest import mock

        from sglang.srt.weight_cache import ipc_loader as ipc_loader_module
        from sglang.srt.weight_cache.registry import identity_for

        identity = identity_for(_make_cache_config(), "GPU-race-daemon")
        loader = self._loader("daemon")

        with mock.patch.object(ipc_loader_module, "CLIENT_READY_TIMEOUT_S", 0.2):
            start = time.monotonic()
            # Eventually gives up (no test daemon will ever appear) -- the
            # regression is exiting *instantly* via the uncontended-lock fast
            # path, not the eventual give-up itself. Daemon mode never falls
            # back to disk, so the give-up is a hard error, not a return.
            with self.assertRaises(RuntimeError):
                loader._claim_fallback_or_wait(
                    identity, "/tmp/sglang-weight-cache-race-test-nonexistent.sock"
                )
            elapsed = time.monotonic() - start

        self.assertGreaterEqual(elapsed, 0.15)

    def test_client_mode_still_fast_exits_on_uncontended_lock(self):
        from sglang.srt.weight_cache.registry import identity_for

        identity = identity_for(_make_cache_config(), "GPU-race-client")
        loader = self._loader("client")

        start = time.monotonic()
        result = loader._claim_fallback_or_wait(
            identity, "/tmp/sglang-weight-cache-race-test-nonexistent-2.sock"
        )
        elapsed = time.monotonic() - start

        # Client mode legitimately uses "no one holds this lock" as its
        # signal to proceed with a disk load, protected by the shared lock
        # it just acquired -- this must stay instant, unlike daemon mode.
        self.assertTrue(result)
        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(loader._fallback_claim)

    def test_client_mode_raises_instead_of_disk_loading_a_live_daemons_weights(self):
        """If the lock stays contended until the deadline, a real daemon is
        confirmed alive (just slow) -- disk-loading anyway would create a
        second full copy of the same weights on the same GPU."""
        import fcntl
        from unittest import mock

        from sglang.srt.weight_cache import ipc_loader as ipc_loader_module
        from sglang.srt.weight_cache.registry import IdentityLock, identity_for

        identity = identity_for(_make_cache_config(), "GPU-contended-timeout")
        holder = IdentityLock(identity)
        self.assertTrue(holder.acquire(fcntl.LOCK_EX))
        self.addCleanup(holder.release)

        loader = self._loader("client")
        with mock.patch.object(ipc_loader_module, "CLIENT_READY_TIMEOUT_S", 0.2):
            with self.assertRaises(RuntimeError) as ctx:
                loader._claim_fallback_or_wait(
                    identity,
                    "/tmp/sglang-weight-cache-contended-test-nonexistent.sock",
                )
        self.assertIn(identity.key, str(ctx.exception))
        self.assertIsNone(loader._fallback_claim)

    def test_client_mode_never_disk_loads_while_ex_blocked_past_timeout(self):
        """End-to-end through load_model(), not just _claim_fallback_or_wait:
        an exclusive holder (a live daemon, just slow) that never releases
        and a socket that never appears must reach the bounded timeout as a
        hard error, and _fallback_load must never run -- disk-loading here
        would create a second copy of the same weights on the same GPU the
        real daemon already owns."""
        import fcntl
        from unittest import mock

        from sglang.srt.weight_cache import ipc_loader as ipc_loader_module
        from sglang.srt.weight_cache.registry import IdentityLock, identity_for

        engine_config = _make_cache_config()
        identity = identity_for(engine_config, "GPU-e2e-ex-blocked")
        holder = IdentityLock(identity)
        self.assertTrue(holder.acquire(fcntl.LOCK_EX))
        self.addCleanup(holder.release)

        loader = self._loader("client")
        loader._build_engine_config = mock.Mock(return_value=engine_config)
        loader._fallback_load = mock.Mock(
            side_effect=AssertionError("fallback loader must not run")
        )

        device_config = SimpleNamespace(gpu_id=0)
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"], quantization_config=None
            ),
            quantization=None,
        )

        with mock.patch(
            "sglang.srt.platforms.current_platform.get_device_uuid",
            return_value="GPU-e2e-ex-blocked",
        ), mock.patch.object(
            ipc_loader_module, "CLIENT_READY_TIMEOUT_S", 0.2
        ), mock.patch(
            "sglang.srt.weight_cache.ipc_loader.check_ipc_quant_support"
        ):
            with self.assertRaises(RuntimeError):
                loader.load_model(
                    model_config=model_config, device_config=device_config
                )

        loader._fallback_load.assert_not_called()

    def test_daemon_mode_attaches_once_the_socket_appears_mid_wait(self):
        """The positive path this whole class exists to protect: a
        co-terminal daemon that's merely slow to claim still lets the
        client through as soon as its socket is ready, well before the
        timeout that would otherwise raise."""
        import socket as socket_mod
        import threading
        from unittest import mock

        from sglang.srt.weight_cache import ipc_loader as ipc_loader_module
        from sglang.srt.weight_cache.registry import identity_for

        identity = identity_for(_make_cache_config(), "GPU-race-daemon-attaches")
        loader = self._loader("daemon")
        sock_path = "/tmp/sglang-weight-cache-race-test-appears.sock"
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        self.addCleanup(lambda: os.path.exists(sock_path) and os.unlink(sock_path))

        srv = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        self.addCleanup(srv.close)

        def _bind_after_delay():
            time.sleep(0.1)
            srv.bind(sock_path)

        thread = threading.Thread(target=_bind_after_delay, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 1.0)

        with mock.patch.object(ipc_loader_module, "CLIENT_READY_TIMEOUT_S", 5.0):
            start = time.monotonic()
            result = loader._claim_fallback_or_wait(identity, sock_path)
            elapsed = time.monotonic() - start

        self.assertFalse(result)
        self.assertLess(elapsed, 2.0)


class TestFallbackClaimSurvivesDiskLoad(CustomTestCase):
    """A client that disk-loads a model keeps its identity claim held for
    this process's lifetime, not just for the duration of the load -- a
    daemon claiming the same identity afterward would load a duplicate copy
    of the same weights onto the same GPU while these are still resident."""

    def _loader_with_claim(self):
        import fcntl

        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
        from sglang.srt.weight_cache.registry import IdentityLock, identity_for

        loader = IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=None,
            weight_cache_mode="client",
            fallback_load_format="auto",
        )
        identity = identity_for(_make_cache_config(), "GPU-fallback-lifetime")
        claim = IdentityLock(identity)
        self.assertTrue(claim.acquire(fcntl.LOCK_SH))
        loader._fallback_claim = claim
        self.addCleanup(claim.release)
        return loader

    def test_claim_stays_held_after_a_successful_fallback_load(self):
        from unittest import mock

        loader = self._loader_with_claim()
        claim = loader._fallback_claim
        sentinel_model = object()

        with mock.patch.object(loader, "_fetch_from_cache", return_value=None):
            with mock.patch.object(
                loader, "_fallback_load", return_value=sentinel_model
            ):
                with mock.patch(
                    "sglang.srt.weight_cache.ipc_loader.check_ipc_quant_support"
                ):
                    result = loader.load_model(
                        model_config=SimpleNamespace(
                            hf_config=SimpleNamespace(
                                architectures=["LlamaForCausalLM"],
                                quantization_config=None,
                            ),
                            quantization=None,
                        ),
                        device_config=None,
                    )

        self.assertIs(result, sentinel_model)
        self.assertIs(loader._fallback_claim, claim)
        self.assertIsNotNone(claim.fd)

        # The invariant this test exists for: a daemon for this same
        # identity must not be able to acquire EX while these disk-loaded
        # weights remain resident, or it would load a duplicate copy onto
        # the same GPU.
        import fcntl

        from sglang.srt.weight_cache.registry import IdentityLock

        would_be_daemon = IdentityLock(claim.identity)
        self.assertFalse(would_be_daemon.acquire(fcntl.LOCK_EX))

    def test_claim_is_released_when_the_fallback_load_itself_fails(self):
        from unittest import mock

        loader = self._loader_with_claim()

        with mock.patch.object(loader, "_fetch_from_cache", return_value=None):
            with mock.patch.object(
                loader, "_fallback_load", side_effect=RuntimeError("disk load failed")
            ):
                with mock.patch(
                    "sglang.srt.weight_cache.ipc_loader.check_ipc_quant_support"
                ):
                    with self.assertRaises(RuntimeError):
                        loader.load_model(
                            model_config=SimpleNamespace(
                                hf_config=SimpleNamespace(
                                    architectures=["LlamaForCausalLM"],
                                    quantization_config=None,
                                ),
                                quantization=None,
                            ),
                            device_config=None,
                        )

        # Nothing is resident on the GPU, so nothing to protect -- release.
        self.assertIsNone(loader._fallback_claim)


class TestWaitForReadyWatchingDaemons(CustomTestCase):
    """Composed entirely around scheduler_init_result.wait_for_ready() in
    _launch_subprocesses -- _wait_for_scheduler_ready and
    _launch_scheduler_processes (an override point subclasses like RayEngine
    use) stay at their upstream signatures and never learn about weight-cache
    daemons at all."""

    def test_skips_the_watcher_entirely_with_no_daemons(self):
        """No thread, no polling -- just the plain call, so non-weight-cache
        launches (the overwhelming common case) pay nothing extra."""
        from sglang.srt.entrypoints.engine import _wait_for_ready_watching_daemons

        calls = []
        result = SimpleNamespace(wait_for_ready=lambda: calls.append(1))

        _wait_for_ready_watching_daemons(result, None)
        _wait_for_ready_watching_daemons(result, [])
        self.assertEqual(calls, [1, 1])

    def test_raises_promptly_when_a_watched_daemon_dies(self):
        from unittest import mock

        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import _wait_for_ready_watching_daemons

        # wait_for_ready() would block forever (simulating the scheduler
        # stuck waiting on the daemon's socket) -- the death check must fire
        # without waiting for it.
        result = SimpleNamespace(wait_for_ready=lambda: time.sleep(3600))

        dead_daemon = mock.Mock(pid=222, exitcode=1)
        dead_daemon.is_alive.return_value = False

        with mock.patch.object(
            engine_module, "_DAEMON_LIVENESS_WATCH_INTERVAL_S", 0.05
        ):
            start = time.monotonic()
            with self.assertRaises(RuntimeError) as ctx:
                _wait_for_ready_watching_daemons(result, [dead_daemon])
            elapsed = time.monotonic() - start
        self.assertIn("222", str(ctx.exception))
        self.assertLess(elapsed, 1.0)

    def test_does_not_raise_while_watched_daemons_stay_alive(self):
        from unittest import mock

        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import _wait_for_ready_watching_daemons

        # wait_for_ready() takes a couple of poll cycles to finish -- a live
        # daemon must not trip the death check in the meantime.
        result = SimpleNamespace(wait_for_ready=lambda: time.sleep(0.15))

        healthy_daemon = mock.Mock(pid=333)
        healthy_daemon.is_alive.return_value = True

        with mock.patch.object(
            engine_module, "_DAEMON_LIVENESS_WATCH_INTERVAL_S", 0.05
        ):
            _wait_for_ready_watching_daemons(result, [healthy_daemon])
        # No exception -- reaching here is the assertion.

    def test_propagates_wait_for_ready_own_exception(self):
        """A dead scheduler (wait_for_ready's own failure mode) must still
        surface correctly even though a healthy daemon is also being
        watched."""
        from unittest import mock

        from sglang.srt.entrypoints.engine import _wait_for_ready_watching_daemons

        def _raise():
            raise RuntimeError("scheduler died")

        result = SimpleNamespace(wait_for_ready=_raise)
        healthy_daemon = mock.Mock(pid=444)
        healthy_daemon.is_alive.return_value = True

        with self.assertRaises(RuntimeError) as ctx:
            _wait_for_ready_watching_daemons(result, [healthy_daemon])
        self.assertEqual(str(ctx.exception), "scheduler died")


if __name__ == "__main__":
    unittest.main()
