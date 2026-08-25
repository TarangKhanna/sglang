import os
import socket
import subprocess
import sys
import time
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# A ~1B model keeps the daemon->client IPC handoff cheap to exercise on every
# PR (fast download + load) while still covering the real block-load path; the
# test asserts the IPC path ran, not any particular model's quality.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# This file runs in two suites. The TP=2 class needs the 2-GPU runner (extra-a);
# the TP=1 smoke class is always-on (base-b / 1-gpu-small) so the daemon->client
# IPC handoff is exercised on every PR. Since the CI runner executes the whole
# file per suite, TestWeightCacheDaemonTP2 self-skips when fewer than 2 GPUs are
# visible (i.e. on the 1-gpu runner).
register_cuda_ci(est_time=100, stage="extra-a", runner_config="2-gpu-large")
register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

# Capture the client server's logs so test_loaded_via_ipc can assert the IPC
# load path actually ran (and did not silently fall back to disk).
STDOUT_FILENAME = "/tmp/test_weight_cache_daemon_stdout.log"
STDERR_FILENAME = "/tmp/test_weight_cache_daemon_stderr.log"
SMOKE_STDOUT_FILENAME = "/tmp/test_weight_cache_daemon_smoke_stdout.log"
SMOKE_STDERR_FILENAME = "/tmp/test_weight_cache_daemon_smoke_stderr.log"

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
]


def _cache_socket_paths() -> set[str]:
    from pathlib import Path

    from sglang.srt.weight_cache.registry import default_runtime_dir

    return {str(path) for path in Path(default_runtime_dir()).glob("*.sock")}


def _wait_for_daemon_sockets(
    daemon_process: subprocess.Popen,
    expected_count: int,
    existing_paths: set[str],
    expected_model: str,
) -> list[str]:
    """Wait for the expected identity sockets and prove each one is ready."""
    from sglang.srt.weight_cache.protocol import (
        CacheConfig,
        normalize_model_path_for_cache,
        recv_msg,
        send_msg,
    )
    from sglang.srt.weight_cache.registry import identity_for, socket_path

    deadline = time.monotonic() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    expected_model = normalize_model_path_for_cache(expected_model)
    while time.monotonic() < deadline:
        if daemon_process.poll() is not None:
            raise RuntimeError(
                "Weight cache daemon launcher exited before readiness "
                f"with code {daemon_process.returncode}"
            )

        ready_paths_by_rank = {}
        for path in sorted(_cache_socket_paths() - existing_paths):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(1.0)
                    conn.connect(path)
                    send_msg(conn, {"type": "ping"})
                    if recv_msg(conn) != {"status": "ok"}:
                        continue
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(1.0)
                    conn.connect(path)
                    send_msg(conn, {"type": "query_config"})
                    response = recv_msg(conn)
                if response.get("status") != "ok":
                    continue
                config = CacheConfig.from_dict(response["config"])
                if (
                    config.model_path != expected_model
                    or config.tp_size != expected_count
                    or config.tp_rank not in range(expected_count)
                ):
                    continue
                identity = identity_for(config, response["daemon"]["device_uuid"])
                if socket_path(identity) != path:
                    raise RuntimeError(
                        f"Daemon reported an identity that does not own {path}"
                    )
                ready_paths_by_rank[config.tp_rank] = path
            except (OSError, KeyError, TypeError, ValueError):
                continue

        if set(ready_paths_by_rank) == set(range(expected_count)):
            return [ready_paths_by_rank[rank] for rank in range(expected_count)]
        time.sleep(0.2)

    raise TimeoutError(
        f"Expected {expected_count} new ready weight-cache identity sockets, "
        f"found ranks {sorted(ready_paths_by_rank)} for {expected_model}"
    )


@unittest.skipIf(
    torch.cuda.device_count() < 2,
    "TP=2 weight cache daemon test requires >=2 GPUs (skipped on the 1-gpu runner)",
)
class TestWeightCacheDaemonTP2(CustomTestCase):
    """Start cache daemons, then launch a TP=2 server in client mode."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tp_size = 2

        # Wait for the daemons' identity sockets before starting the client --
        # client mode may take SH and disk-load before a daemon claims EX, so
        # launching it immediately would race.
        existing_paths = _cache_socket_paths()
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(cls.tp_size),
            ]
        )
        cls.daemon_sockets = _wait_for_daemon_sockets(
            cls.daemon_process, cls.tp_size, existing_paths, cls.model
        )

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "daemon_process") and cls.daemon_process:
            kill_process_tree(cls.daemon_process.pid)
        for stream in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def test_generate(self):
        for prompt in PROMPTS:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            text = data["choices"][0]["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Empty output for prompt: {prompt}")

    def test_chat(self):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 2+3?"}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_loaded_via_ipc(self):
        """Assert the server actually loaded weights over IPC.

        Without this, the test would still pass if the IPC path silently
        regressed to disk loading (the daemon would just sit unused), because
        generation output looks identical either way. The daemon-side loader
        logs "[IpcModelLoader] Loaded model via IPC" on every rank, so its
        presence in the captured server logs is our proof the IPC path ran.
        """
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
        logs = ""
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as f:
                    logs += f.read()
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )


class TestWeightCacheDaemonTP1Smoke(CustomTestCase):
    """Always-on TP=1 smoke: start a single weight cache daemon, launch a server
    in client mode, and confirm it loads weights via IPC and generates.

    This is the fast single-GPU sanity check (small model) that runs on every PR
    in the base-b / 1-gpu-small suite; the heavier TP=2 case above only runs on
    the 2-GPU runner.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tp_size = 1

        # Wait for the daemon's identity socket -- don't race the client
        # against its own legitimate SH fallback path.
        existing_paths = _cache_socket_paths()
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(cls.tp_size),
            ]
        )
        cls.daemon_sockets = _wait_for_daemon_sockets(
            cls.daemon_process, cls.tp_size, existing_paths, cls.model
        )

        cls.stdout = open(SMOKE_STDOUT_FILENAME, "w")
        cls.stderr = open(SMOKE_STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "daemon_process") and cls.daemon_process:
            kill_process_tree(cls.daemon_process.pid)
        for stream in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for path in (SMOKE_STDOUT_FILENAME, SMOKE_STDERR_FILENAME):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def test_generate(self):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        text = data["choices"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0, "Empty generation output")

    def test_loaded_via_ipc(self):
        """Assert the server actually loaded weights over IPC (see the TP=2
        variant for why this guard matters)."""
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
        logs = ""
        for path in (SMOKE_STDOUT_FILENAME, SMOKE_STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as f:
                    logs += f.read()
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )


if __name__ == "__main__":
    unittest.main()
