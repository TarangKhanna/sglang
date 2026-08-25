# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via CUDA IPC handles.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Exports every parameter/buffer as a CUDA IPC handle
3. Serves handles over a Unix socket to requesting engine processes
4. Validates CacheConfig compatibility before serving

Usage:
    # Single-node: launch all TP rank daemons with a single command:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 4 \
        --load-format auto --dtype auto --quantization fp8

    # Multi-node: run on each node with --nnodes and --node-rank:
    # Node 0:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 0 \
        --dist-init-method tcp://node0-ip:29500

    # Node 1:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 1 \
        --dist-init-method tcp://node0-ip:29500

    # Or launch a single daemon for a specific rank:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model \
        --gpu-id 0 --tp-size 4 --tp-rank 0 \
        --dist-init-method tcp://127.0.0.1:29500
"""

import argparse
import dataclasses
import fcntl
import logging
import multiprocessing
import os
import signal
import socket
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import psutil
import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, publish

from .protocol import (
    CacheConfig,
    check_ipc_quant_support,
    compute_env_stamp,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_resolved_model_revision,
    hash_loader_extra_config,
    hash_quant_config,
    normalize_model_path_for_cache,
    recv_msg,
    send_msg,
)
from .registry import (
    CacheIdentity,
    IdentityLock,
    identity_for,
    lock_holder_diagnostics,
    socket_path,
    unlink_socket_for_owner,
)
from .transport import choose_daemon_transport_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other
# engine ranks indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0


@dataclasses.dataclass
class WeightCacheDaemonArgs:
    """Daemon-private worker identity and standalone launcher controls.

    ``gpu_id``, ``tp_rank``, ``pp_rank``, and ``dist_init_method`` identify a
    worker after the shared server configuration has been resolved. None has
    a corresponding ServerArgs field with these per-worker semantics.
    """

    gpu_id: Optional[int] = None
    tp_rank: Optional[int] = None
    pp_rank: int = 0
    dist_init_method: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gpu-id",
            type=int,
            default=None,
            help="GPU device ID for a single daemon. If omitted, launches all local ranks.",
        )
        parser.add_argument(
            "--tp-rank",
            type=int,
            default=None,
            help="TP rank for a single daemon. If omitted, launches all local ranks.",
        )
        parser.add_argument("--pp-rank", type=int, default=0)
        parser.add_argument(
            "--dist-init-method",
            default=None,
            help="Daemon distributed init method (for example tcp://host:29500).",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "WeightCacheDaemonArgs":
        return cls(
            gpu_id=args.gpu_id,
            tp_rank=args.tp_rank,
            pp_rank=args.pp_rank,
            dist_init_method=args.dist_init_method,
        )


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single TP rank.

    Holds the complete post-quantization state_dict in GPU memory and
    serves CUDA IPC handles to engine processes via Unix socket.
    """

    def __init__(
        self,
        server_args: "ServerArgs",
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dist_init_method: Optional[str] = None,
    ):
        self.server_args = server_args
        self.model_path = server_args.model_path
        self.gpu_id = gpu_id
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_size = server_args.pp_size
        self.pp_rank = pp_rank
        self.dp_size = server_args.dp_size
        self.ep_size = server_args.ep_size
        self.moe_dp_size = server_args.moe_dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        self.enable_dp_lm_head = server_args.enable_dp_lm_head
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dense_tp_size = server_args.moe_dense_tp_size
        self.moe_a2a_backend = server_args.moe_a2a_backend
        # Not part of CacheConfig: deepep_mode only picks a runtime MoE
        # dispatch path and never affects exported tensor values.
        self.deepep_mode = server_args.deepep_mode
        self.load_format = server_args.load_format
        self.dtype = server_args.dtype
        self.quantization = server_args.quantization
        self.model_loader_extra_config = server_args.model_loader_extra_config
        self.trust_remote_code = server_args.trust_remote_code
        self.revision = server_args.revision
        self.dist_init_method = dist_init_method

        self.identity: Optional[CacheIdentity] = None
        self.claim_lock: Optional[IdentityLock] = None
        self.device_uuid: Optional[str] = None
        self.socket_path: Optional[str] = None

        self.model = None
        self.config: Optional[CacheConfig] = None
        # name -> transport-specific tensor entry metadata.
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        self.transport_backend = None
        self._gpu_work_started = False

    def _init_distributed(self, server_args, model_config):
        """Initialize the distributed backend required for model loading.

        Uses the same world_size/rank formula as the engine:
            world_size = tp_size * pp_size
            rank = tp_size * pp_rank + tp_rank
        """
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )

        if model_parallel_is_initialized():
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Distributed already initialized, skipping"
            )
            return

        if not dist.is_initialized():
            if self.dist_init_method is None:
                # Fallback: auto-assign a port. This only works for single-process.
                import socket as sock_mod

                with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    free_port = s.getsockname()[1]
                self.dist_init_method = f"tcp://127.0.0.1:{free_port}"

            init_distributed_environment(
                world_size=self.tp_size * self.pp_size,
                rank=compute_global_rank(self.tp_size, self.pp_rank, self.tp_rank),
                distributed_init_method=self.dist_init_method,
                local_rank=self.gpu_id,
                backend=current_platform.get_torch_distributed_backend_str(),
                moe_a2a_backend=server_args.moe_a2a_backend,
            )

        initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.ep_size,
            attention_data_parallel_size=(
                self.dp_size if self.enable_dp_attention else 1
            ),
            attention_context_model_parallel_size=self.attn_cp_size,
            moe_data_model_parallel_size=self.moe_dp_size,
        )

        # Initialize DP attention state (required by some models like Qwen3 MoE)
        from sglang.srt.layers.dp_attention import initialize_dp_attention

        initialize_dp_attention(server_args, model_config)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Distributed backend initialized (tp_size={self.tp_size}, "
            f"pp_size={self.pp_size}, "
            f"world_size={self.tp_size * self.pp_size})"
        )

    def load(self):
        """Full loading pipeline: disk → TP shard → quantize → export IPC handles."""
        # CUDA IPC weight sharing relies on torch's _share_cuda_ handle export,
        # which only exists on CUDA-alike platforms (CUDA / ROCm). Fail loud here
        # instead of dying deep inside the export with an opaque error.
        if not current_platform.is_cuda_alike():
            raise RuntimeError(
                f"[WeightCacheDaemon] the weight cache daemon requires a CUDA-alike "
                f"platform (CUDA or ROCm) for CUDA IPC weight sharing, but the "
                f"active platform device type is {current_platform.device_type!r}. "
                f"Disable the weight cache (--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device.
        self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))
        self._gpu_work_started = True

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import get_model_loader
        from sglang.version import __version__ as sglang_version

        server_args = self.server_args
        # ServerArgs resolves source-specific "auto" values (for example
        # mistral, remote, or runai_streamer). The daemon must fingerprint and
        # load with that exact resolved value or every client will mismatch.
        self.load_format = str(
            getattr(server_args.load_format, "value", server_args.load_format)
        )
        publish(server_args, role="weight_cache_daemon")

        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config(server_args)

        # Initialize distributed backend for model loading
        # (must be done after server_args and model_config are available)
        # Build model config first, then init distributed
        model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            dtype=self.dtype,
            quantization=self.quantization,
        )

        # Build cache config fingerprint BEFORE loading the model.
        # Loading may mutate hf_config.quantization_config (e.g. via
        # process_weights_after_loading), which would produce a different
        # hash than what the engine computes from the original config.
        # ModelConfig always exposes hf_config/quantization directly;
        # quantization_config is the only genuinely-optional attribute.
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(
            self.quantization or model_config.quantization
        )
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)

        # Give unsupported methods the actionable allowlist error before
        # attempting to serialize vendor-specific quantization objects.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        # moe_dp_rank/moe_ep_rank are only known once this rank joins its MoE
        # process groups, so the identity claim waits until after this call.
        self._init_distributed(server_args, model_config)
        moe_dp_rank = get_parallel().moe_dp_rank
        moe_ep_rank = get_parallel().moe_ep_rank

        self.config = CacheConfig(
            model_path=normalize_model_path_for_cache(self.model_path),
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            moe_dp_size=self.moe_dp_size,
            moe_dp_rank=moe_dp_rank,
            moe_ep_rank=moe_ep_rank,
            enable_dp_attention=self.enable_dp_attention,
            enable_dp_lm_head=self.enable_dp_lm_head,
            attn_cp_size=self.attn_cp_size,
            moe_dense_tp_size=self.moe_dense_tp_size,
            moe_a2a_backend=self.moe_a2a_backend,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            resolved_revision=get_resolved_model_revision(model_config),
            load_format=self.load_format,
            model_loader_extra_config_hash=hash_loader_extra_config(
                self.model_loader_extra_config
            ),
            trust_remote_code=self.trust_remote_code,
            sglang_version=sglang_version,
            **compute_env_stamp(self.gpu_id),
        )

        # Reserve this exact identity before the expensive disk load. A stale
        # daemon for a different fingerprint has a different lock and cannot
        # block us.
        self.device_uuid = current_platform.get_device_uuid(self.gpu_id)
        identity = identity_for(self.config, self.device_uuid)
        listener_path = socket_path(identity)
        claim_lock = IdentityLock(identity)
        claim_timeout_s = server_args.weight_cache_timeout
        if not claim_lock.acquire(fcntl.LOCK_EX, claim_timeout_s):
            raise RuntimeError(
                "timed out waiting for weight-cache claim "
                f"{identity.key} after {claim_timeout_s:.1f}s: "
                f"{lock_holder_diagnostics(identity)}"
            )
        self.identity = identity
        self.claim_lock = claim_lock
        self.socket_path = listener_path

        # Build load config
        load_config = LoadConfig(
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
            tp_rank=self.tp_rank,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Loading model from disk: {self.model_path}"
        )
        tic = time.perf_counter()

        # DefaultModelLoader includes TP sharding and quant post-processing.
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        self.model = loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
        )

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Model loaded from disk in {elapsed:.2f}s"
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would otherwise
        # risk observing half-written weights.
        current_platform.synchronize()

        # Export all parameters and buffers as IPC handles
        self._export_state()

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} tensors as IPC handles. "
            f"Ready to serve."
        )

    @staticmethod
    def _assert_ipc_compatible_allocator() -> None:
        """Reject allocator configs incompatible with CUDA IPC export.

        The expandable-segments allocator returns memory that cannot be shared
        through torch's _share_cuda_ handle, which would make the export fail
        partway with an opaque error. Detect it up front and fail loud.
        """
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            conf = os.environ.get(var, "")
            for field in conf.split(","):
                key, _, value = field.partition(":")
                if (
                    key.strip() == "expandable_segments"
                    and value.strip().lower() == "true"
                ):
                    raise RuntimeError(
                        f"[WeightCacheDaemon] {var} sets expandable_segments:True, "
                        f"which is incompatible with CUDA IPC weight sharing: the "
                        f"expandable-segments allocator hands out memory that cannot "
                        f"be exported via _share_cuda_, so the IPC handle export "
                        f"would fail mid-way. Unset expandable_segments for the "
                        f"weight cache daemon process (it can stay enabled for the "
                        f"engine itself)."
                    )

    def _export_state(self):
        """Export model parameters and buffers as CUDA IPC handles.

        This includes both persistent buffers (in state_dict) and non-persistent
        buffers (e.g. rotary embedding cos_sin_cache) so the engine can fully
        reconstruct the model state via zero-copy IPC.
        """
        self.state_entries.clear()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with the
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())

        state_tensors = {}
        # Export all items from state_dict (parameters + persistent buffers)
        for name, tensor in self.model.state_dict().items():
            state_tensors[name] = (tensor.data, name in param_names)

        # Also export non-persistent buffers (not in state_dict but needed
        # for inference, e.g. rotary embedding cos_sin_cache)
        non_persistent_count = 0
        for name, buf in self.model.named_buffers():
            if name not in state_dict_names:
                state_tensors[name] = (buf.data, False)
                non_persistent_count += 1

        self.transport_backend = choose_daemon_transport_backend(state_tensors)
        self.state_entries = self.transport_backend.prepare_export(state_tensors)

        # Log approximate serialized metadata size (not payload-backed bytes).
        # Only the handle blob carries real weight, so measure it directly:
        # stringifying every entry would allocate a copy of all handles.
        total_bytes = sum(
            len(handle)
            for handle in (entry.get("handle") for entry in self.state_entries.values())
            if isinstance(handle, (str, bytes, bytearray))
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} tensors "
            f"({non_persistent_count} non-persistent buffers), "
            f"transport={self.transport_backend.name}, "
            f"serialized handle size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def serve(self):
        """Block and serve IPC handles over Unix socket."""
        if self.identity is None or self.config is None or self.socket_path is None:
            raise RuntimeError("weight cache daemon must load and claim before serve")
        # We hold LOCK_EX, so no other daemon can own this identity's path.
        # Only the owner unlinks a stale socket; clients never do.
        unlink_socket_for_owner(self.socket_path, self.claim_lock)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            old_umask = os.umask(0o177)
            try:
                sock.bind(self.socket_path)
            finally:
                os.umask(old_umask)
            sock.listen(8)
            sock.settimeout(1.0)  # Allow periodic shutdown check
            # bind+listen is the readiness signal; there is no record to publish.

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] Listening on {self.socket_path}"
            )

            self._running = True

            def _signal_handler(signum, frame):
                logger.info(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] Received signal "
                    f"{signum}, shutting down"
                )
                self._running = False

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)

            while self._running:
                try:
                    conn, _ = sock.accept()
                    # The listen-socket timeout above only bounds accept(); the
                    # accepted connection is blocking by default. Since we serve
                    # connections serially, a client that connects but never sends
                    # (or dies mid-send) would block recv_msg forever and stall
                    # every other engine rank. Bound each exchange instead.
                    conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
                    try:
                        self._handle_connection(conn)
                    except Exception as e:
                        logger.error(
                            f"[WeightCacheDaemon gpu={self.gpu_id}] "
                            f"Error handling connection: {e}",
                            exc_info=True,
                        )
                    finally:
                        conn.close()
                except socket.timeout:
                    continue
        finally:
            sock.close()
            unlink_socket_for_owner(self.socket_path, self.claim_lock)
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                "Stopped accepting connections"
            )

    def _daemon_metadata(self) -> Dict[str, Any]:
        if self.identity is None:
            raise RuntimeError("weight cache daemon identity is unavailable")
        return {
            "device_uuid": self.identity.device_uuid,
            "config_fingerprint": self.identity.config_fingerprint,
            "pid": os.getpid(),
            "process_start_time": float(psutil.Process().create_time()),
        }

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    "daemon": self._daemon_metadata(),
                },
            )

        elif req.get("type") == "fetch_state":
            # Client requests full state with IPC handles
            engine_config = CacheConfig.from_dict(req["config"])
            if not self.config.matches(engine_config):
                # Log detailed mismatch info for debugging
                daemon_dict = self.config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"Config mismatch: {mismatches}"
                )
                send_msg(
                    conn, {"status": "mismatch", "daemon_config": self.config.to_dict()}
                )
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Serving {len(self.state_entries)} entries via "
                f"{self.transport_backend.name} transport"
            )
            self.transport_backend.send_fetch_state_response(
                conn,
                config=self.config.to_dict(),
                entries=self.state_entries,
                daemon=self._daemon_metadata(),
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok"})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )

    def shutdown(self):
        """Stop serving. EX stays held until process exit -- the caching
        allocator doesn't free VRAM just because Python drops the refs, so
        releasing the lock here would let a replacement claim while this
        process still occupies the GPU."""
        self._running = False
        if dist.is_initialized():
            dist.destroy_process_group()
        self.state_entries.clear()
        self.model = None


def run_weight_cache_daemon(
    server_args: "ServerArgs",
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: Optional[str] = None,
):
    """Entry point for running a weight cache daemon process."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s [Daemon gpu={gpu_id} tp_rank={tp_rank}] "
            "%(levelname)s %(message)s"
        ),
    )

    # Die if our parent (the engine or the standalone launcher that spawned us)
    # dies, even on SIGKILL/OOM-kill. Without this an orphaned daemon keeps a
    # full weight copy pinned in GPU memory and its live kernel claim blocks
    # the next launch.
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    daemon = WeightCacheDaemon(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dist_init_method=dist_init_method,
    )
    try:
        daemon.load()
        daemon.serve()
    finally:
        daemon.shutdown()


def spawn_weight_cache_daemon(
    server_args: "ServerArgs",
    *,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: str,
):
    """Start one daemon from the complete resolved server configuration.

    Uses ``multiprocessing`` with the ``spawn`` start method: the child starts
    in a clean interpreter (no inherited CUDA context) and receives the
    complete resolved ``ServerArgs`` through pickle, avoiding a second
    hand-maintained CLI configuration path.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=run_weight_cache_daemon,
        args=(server_args, gpu_id, tp_rank, pp_rank, dist_init_method),
    )
    proc.start()
    return proc


def launch_weight_cache_daemons(
    server_args: "ServerArgs",
    dist_init_method: Optional[str] = None,
):
    """Launch weight cache daemon processes for this node's PP×TP ranks.

    For single-node (nnodes=1): spawns pp_size * tp_size daemons.
    For multi-node (nnodes>1): spawns this node's share of PP×TP daemons,
    mapping local gpu_id to the correct global (pp_rank, tp_rank).

    Uses ``multiprocessing`` with the ``spawn`` start method. The child starts
    in a clean interpreter and receives the complete resolved ``ServerArgs``
    through pickle, avoiding a second hand-maintained CLI configuration path.

    Usage (single-node):
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 4

    Usage (multi-node, run on each node):
        # Node 0:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 0 \\
            --dist-init-method tcp://node0-ip:29500

        # Node 1:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 1 \\
            --dist-init-method tcp://node0-ip:29500
    """
    import socket as sock_mod

    # Replicate _calculate_rank_ranges logic from engine.py
    pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
    nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
    )
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
        tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
    )

    if server_args.nnodes > 1 and dist_init_method is None:
        raise ValueError(
            "dist_init_method is required for multi-node weight cache daemons. "
            "Use --dist-init-method tcp://<node0-ip>:<port> to specify the "
            "rendezvous address accessible from all nodes."
        )

    # Auto-allocate a free port for the distributed init method
    if dist_init_method is None:
        with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        dist_init_method = f"tcp://127.0.0.1:{free_port}"

    procs = []
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            gpu_id = compute_local_gpu_id(
                pp_rank,
                tp_rank,
                pp_size_per_node,
                tp_size_per_node,
                base_gpu_id=server_args.base_gpu_id,
                gpu_id_step=server_args.gpu_id_step,
            )
            proc = spawn_weight_cache_daemon(
                server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                dist_init_method=dist_init_method,
            )
            procs.append(proc)
            logger.info(
                f"Launched weight cache daemon gpu={gpu_id} "
                f"pp_rank={pp_rank} tp_rank={tp_rank} pid={proc.pid}"
            )

    # Readiness is the identity socket itself; clients perform bounded retries.
    # The launcher only detects children that exit before it starts monitoring.
    num_daemons = len(procs)

    def _terminate_and_reap_daemons() -> None:
        for child in procs:
            if child.is_alive():
                child.terminate()
        for child in procs:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join()

    logger.info(
        "Launched %s weight-cache daemons; clients wait for socket readiness",
        num_daemons,
    )

    # Monitor daemons — poll all of them and, the moment any one exits,
    # terminate the rest and raise. A serial proc.join() would not notice a
    # mid-list death (e.g. procs[1] dying while procs[0] is still alive) until
    # the earlier proc happened to exit, and it never surfaced the failure.
    exited = None
    try:
        while exited is None:
            for proc in procs:
                if not proc.is_alive():
                    exited = proc
                    break
            else:
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down daemons")
    finally:
        _terminate_and_reap_daemons()
        logger.info("All weight cache daemons have been terminated")

    if exited is not None:
        raise RuntimeError(
            f"Weight cache daemon (pid={exited.pid}) exited with code "
            f"{exited.exitcode}; terminated the remaining daemons."
        )


if __name__ == "__main__":
    worker_parser = argparse.ArgumentParser(add_help=False)
    WeightCacheDaemonArgs.add_cli_args(worker_parser)
    worker_ns, server_argv = worker_parser.parse_known_args()

    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(server_argv)
    daemon_args = WeightCacheDaemonArgs.from_cli_args(worker_ns)
    if daemon_args.gpu_id is not None or daemon_args.tp_rank is not None:
        gpu_id = (
            daemon_args.gpu_id
            if daemon_args.gpu_id is not None
            else daemon_args.tp_rank
        )
        tp_rank = (
            daemon_args.tp_rank
            if daemon_args.tp_rank is not None
            else daemon_args.gpu_id
        )
        run_weight_cache_daemon(
            server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=daemon_args.pp_rank,
            dist_init_method=daemon_args.dist_init_method,
        )
    else:
        launch_weight_cache_daemons(
            server_args,
            dist_init_method=daemon_args.dist_init_method,
        )
