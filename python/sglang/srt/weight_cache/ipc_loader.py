# SPDX-License-Identifier: Apache-2.0
"""IPC Model Loader — loads model weights from a Weight Cache Daemon via CUDA IPC.

Zero-copy mode: param.data points directly to IPC-mapped GPU memory. Engine
depends on daemon staying alive.
"""

import fcntl
import logging
import os
import signal
import stat
import threading
import time
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
)

from .protocol import (
    WEIGHT_CACHE_FORMAT_VERSION,
    CacheConfig,
    check_ipc_quant_support,
    compute_env_stamp,
    get_quant_method_name,
    get_resolved_model_revision,
    hash_loader_extra_config,
    hash_quant_config,
    normalize_model_path_for_cache,
    recv_msg,
    send_msg,
)
from .registry import (
    CLIENT_READY_TIMEOUT_S,
    IdentityLock,
    identity_for,
    lock_holder_diagnostics,
    process_identity_is_alive,
)
from .registry import socket_path as identity_socket_path
from .transport import TORCH_IPC_BACKEND, get_client_transport_backend

logger = logging.getLogger(__name__)

# How often the client polls the serving daemon's PID for liveness.
_DAEMON_LIVENESS_POLL_INTERVAL = 5.0

# Bound on _fetch_from_cache's socket-appeared-mid-wait retries. Each retry is
# a real state transition (the socket showed up), not a poll, so this only
# caps a pathological crash-looping daemon, not normal operation.
_FETCH_FROM_CACHE_MAX_ATTEMPTS = 50


class IpcModelLoader(BaseModelLoader):
    """Load model weights from a Weight Cache Daemon via CUDA IPC handles.

    In daemon mode (weight_cache_mode="daemon"), the engine and daemon share
    the same GPU. Falling back to disk loading would cause OOM because both
    processes would hold weights on the same GPU. Therefore, daemon mode
    raises an error if the daemon is unavailable instead of falling back.

    In client mode, a connection or handshake miss re-enters identity claim
    handling. A disk fallback holds a shared claim for this process's
    lifetime (not just during the load), so a daemon can never claim this
    identity while these weights remain resident.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: Optional[str],
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str = "auto",
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format
        self._fallback_claim: Optional[IdentityLock] = None
        self._transport_backend = get_client_transport_backend(TORCH_IPC_BACKEND)

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> nn.Module:
        """Load model weights from the weight cache daemon.

        In daemon mode, raises RuntimeError if the daemon is unavailable
        (fallback to disk loading would cause OOM on shared GPUs).
        In client mode, falls back to DefaultModelLoader.
        """
        tic = time.perf_counter()

        # Hard-gate unsupported quant methods before touching the daemon, so an
        # unsupported model fails explicitly instead of silently disk-loading
        # (client mode) or serving wrong-numerics IPC weights. Checked here so
        # it applies regardless of whether the daemon is reachable.
        quant_method, engine_quant_config = self._resolve_engine_quant(model_config)
        check_ipc_quant_support(quant_method, engine_quant_config, where="client")

        # Try to fetch state from daemon
        cache_data = self._fetch_from_cache(model_config, device_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                if self._fallback_claim is not None:
                    self._fallback_claim.release()
                    self._fallback_claim = None
                raise RuntimeError(
                    "[IpcModelLoader] No matching weight cache daemon is "
                    "registered. In daemon mode, fallback to disk "
                    "loading is disabled because the daemon process already "
                    "holds weights on the same GPU — loading from disk would "
                    "cause OOM. Please ensure the weight cache daemon is "
                    "running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] No weight cache is registered for this GPU and "
                "config; falling back to disk load"
            )
            try:
                model = self._fallback_load(model_config, device_config)
            except BaseException:
                if self._fallback_claim is not None:
                    self._fallback_claim.release()
                    self._fallback_claim = None
                raise
            # Deliberately not released: these weights are resident on this
            # GPU under this identity, so a daemon claiming it later would
            # load a duplicate copy. Released at process exit, like the
            # daemon's own claim.
            return model

        # Check synchronously before importing any handles. A long-lived watcher
        # starts only after model construction has succeeded, so a failed import
        # cannot leave an orphan watcher behind.
        daemon_metadata = cache_data["daemon"]
        daemon_pid = daemon_metadata["pid"]
        daemon_start_time = daemon_metadata["process_start_time"]
        self._require_daemon_alive(
            daemon_pid, daemon_start_time, "before tensor import"
        )

        entries = cache_data["entries"]
        logger.info(
            f"[IpcModelLoader] Fetched {len(entries)} IPC handles from daemon "
            f"in {time.perf_counter() - tic:.2f}s"
        )

        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        model = self._load_zero_copy_mode(
            model_config,
            device_config,
            entries,
            quant_config,
        )

        # Skip _post_load_weights: the daemon already ran
        # process_weights_after_loading on the weights before exporting
        # IPC handles. Running it again would double-process (e.g.,
        # re-quantize already-quantized weights), corrupting tensor data.

        # Rebuild stale tensor views. Some modules store tensor views as
        # plain attributes (not parameters/buffers) during __init__. When
        # the model is initialized on meta device and then weights are
        # replaced via IPC mapping, these views still point to the old
        # meta storage. We must recreate them from the now-valid tensors.
        self._rebuild_stale_views(model)
        self._require_daemon_alive(daemon_pid, daemon_start_time, "after tensor import")
        self._start_daemon_liveness_watchdog(daemon_pid, daemon_start_time)

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    @staticmethod
    def _require_daemon_alive(
        daemon_pid: int, process_start_time: float, phase: str
    ) -> None:
        if not process_identity_is_alive(daemon_pid, process_start_time):
            raise RuntimeError(
                f"[IpcModelLoader] Weight cache daemon pid={daemon_pid} died {phase}"
            )

    def _start_daemon_liveness_watchdog(
        self,
        daemon_pid: int,
        process_start_time: float,
    ) -> None:
        """Terminate if the producer dies while this process holds its tensors."""

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not process_identity_is_alive(daemon_pid, process_start_time):
                    logger.critical(
                        f"[IpcModelLoader] Weight cache daemon (pid={daemon_pid}) "
                        f"died while this engine holds its weights via CUDA IPC. "
                        f"The current transport requires a live producer; "
                        f"terminating rather than serving from mappings whose "
                        f"post-exit lifetime is not supported."
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            f"[IpcModelLoader] Started daemon-liveness watchdog for pid={daemon_pid} "
            f"start_time={process_start_time}"
        )

    def _resolve_engine_quant(self, model_config):
        """Return (quant_method, quant_config) matching the daemon's fingerprint.

        Shared by the IPC allowlist gate and the CacheConfig fingerprint so the
        two can never drift apart. ModelConfig always exposes
        hf_config/quantization directly; quantization_config is the only
        genuinely-optional attribute.
        """
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(model_config.quantization)
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)
        return quant_method, quant_config

    @staticmethod
    def _rebuild_stale_views(model):
        """Rebuild tensor views that went stale after IPC weight replacement.

        RadixLinearAttention.conv_weights is a view of conv1d.weight created
        during __init__. After IPC mapping replaces conv1d.weight with a new
        tensor, the old view still points to meta-device storage. Recreate
        it from the now-valid parameter.
        """
        try:
            from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        except ImportError:
            return

        count = 0
        for _, module in model.named_modules():
            conv1d = getattr(module, "conv1d", None)
            attn = getattr(module, "attn", None)
            if conv1d is not None and isinstance(attn, RadixLinearAttention):
                if hasattr(conv1d, "weight") and conv1d.weight is not None:
                    attn.conv_weights = conv1d.weight.view(
                        conv1d.weight.size(0), conv1d.weight.size(2)
                    )
                    if hasattr(conv1d, "bias") and conv1d.bias is not None:
                        attn.bias = conv1d.bias
                    count += 1

        if count > 0:
            logger.info(f"[IpcModelLoader] Rebuilt {count} stale conv_weights views")

    @staticmethod
    def _set_module_tensor(model, name, tensor, is_param=True):
        """Replace or register a parameter/buffer in the model by its full dotted name.

        This is necessary because setting param.data on a meta-device tensor
        raises a type mismatch error (meta and CUDA tensors have incompatible
        dispatch keys). Instead, we walk the module tree and use setattr to
        replace the entire parameter/buffer object.

        If the attribute already exists as a parameter/buffer, it is replaced.
        If it doesn't exist (e.g. post-quantization params like weight_scale),
        it is registered as a new parameter or buffer.
        """
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf_name = parts[-1]
        if is_param:
            # requires_grad=False: the IPC memory is shared/read-only and SGLang
            # is inference-only, so autograd must never write into it.
            new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(obj, leaf_name, new_param)
        else:
            # register_buffer raises KeyError if the name already exists as a
            # parameter or plain attribute (not a buffer). This happens when
            # process_weights_after_loading converts a parameter to a buffer
            # (e.g. Mamba's A_log). Remove the old attribute first.
            if leaf_name in obj._parameters:
                del obj._parameters[leaf_name]
            elif hasattr(obj, leaf_name) and leaf_name not in obj._buffers:
                delattr(obj, leaf_name)
            obj.register_buffer(leaf_name, tensor)

    def _load_zero_copy_mode(
        self,
        model_config,
        device_config,
        entries,
        quant_config,
    ) -> nn.Module:
        """Zero-copy load: map IPC tensors directly as param.data.

        The model is initialized on the meta device (no memory allocation),
        then each parameter's data is replaced with the IPC-mapped GPU tensor.
        The engine and daemon share the same physical GPU memory via CUDA IPC.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        # Initialize model on meta device to avoid any GPU/CPU memory allocation.
        # This creates the model structure with the correct parameter shapes/dtypes
        # but without allocating actual storage.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

        # Build lookup dicts of existing parameter/buffer names in the
        # meta-device model. Post-quantization parameters (e.g. weight_scale
        # from FP8) are created by process_weights_after_loading, which the
        # daemon already ran. These params exist in the daemon's entries but
        # NOT in the meta-device model — we must register them as new attrs.
        # Use dicts (not sets) so we can do O(1) shape/dtype validation
        # without re-traversing the model tree on every lookup.
        # remove_duplicate=False mirrors the daemon's export (which keys tied
        # weights under every name) so a tied parameter is recognized under all
        # of its names here too.
        existing_params = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        mismatched = []
        new_params_count = 0
        map_tic = time.perf_counter()

        # Iterate over ALL daemon entries (not just model params/buffers).
        # This ensures post-quantization parameters (weight_scale, etc.)
        # that were created by process_weights_after_loading are also mapped.
        for name, entry in entries.items():
            imported_tensor = self._transport_backend.import_tensor(entry)
            is_param = entry.get("is_param", True)

            if name in existing_names:
                # Existing parameter/buffer — validate shape/dtype
                if name in existing_params:
                    ref_param = existing_params[name]
                else:
                    ref_param = existing_buffers[name]
                if (
                    imported_tensor.shape != ref_param.shape
                    or imported_tensor.dtype != ref_param.dtype
                ):
                    mismatched.append(
                        f"  {name}: IPC={imported_tensor.shape}/"
                        f"{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    continue

            # Replace or register the tensor in the model
            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        if mismatched:
            raise RuntimeError(
                f"[IpcModelLoader] {len(mismatched)} tensor(s) have shape/dtype "
                f"mismatch between the IPC daemon and the meta-initialized model. "
                f"The quantization method passed the IPC allowlist gate "
                f"(check_ipc_quant_support), so this is NOT an unsupported-quant "
                f"case — it indicates the daemon's weight fingerprint is "
                f"incomplete or the daemon/client configs drifted (a bug to fix), "
                f"not merely uninitialized weights:\n" + "\n".join(mismatched)
            )

        # After mapping every daemon entry, any tensor still on the meta device
        # is one the daemon did NOT provide. Filling it with torch.empty() would
        # hand the model uninitialized GPU memory — silently producing wrong
        # output, the worst failure mode for a load path. Hard-error and list the
        # offenders instead.
        #
        # The daemon exports the full state_dict AND non-persistent buffers
        # (e.g. rotary embedding cos_sin_cache), so a correct setup leaves nothing
        # on meta here. A non-empty list means the daemon's export is incomplete,
        # or the model has a genuinely-recomputable buffer that must be recomputed
        # explicitly (not filled with garbage) — add that handling here if needed.
        still_on_meta_params = [
            name
            for name, param in model.named_parameters()
            if param.device.type == "meta"
        ]
        still_on_meta_buffers = [
            name for name, buf in model.named_buffers() if buf.device.type == "meta"
        ]

        if still_on_meta_params or still_on_meta_buffers:
            raise RuntimeError(
                f"[IpcModelLoader] After IPC mapping, "
                f"{len(still_on_meta_params)} parameter(s) and "
                f"{len(still_on_meta_buffers)} buffer(s) remain on the meta device "
                f"— the daemon did not export them. Refusing to fill them with "
                f"uninitialized memory, which would silently produce wrong output. "
                f"This means the daemon's export is incomplete, or a recomputable "
                f"buffer needs explicit recompute logic here.\n"
                f"  params: {still_on_meta_params[:10]}"
                f"{'...' if len(still_on_meta_params) > 10 else ''}\n"
                f"  buffers: {still_on_meta_buffers[:10]}"
                f"{'...' if len(still_on_meta_buffers) > 10 else ''}"
            )

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory)
        if imported_refs:
            model._ipc_imported_tensors = imported_refs
        model._weight_cache_transport_backend = self._transport_backend

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors "
            f"({new_params_count} new post-quant), time={map_elapsed:.3f}s"
        )

        return model

    def _build_engine_config(self, model_config, device_id: int) -> CacheConfig:
        from sglang.srt.runtime_context import get_exec, get_parallel

        ps = get_parallel()
        quant_method, quant_config = self._resolve_engine_quant(model_config)
        load_format = getattr(
            self._fallback_load_format, "value", self._fallback_load_format
        )
        return CacheConfig(
            model_path=normalize_model_path_for_cache(
                getattr(model_config, "model_weights", model_config.model_path)
            ),
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            pp_size=ps.pp_size,
            pp_rank=ps.pp_rank,
            dp_size=ps.dp_size,
            ep_size=ps.moe_ep_size,
            moe_dp_size=ps.moe_dp_size,
            moe_dp_rank=ps.moe_dp_rank,
            moe_ep_rank=ps.moe_ep_rank,
            enable_dp_attention=ps.enable_dp_attention,
            enable_dp_lm_head=ps.enable_dp_lm_head,
            attn_cp_size=ps.attn_cp_size,
            moe_dense_tp_size=ps.moe_dense_tp_size,
            moe_a2a_backend=get_exec().moe.moe_a2a_backend,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=model_config.revision or "",
            resolved_revision=get_resolved_model_revision(model_config),
            load_format=str(load_format),
            model_loader_extra_config_hash=hash_loader_extra_config(
                self.load_config.model_loader_extra_config
            ),
            trust_remote_code=self.load_config.weight_cache_trust_remote_code,
            cache_format_version=WEIGHT_CACHE_FORMAT_VERSION,
            **compute_env_stamp(device_id),
        )

    @staticmethod
    def _verify_response_identity(
        result: dict,
        *,
        engine_config: CacheConfig,
        device_uuid: str,
    ) -> None:
        try:
            returned_config = CacheConfig.from_dict(result["config"])
            daemon = result["daemon"]
            returned_device_uuid = str(daemon["device_uuid"])
            fingerprint = str(daemon["config_fingerprint"])
            pid = int(daemon["pid"])
            process_start_time = float(daemon["process_start_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("daemon returned incomplete identity metadata") from exc

        if not returned_config.matches(engine_config):
            raise RuntimeError(
                "daemon returned a CacheConfig different from the requested config"
            )
        if fingerprint != engine_config.fingerprint():
            raise RuntimeError("daemon returned the wrong CacheConfig fingerprint")
        if returned_device_uuid != device_uuid:
            raise RuntimeError(
                "daemon is attached to a different physical GPU: "
                f"expected {device_uuid}, got {returned_device_uuid}"
            )
        if pid <= 0 or process_start_time <= 0:
            raise RuntimeError("daemon returned invalid process identity metadata")

    def _claim_fallback_or_wait(
        self, identity, path: str, *, allow_ready_socket: bool = True
    ) -> bool:
        """Return True when the caller may fall back to disk, False when the
        socket became ready and the caller should retry connecting instead.
        Raises at the timeout instead of ever returning True from it.

        An uncontended shared-lock acquire is the only way this returns
        True: it proves no daemon holds this identity right now (the caller
        holds the lock afterward; see load_model for how long). In daemon
        mode, a co-terminal daemon this engine just spawned may not have
        claimed yet, so it skips that fast path entirely and always waits
        out the full bound for the socket instead.

        Timing out means different things per mode, but both raise instead
        of disk-loading a second copy: in client mode, the deadline is only
        reachable while the lock stays contended, so a live exclusive
        holder -- a real daemon, just slow -- is confirmed, not merely
        possible. In daemon mode, disk fallback was never an option
        regardless of why the socket never appeared (daemon slow, or dead).
        """
        is_daemon_mode = self.weight_cache_mode == "daemon"

        if not is_daemon_mode:
            claim = IdentityLock(identity)
            if claim.acquire(fcntl.LOCK_SH):
                self._fallback_claim = claim
                return True

        timeout_s = self.load_config.weight_cache_timeout or CLIENT_READY_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not is_daemon_mode:
                claim = IdentityLock(identity)
                if claim.acquire(fcntl.LOCK_SH):
                    self._fallback_claim = claim
                    return True
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                pass
            else:
                if (
                    allow_ready_socket
                    and stat.S_ISSOCK(info.st_mode)
                    and info.st_uid == os.getuid()
                ):
                    return False
            time.sleep(0.05)

        raise RuntimeError(
            f"[IpcModelLoader] weight-cache owner for {identity.key} never "
            f"became ready within {timeout_s:.1f}s"
            + (
                "; refusing to disk-load a second copy of the same weights "
                f"onto the same GPU. Holder: {lock_holder_diagnostics(identity)}"
                if not is_daemon_mode
                else " and daemon mode never falls back to disk"
            )
        )

    def _fetch_from_cache(self, model_config, device_config) -> Optional[dict]:
        """Discover/connect, bind the handshake to that daemon, then fetch.

        A socket that appears mid-wait retries the same identity rather than
        falling back to disk. Bounded, not recursive: an unsupervised
        crash-looping daemon would otherwise let each retry recurse forever.
        """
        import socket as socket_mod

        from sglang.srt.platforms import current_platform

        # DeviceConfig is required for every real load path. Do not silently
        # substitute GPU 0: physical-device identity is the cache key. Computed
        # once: a retry must keep chasing the identity it started with, not
        # recompute it against a model_config that could mutate mid-retry.
        device_id = int(device_config.gpu_id)
        engine_config = self._build_engine_config(model_config, device_id)
        device_uuid = current_platform.get_device_uuid(device_id)
        identity = identity_for(engine_config, device_uuid)
        socket_path = self.socket_path or identity_socket_path(identity)

        for _ in range(_FETCH_FROM_CACHE_MAX_ATTEMPTS):
            # Only connect to a real socket node owned by us: reject a symlink,
            # a plain file, or another user's socket planted at this /tmp path.
            # An absent socket means no daemon -> fall back to disk (None).
            try:
                st = os.lstat(socket_path)
            except FileNotFoundError:
                if not self._claim_fallback_or_wait(identity, socket_path):
                    continue
                logger.info(
                    f"[IpcModelLoader] Daemon socket not found at {socket_path}."
                )
                return None
            if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
                raise RuntimeError(
                    f"[IpcModelLoader] Refusing to connect: {socket_path} is not "
                    f"a socket owned by this user."
                )

            sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
            try:
                sock.settimeout(30)
                sock.connect(socket_path)
            except FileNotFoundError:
                sock.close()
                if not self._claim_fallback_or_wait(
                    identity, socket_path, allow_ready_socket=False
                ):
                    continue
                return None
            except ConnectionRefusedError:
                sock.close()
                if not self._claim_fallback_or_wait(
                    identity, socket_path, allow_ready_socket=False
                ):
                    continue
                return None
            except Exception as e:
                sock.close()
                raise RuntimeError(
                    f"[IpcModelLoader] Failed to connect to daemon at "
                    f"{socket_path}: {e}"
                ) from e

            try:
                logger.info(
                    f"[IpcModelLoader] Requesting weights from daemon at "
                    f"{socket_path} with config: "
                    f"model={engine_config.model_path}, "
                    f"arch={engine_config.model_arch}, "
                    f"tp={engine_config.tp_size}/{engine_config.tp_rank}, "
                    f"quant={engine_config.quant_method}, "
                    f"dtype={engine_config.dtype}"
                )

                send_msg(
                    sock, {"type": "fetch_state", "config": engine_config.to_dict()}
                )
                result = recv_msg(sock)

                if result.get("status") != "ok":
                    daemon_config = result.get("daemon_config", {})
                    raise RuntimeError(
                        f"[IpcModelLoader] Daemon config mismatch!\n"
                        f"  Engine config: {engine_config.to_dict()}\n"
                        f"  Daemon config: {daemon_config}"
                    )

                # Validate the returned config and physical/process identity
                # before importing any tensor mappings.
                self._verify_response_identity(
                    result,
                    engine_config=engine_config,
                    device_uuid=device_uuid,
                )

                backend_name = result.get("transport_backend", TORCH_IPC_BACKEND)
                self._transport_backend = get_client_transport_backend(backend_name)
                return self._transport_backend.recv_fetch_state_response(sock, result)

            except (ConnectionError, OSError) as e:
                # Only a transport failure implies the producer may have died
                # and released EX. A declared mismatch or bad response is a
                # real error and should propagate, not get swallowed into a
                # disk load.
                if not self._claim_fallback_or_wait(
                    identity, socket_path, allow_ready_socket=False
                ):
                    continue
                logger.warning(
                    "[IpcModelLoader] Daemon communication failed and its "
                    "identity claim was released; falling back to a private "
                    "disk load: %s",
                    e,
                )
                return None
            finally:
                sock.close()

        raise RuntimeError(
            f"[IpcModelLoader] gave up on weight-cache identity {identity.key} "
            f"after {_FETCH_FROM_CACHE_MAX_ATTEMPTS} socket-appeared-mid-wait "
            f"retries"
        )

    def _fallback_load(self, model_config, device_config) -> nn.Module:
        """Fall back to DefaultModelLoader for disk-based loading."""
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.loader import DefaultModelLoader

        fallback_config = LoadConfig(
            load_format=self._fallback_load_format,
            download_dir=self.load_config.download_dir,
            model_loader_extra_config=self.load_config.model_loader_extra_config,
            tp_rank=self.load_config.tp_rank,
        )
        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(fallback_config)
        return fallback.load_model(
            model_config=model_config, device_config=device_config
        )

    def download_model(self, model_config) -> None:
        """No-op: daemon handles its own model downloading."""
        pass
