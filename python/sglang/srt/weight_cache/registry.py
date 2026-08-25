# SPDX-License-Identifier: Apache-2.0
"""Node-local identity, claim, and readiness paths for weight cache daemons."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import time
import weakref

import msgspec
import psutil

from .protocol import CacheConfig

UNIX_SOCKET_PATH_MAX_BYTES = 107
# Legacy defaults for direct loader construction. ServerArgs supplies the
# public --weight-cache-timeout bound in normal daemon and client launches.
DAEMON_CLAIM_TIMEOUT_S = 20 * 60
CLIENT_READY_TIMEOUT_S = 10 * 60
_RUNTIME_DIR = f"/tmp/sglang-weight-cache-{os.getuid()}"
_ACTIVE_LOCKS: weakref.WeakSet = weakref.WeakSet()


def _close_inherited_identity_locks() -> None:
    """Drop every inherited lock fd in a forked child exactly once."""
    for claim in list(_ACTIVE_LOCKS):
        claim._close_after_fork()


os.register_at_fork(after_in_child=_close_inherited_identity_locks)


def default_runtime_dir() -> str:
    """Return the patchable, owner-private node-local cache directory."""
    return _RUNTIME_DIR


def ensure_runtime_dir() -> str:
    path = default_runtime_dir()
    os.makedirs(path, mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"weight cache runtime path is not a directory: {path}")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(f"weight cache runtime directory must be private: {path}")
    return path


class CacheIdentity(msgspec.Struct, frozen=True):
    device_uuid: str
    config_fingerprint: str

    @property
    def key(self) -> str:
        if not self.device_uuid:
            raise ValueError("physical device UUID must not be empty")
        return hashlib.sha256(
            json.dumps(
                {"device_uuid": self.device_uuid, "config": self.config_fingerprint},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def identity_for(config: CacheConfig, device_uuid: str) -> CacheIdentity:
    return CacheIdentity(str(device_uuid).strip(), config.fingerprint())


def _path(identity: CacheIdentity, suffix: str) -> str:
    path = os.path.join(ensure_runtime_dir(), f"{identity.key}{suffix}")
    if len(os.fsencode(path)) > UNIX_SOCKET_PATH_MAX_BYTES:
        raise ValueError(
            f"weight cache socket path exceeds {UNIX_SOCKET_PATH_MAX_BYTES} bytes: {path}"
        )
    return path


def lock_path(identity: CacheIdentity) -> str:
    return _path(identity, ".lock")


def socket_path(identity: CacheIdentity) -> str:
    return _path(identity, ".sock")


def lock_holder_diagnostics(identity: CacheIdentity) -> str:
    """Return the matching Linux kernel lock entry when it is available."""
    locks_path = "/proc/locks"
    if not os.path.exists(locks_path):
        return "holder metadata unavailable (no /proc/locks)"
    inode = str(os.stat(lock_path(identity)).st_ino)
    try:
        with open(locks_path) as locks:
            for line in locks:
                fields = line.split()
                if len(fields) > 5 and fields[5].endswith(f":{inode}"):
                    return line.strip()
    except OSError as exc:
        return f"holder metadata unavailable ({exc})"
    return "holder metadata unavailable (no matching kernel lock entry)"


class IdentityLock:
    """One advisory lock fd. The lock file is permanent; only the fd is released."""

    def __init__(self, identity: CacheIdentity):
        self.identity = identity
        self.fd: int | None = None
        self.mode: int | None = None

    def __del__(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass

    def acquire(self, mode: int, timeout: float = 0.0) -> bool:
        if self.fd is not None:
            raise RuntimeError("weight cache identity lock is already acquired")
        fd = os.open(lock_path(self.identity), os.O_CREAT | os.O_RDWR, 0o600)
        os.set_inheritable(fd, False)
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    self.fd = fd
                    self.mode = mode
                    # CLOEXEC protects today's Popen/spawn path. The module-level
                    # fork hook closes this fd in a plain-fork child without retaining
                    # one callback per successful acquisition.
                    _ACTIVE_LOCKS.add(self)
                    return True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        return False
                    time.sleep(0.05)
        except BaseException:
            os.close(fd)
            raise

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
            self.mode = None
            _ACTIVE_LOCKS.discard(self)

    def _close_after_fork(self) -> None:
        """Drop the inherited fd without unlocking the parent's claim."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.mode = None
            _ACTIVE_LOCKS.discard(self)


def unlink_socket_for_owner(path: str, claim: IdentityLock) -> None:
    """Remove a stale readiness socket while holding this identity's EX claim."""
    if claim.fd is None or claim.mode != fcntl.LOCK_EX:
        raise RuntimeError("weight cache socket removal requires an exclusive claim")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"refusing to unlink unowned non-socket path: {path}")
    os.unlink(path)


def process_identity_is_alive(pid: int, process_start_time: float) -> bool:
    """Watchdog-only PID reuse check; never used for claim ownership.

    A False return triggers self-termination, so an inconclusive probe
    (AccessDenied under a restricted /proc) must not read as confirmed-dead.
    """
    try:
        process = psutil.Process(pid)
        return abs(float(process.create_time()) - process_start_time) < 1e-3
    except psutil.AccessDenied:
        return True
    except (psutil.Error, ValueError):
        return False
