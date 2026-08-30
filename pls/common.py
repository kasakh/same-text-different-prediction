#!/usr/bin/env python3
"""Common utilities for the production label-stability harness.

Design goals:
- deterministic job IDs from canonical JSON specs
- idempotent local artifact layout
- atomic writes and _SUCCESS markers
- explicit seed profiles for strict vs production-like inference
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash relative paths and bytes for every regular file in a directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with open(item, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def make_job_id(job_type: str, spec: Mapping[str, Any], n: int = 16) -> str:
    return f"{job_type}_{stable_hash(spec, n)}"


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_atomic(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, path)
        return n
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def write_csv_atomic(rows: List[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        write_text_atomic("", path)
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def write_text_atomic(text: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def marker(path: str | Path, name: str = "_SUCCESS") -> Path:
    return Path(path) / name


def mark_success(run_dir: str | Path, details: Optional[Mapping[str, Any]] = None) -> None:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    if details is not None:
        write_json_atomic(dict(details), run_dir / "success.json")
    write_text_atomic(str(time.time()) + "\n", marker(run_dir))


def has_success(run_dir: str | Path) -> bool:
    return marker(run_dir).exists()


def artifact_valid(run_dir: str | Path, expected_files: Optional[List[str]] = None) -> bool:
    run_dir = Path(run_dir)
    if not has_success(run_dir):
        return False
    if expected_files:
        for rel in expected_files:
            if not (run_dir / rel).exists():
                return False
    return True


@contextlib.contextmanager
def file_lock(lock_path: str | Path, stale_seconds: int = 24 * 3600) -> Iterator[bool]:
    """Acquire an atomic lock using O_EXCL. Yields True if acquired.

    This is intentionally simple and works on filesystems that support atomic creation for idempotent workers.
    For very large jobs, DynamoDB conditional writes are still preferable.
    """
    lock_path = Path(lock_path)
    ensure_dir(lock_path.parent)
    acquired = False
    now = time.time()
    if lock_path.exists():
        try:
            age = now - lock_path.stat().st_mtime
            if age > stale_seconds:
                lock_path.unlink()
        except FileNotFoundError:
            pass
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": now}) + "\n")
        acquired = True
        yield True
    except FileExistsError:
        yield False
    finally:
        if acquired:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()


def environment_fingerprint() -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "time": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "TOKENIZERS_PARALLELISM",
                "OMP_NUM_THREADS",
            ]
        },
    }
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            fp["cuda_version"] = torch.version.cuda
            fp["cudnn_version"] = torch.backends.cudnn.version()
            fp["cuda_device_count"] = torch.cuda.device_count()
            fp["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        fp["torch_error"] = repr(e)
    try:
        import transformers

        fp["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import pytorch_lightning as pl

        fp["pytorch_lightning"] = pl.__version__
    except Exception:
        pass
    return fp


def apply_seed_profile(seed: int, profile: str = "production_like") -> None:
    """Apply explicit seed behavior.

    strict: minimize incidental randomness for debugging and sanity checks.
    production_like: fix obvious RNGs while keeping normal fast kernels enabled.
    none: do not set seeds.
    """
    if profile == "none":
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if profile == "strict":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=False)
        elif profile == "production_like":
            # Keep normal production-like kernel choices unless explicitly configured elsewhere.
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(f"Unknown seed profile: {profile}")
    except Exception as e:
        raise RuntimeError(f"Failed to apply seed profile {profile}: {e}") from e


def to_jsonable(x: Any) -> Any:
    """Convert tensors/arrays/scalars to JSON serializable values."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    return x


def normalize_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_copy(src: str | Path, dst: str | Path) -> None:
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)
