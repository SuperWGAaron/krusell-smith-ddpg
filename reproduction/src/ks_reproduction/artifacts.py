"""Run-directory creation and durable artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON through an adjacent temporary file and atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    """Atomically save a PyTorch object without importing torch at module import."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def verification(self) -> Path:
        return self.root / "verification"

    def create(self, *, allow_existing: bool = False) -> RunLayout:
        if self.root.exists() and any(self.root.iterdir()) and not allow_existing:
            raise FileExistsError(
                f"output directory is not empty: {self.root}; "
                "pass --resume or choose another directory"
            )
        for path in (
            self.root,
            self.checkpoints,
            self.data,
            self.metrics,
            self.figures,
            self.logs,
            self.verification,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def checkpoint(self, round_number: int) -> Path:
        return self.checkpoints / f"round_{round_number:04d}.pt"


def provenance(device: str, deterministic: bool) -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "deterministic_algorithms": deterministic,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def write_manifest(root: Path, *, exclude: Iterable[Path] = ()) -> Path:
    excluded = {path.resolve() for path in exclude}
    manifest = root / "manifest.sha256"
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest or path.resolve() in excluded:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
