"""Deterministic runtime and named random-number streams.

Named streams make results independent of the order in which unrelated
components are constructed.  For example, adding a plotting draw cannot alter
the replay-buffer samples because ``"plots"`` and ``"replay"`` receive stable,
separate seeds.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from typing import Any

import numpy as np
import torch

from .config import RuntimeConfig


class RNGStream(Enum):
    NETWORK_INITIALIZATION = "network_initialization"
    TRAIN_INITIAL_STATES = "train_initial_states"
    TRAIN_TRANSITIONS = "train_transitions"
    OU_EXPLORATION = "ou_exploration"
    REPLAY_SAMPLING = "replay_sampling"
    ALM_SIMULATION = "alm_simulation"
    EVALUATION_SHOCKS = "evaluation_shocks"
    EVALUATION_DDPG_SHOCKS = "evaluation_ddpg_shocks"
    EVALUATION_DP_SHOCKS = "evaluation_dp_shocks"
    ROBUSTNESS = "robustness"
    PLOTS = "plots"


def _stream_name(name: str | RNGStream) -> str:
    value = name.value if isinstance(name, RNGStream) else str(name)
    if not value:
        raise ValueError("RNG stream name must not be empty")
    return value


class RNGStreams:
    """Factory/cache for reproducible NumPy and Torch generators."""

    def __init__(self, base_seed: int = 0) -> None:
        if int(base_seed) < 0:
            raise ValueError("base_seed must be non-negative")
        self.base_seed = int(base_seed)
        self._numpy: dict[str, np.random.Generator] = {}
        self._torch: dict[tuple[str, str], torch.Generator] = {}

    def seed_for(self, name: str | RNGStream) -> int:
        """Return a stable 63-bit seed for ``name``."""

        label = _stream_name(name)
        payload = f"ks-reproduction-v1\0{self.base_seed}\0{label}".encode()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)

    def numpy(self, name: str | RNGStream) -> np.random.Generator:
        label = _stream_name(name)
        if label not in self._numpy:
            self._numpy[label] = np.random.default_rng(self.seed_for(label))
        return self._numpy[label]

    def fresh_numpy(self, name: str | RNGStream) -> np.random.Generator:
        """Return a generator reset to the named stream's initial state."""

        return np.random.default_rng(self.seed_for(name))

    def torch(
        self,
        name: str | RNGStream,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        label = _stream_name(name)
        device_name = str(torch.device(device))
        key = (label, device_name)
        if key not in self._torch:
            generator = torch.Generator(device=device_name)
            generator.manual_seed(self.seed_for(label))
            self._torch[key] = generator
        return self._torch[key]

    def fresh_torch(
        self,
        name: str | RNGStream,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        generator = torch.Generator(device=str(torch.device(device)))
        generator.manual_seed(self.seed_for(name))
        return generator

    def seed_global(self, *, deterministic_torch: bool = True) -> None:
        """Seed legacy global APIs and configure deterministic Torch kernels."""

        random.seed(self.base_seed)
        np.random.seed(self.base_seed % (2**32))
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    def state_dict(self) -> dict[str, Any]:
        """Capture already-created streams for an exact checkpoint resume."""

        return {
            "base_seed": self.base_seed,
            "numpy": {
                name: generator.bit_generator.state
                for name, generator in self._numpy.items()
            },
            "torch": {
                f"{name}\0{device}": generator.get_state()
                for (name, device), generator in self._torch.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["base_seed"]) != self.base_seed:
            raise ValueError("RNG checkpoint base seed does not match this runtime")
        for name, generator_state in state.get("numpy", {}).items():
            generator = self.numpy(name)
            generator.bit_generator.state = generator_state
        for key, generator_state in state.get("torch", {}).items():
            name, device = key.split("\0", maxsplit=1)
            generator = self.torch(name, device=device)
            generator.set_state(generator_state)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if preference == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(preference)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported Torch dtype: {name!r}")


_DEFAULT_RUNTIME = RuntimeConfig()


def configure_runtime(
    config: RuntimeConfig = _DEFAULT_RUNTIME,
) -> tuple[RNGStreams, torch.device, torch.dtype]:
    """Configure global determinism and return named streams/device/dtype."""

    config.validate()
    os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
    streams = RNGStreams(config.seed)
    streams.seed_global(deterministic_torch=config.deterministic_torch)
    dtype = resolve_dtype(config.dtype)
    torch.set_default_dtype(dtype)
    return streams, resolve_device(config.device), dtype


@contextmanager
def torch_seed_context(
    streams: RNGStreams,
    name: str | RNGStream,
    *,
    cuda_devices: tuple[int, ...] = (),
) -> Iterator[None]:
    """Temporarily seed global Torch initialization without coupling modules.

    ``torch.nn`` constructors generally use the global generator.  This context
    gives one constructor a named seed, then restores the caller's RNG state.
    """

    cpu_state = torch.random.get_rng_state()
    cuda_states: dict[int, torch.Tensor] = {}
    if torch.cuda.is_available():
        for device in cuda_devices:
            cuda_states[device] = torch.cuda.get_rng_state(device)
    try:
        seed = streams.seed_for(name)
        torch.manual_seed(seed)
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        for device, state in cuda_states.items():
            torch.cuda.set_rng_state(state, device)


def seed_data_loader_worker(
    worker_id: int,
    *,
    streams: RNGStreams,
    stream: str | RNGStream = RNGStream.REPLAY_SAMPLING,
) -> None:
    """Seed Python, NumPy, and Torch globals inside a DataLoader worker."""

    seed = streams.seed_for(f"{_stream_name(stream)}.worker.{int(worker_id)}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


__all__ = [
    "RNGStream",
    "RNGStreams",
    "configure_runtime",
    "resolve_device",
    "resolve_dtype",
    "seed_data_loader_worker",
    "torch_seed_context",
]
