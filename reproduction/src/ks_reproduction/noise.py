"""Seedable exploration-noise processes used by the PyTorch DDPG agent."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

_MISSING = object()


def _lookup(config: Any, name: str, default: Any = _MISSING) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping) and name in config:
        return config[name]
    if not isinstance(config, Mapping) and hasattr(config, name):
        return getattr(config, name)
    return default


def _section(config: Any, *names: str) -> Any:
    current = config
    for name in names:
        nested = _lookup(current, name, _MISSING)
        if nested is _MISSING:
            return current
        current = nested
    return current


def _first(config: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        value = _lookup(config, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _shape_tuple(size: int | Sequence[int]) -> tuple[int, ...]:
    shape = (int(size),) if isinstance(size, int) else tuple(int(x) for x in size)
    if not shape or any(x <= 0 for x in shape):
        raise ValueError("noise size must contain positive dimensions")
    return shape


def make_generator(
    seed: int | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Generator:
    """Return a CPU generator suitable for portable deterministic sampling."""

    if generator is not None:
        if str(generator.device) != "cpu":
            raise ValueError("exploration-noise generators must be CPU generators")
        return generator
    result = torch.Generator(device="cpu")
    if seed is None:
        result.seed()
    else:
        result.manual_seed(int(seed))
    return result


def _full(
    value: float | Sequence[float] | Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    try:
        return torch.broadcast_to(tensor, shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"{name} cannot be broadcast to noise shape {shape}") from exc


class OUNoise:
    """Ornstein--Uhlenbeck action noise with serializable RNG and process state."""

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        mean: float | Sequence[float] | Tensor = 0.0,
        mu: float | Sequence[float] | Tensor | None = None,
        sigma: float = 0.15,
        theta: float = 0.2,
        dt: float = 1.0e-2,
        initial: float | Sequence[float] | Tensor | None = None,
        x0: float | Sequence[float] | Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.shape = _shape_tuple(size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mean = _full(
            mean if mu is None else mu,
            self.shape,
            dtype=dtype,
            device=self.device,
            name="mean",
        )
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.dt = float(dt)
        if self.sigma < 0.0 or self.theta < 0.0 or self.dt <= 0.0:
            raise ValueError("sigma/theta must be non-negative and dt must be positive")
        initial_value = initial if x0 is None else x0
        self.initial = (
            torch.zeros(self.shape, dtype=dtype, device=self.device)
            if initial_value is None
            else _full(
                initial_value,
                self.shape,
                dtype=dtype,
                device=self.device,
                name="initial",
            )
        )
        self.generator = make_generator(seed, generator=generator)
        self.previous = self.initial.clone()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        size: int | Sequence[int],
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> OUNoise:
        network = _lookup(config, "network", config)
        ddpg = _lookup(config, "ddpg", network)
        exploration = _lookup(config, "exploration", ddpg)
        section = _lookup(ddpg, "ou_noise", exploration)
        return cls(
            size,
            mean=_first(section, ("ou_mean", "mean", "mu"), 0.0),
            sigma=_first(section, ("ou_sigma", "sigma"), 0.05),
            theta=_first(section, ("ou_theta", "theta"), 0.2),
            dt=_first(section, ("ou_dt", "dt"), 1.0e-2),
            initial=_first(section, ("ou_initial", "initial", "x0"), 0.0),
            seed=seed,
            generator=generator,
            dtype=dtype,
            device=device,
        )

    def reset(self, value: float | Sequence[float] | Tensor | None = None) -> Tensor:
        self.previous = (
            self.initial.clone()
            if value is None
            else _full(
                value,
                self.shape,
                dtype=self.dtype,
                device=self.device,
                name="reset value",
            )
        )
        return self.previous.clone()

    def sample(self) -> Tensor:
        normal = torch.randn(
            self.shape,
            dtype=self.dtype,
            device="cpu",
            generator=self.generator,
        ).to(self.device)
        next_value = (
            self.previous
            + self.theta * (self.mean - self.previous) * self.dt
            + self.sigma * math.sqrt(self.dt) * normal
        )
        self.previous = next_value
        return next_value.clone()

    __call__ = sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "previous": self.previous.detach().cpu(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError(
                f"noise shape mismatch: checkpoint {tuple(state['shape'])}, current {self.shape}"
            )
        self.previous = torch.as_tensor(
            state["previous"], dtype=self.dtype, device=self.device
        ).clone()
        self.generator.set_state(torch.as_tensor(state["generator_state"], device="cpu"))


class UniformNoise:
    """Independent uniform action noise with a seedable CPU RNG."""

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.shape = _shape_tuple(size)
        self.low = float(low)
        self.high = float(high)
        if not self.high > self.low:
            raise ValueError("uniform noise requires high > low")
        self.dtype = dtype
        self.device = torch.device(device)
        self.generator = make_generator(seed, generator=generator)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        size: int | Sequence[int],
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> UniformNoise:
        network = _lookup(config, "network", config)
        ddpg = _lookup(config, "ddpg", network)
        exploration = _lookup(config, "exploration", ddpg)
        section = _lookup(ddpg, "uniform_noise", exploration)
        return cls(
            size,
            low=_first(section, ("uniform_low", "low"), 0.0),
            high=_first(section, ("uniform_high", "high"), 1.0),
            seed=seed,
            generator=generator,
            dtype=dtype,
            device=device,
        )

    def reset(self, value: Any = None) -> None:
        del value

    def sample(self) -> Tensor:
        values = torch.rand(
            self.shape,
            dtype=self.dtype,
            device="cpu",
            generator=self.generator,
        )
        return (self.low + (self.high - self.low) * values).to(self.device)

    __call__ = sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError(
                f"noise shape mismatch: checkpoint {tuple(state['shape'])}, current {self.shape}"
            )
        self.generator.set_state(torch.as_tensor(state["generator_state"], device="cpu"))


# Compatibility name retained for readers of the original TensorFlow source.
OUActionNoise = OUNoise
UniformActionNoise = UniformNoise


__all__ = [
    "OUActionNoise",
    "OUNoise",
    "UniformActionNoise",
    "UniformNoise",
    "make_generator",
]
