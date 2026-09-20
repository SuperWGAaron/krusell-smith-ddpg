"""Typed, seedable replay storage for DDPG transitions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .noise import make_generator

_MISSING = object()


def _lookup(config: Any, name: str, default: Any = _MISSING) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping) and name in config:
        return config[name]
    if not isinstance(config, Mapping) and hasattr(config, name):
        return getattr(config, name)
    return default


def _first(config: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        value = _lookup(config, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _as_dim(value: int | Sequence[int], name: str) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"{name} must be an integer or one-item sequence")
        value = value[0]
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class TransitionBatch:
    """A minibatch whose tensors all use an explicit leading batch dimension."""

    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    @property
    def not_done(self) -> Tensor:
        return 1.0 - self.dones

    @property
    def terminals(self) -> Tensor:
        return self.dones

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> TransitionBatch:
        return TransitionBatch(
            *(tensor.to(device=device, dtype=dtype, non_blocking=non_blocking) for tensor in self)
        )

    def __iter__(self) -> Iterator[Tensor]:
        yield self.states
        yield self.actions
        yield self.rewards
        yield self.next_states
        yield self.dones

    def __len__(self) -> int:
        return int(self.states.shape[0])


class ReplayBuffer:
    """Fixed-capacity ring buffer with deterministic replacement sampling."""

    def __init__(
        self,
        capacity: int,
        state_dim: int | Sequence[int],
        action_dim: int | Sequence[int] = 1,
        *,
        sample_with_replacement: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        storage_device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = _as_dim(state_dim, "state_dim")
        self.action_dim = _as_dim(action_dim, "action_dim")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.sample_with_replacement = bool(sample_with_replacement)
        self.dtype = dtype
        self.storage_device = torch.device(storage_device)
        self.generator = make_generator(seed, generator=generator)

        self.states = torch.empty(
            (self.capacity, self.state_dim), dtype=dtype, device=self.storage_device
        )
        self.actions = torch.empty(
            (self.capacity, self.action_dim), dtype=dtype, device=self.storage_device
        )
        self.rewards = torch.empty((self.capacity, 1), dtype=dtype, device=self.storage_device)
        self.next_states = torch.empty_like(self.states)
        self.dones = torch.empty((self.capacity, 1), dtype=dtype, device=self.storage_device)
        self._cursor = 0
        self._size = 0
        self.total_added = 0

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        storage_device: str | torch.device = "cpu",
    ) -> ReplayBuffer:
        network = _lookup(config, "network", config)
        # ReproductionConfig exposes ``ddpg`` as an alias to NetworkConfig;
        # raw TOML-like mappings do not, so fall back to [network].
        ddpg = _lookup(config, "ddpg", network)
        return cls(
            capacity=_first(ddpg, ("replay_capacity", "max_size"), 1_000_000),
            state_dim=_first(network, ("state_size", "state_dim", "state_dims"), 4),
            action_dim=_first(network, ("action_size", "action_dim", "action_dims"), 1),
            sample_with_replacement=_first(ddpg, ("replay_sample_with_replacement",), True),
            seed=seed,
            generator=generator,
            dtype=dtype,
            storage_device=storage_device,
        )

    def __len__(self) -> int:
        return self._size

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def mem_cntr(self) -> int:
        """Compatibility counter matching the legacy replay-buffer name."""

        return self.total_added

    def clear(self) -> None:
        self._cursor = 0
        self._size = 0
        self.total_added = 0

    def _vector(
        self,
        value: Any,
        width: int,
        *,
        name: str,
    ) -> Tensor:
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.storage_device)
        if tensor.numel() != width:
            raise ValueError(f"{name} must contain {width} value(s)")
        return tensor.reshape(width)

    def add(
        self,
        state: Any,
        action: Any,
        reward: float | Tensor,
        next_state: Any,
        done: bool | float | Tensor,
    ) -> None:
        index = self._cursor
        self.states[index].copy_(self._vector(state, self.state_dim, name="state"))
        self.actions[index].copy_(self._vector(action, self.action_dim, name="action"))
        self.rewards[index, 0] = torch.as_tensor(
            reward, dtype=self.dtype, device=self.storage_device
        ).reshape(())
        self.next_states[index].copy_(self._vector(next_state, self.state_dim, name="next_state"))
        done_value = torch.as_tensor(done, dtype=self.dtype, device=self.storage_device).reshape(())
        self.dones[index, 0] = done_value.clamp(0.0, 1.0)

        self._cursor = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self.total_added += 1

    store_transition = add

    def extend(
        self,
        states: Any,
        actions: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
    ) -> None:
        states_t = torch.as_tensor(states, dtype=self.dtype)
        actions_t = torch.as_tensor(actions, dtype=self.dtype)
        rewards_t = torch.as_tensor(rewards, dtype=self.dtype).reshape(-1)
        next_states_t = torch.as_tensor(next_states, dtype=self.dtype)
        dones_t = torch.as_tensor(dones, dtype=self.dtype).reshape(-1)
        if states_t.ndim != 2 or states_t.shape[1] != self.state_dim:
            raise ValueError(f"states must have shape (batch, {self.state_dim})")
        batch_size = states_t.shape[0]
        if actions_t.shape != (batch_size, self.action_dim):
            raise ValueError(f"actions must have shape (batch, {self.action_dim})")
        if next_states_t.shape != states_t.shape:
            raise ValueError("next_states must have the same shape as states")
        if rewards_t.numel() != batch_size or dones_t.numel() != batch_size:
            raise ValueError("rewards and dones must contain one value per state")
        for index in range(batch_size):
            self.add(
                states_t[index],
                actions_t[index],
                rewards_t[index],
                next_states_t[index],
                dones_t[index],
            )

    def sample(
        self,
        batch_size: int,
        *,
        replacement: bool | None = None,
        device: str | torch.device | None = None,
    ) -> TransitionBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise RuntimeError("cannot sample an empty replay buffer")
        use_replacement = self.sample_with_replacement if replacement is None else bool(replacement)
        if use_replacement:
            indices = torch.randint(
                self._size,
                (batch_size,),
                generator=self.generator,
                device="cpu",
            )
        else:
            if batch_size > self._size:
                raise ValueError("batch_size cannot exceed stored transitions without replacement")
            indices = torch.randperm(self._size, generator=self.generator, device="cpu")[
                :batch_size
            ]
        storage_indices = indices.to(self.storage_device)
        batch = TransitionBatch(
            self.states[storage_indices].clone(),
            self.actions[storage_indices].clone(),
            self.rewards[storage_indices].clone(),
            self.next_states[storage_indices].clone(),
            self.dones[storage_indices].clone(),
        )
        return batch if device is None else batch.to(device)

    sample_buffer = sample

    def state_dict(self) -> dict[str, Any]:
        valid = slice(0, self._size)
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sample_with_replacement": self.sample_with_replacement,
            "cursor": self._cursor,
            "size": self._size,
            "total_added": self.total_added,
            "states": self.states[valid].detach().cpu(),
            "actions": self.actions[valid].detach().cpu(),
            "rewards": self.rewards[valid].detach().cpu(),
            "next_states": self.next_states[valid].detach().cpu(),
            "dones": self.dones[valid].detach().cpu(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.capacity, self.state_dim, self.action_dim)
        found = (
            int(state["capacity"]),
            int(state["state_dim"]),
            int(state["action_dim"]),
        )
        if found != expected:
            raise ValueError(f"replay shape mismatch: checkpoint {found}, current {expected}")
        size = int(state["size"])
        cursor = int(state["cursor"])
        if not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError("invalid replay size or cursor in checkpoint")
        for destination, key in (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.next_states, "next_states"),
            (self.dones, "dones"),
        ):
            source = torch.as_tensor(state[key], dtype=self.dtype, device=self.storage_device)
            if source.shape != destination[:size].shape:
                raise ValueError(f"invalid {key} shape in replay checkpoint")
            destination[:size].copy_(source)
        self._size = size
        self._cursor = cursor
        self.total_added = int(state.get("total_added", size))
        self.sample_with_replacement = bool(
            state.get("sample_with_replacement", self.sample_with_replacement)
        )
        self.generator.set_state(torch.as_tensor(state["generator_state"], device="cpu"))


# Explicit compatibility name for the legacy class.
ReplayBufferDDPG = ReplayBuffer


__all__ = ["ReplayBuffer", "ReplayBufferDDPG", "TransitionBatch"]
