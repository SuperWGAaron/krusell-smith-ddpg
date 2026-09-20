"""Self-contained PyTorch implementation of the thesis's DDPG core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import Actor, Critic
from .noise import OUNoise, UniformNoise
from .replay import ReplayBuffer, TransitionBatch

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


def _nested(config: Any, *names: str, default: Any = _MISSING) -> Any:
    current = config
    for name in names:
        current = _lookup(current, name, _MISSING)
        if current is _MISSING:
            return default
    return current


def _int_dim(value: Any, name: str) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"{name} must be an integer or one-item sequence")
        value = value[0]
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seeded_generator(seed: int) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(int(seed))
    return result


def _resolve_dtype(value: torch.dtype | str | None) -> torch.dtype:
    if value is None:
        return torch.float32
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower().replace("torch.", "")
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    raise ValueError("DDPG dtype must be float32 or float64")


def _resolve_device(value: str | torch.device) -> torch.device:
    if str(value) != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy all parameters and normalization buffers from source to target."""

    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-update parameters and floating buffers, copying integer buffers."""

    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    source_parameters = dict(source.named_parameters())
    for name, target_parameter in target.named_parameters():
        target_parameter.lerp_(source_parameters[name], tau)
    source_buffers = dict(source.named_buffers())
    for name, target_buffer in target.named_buffers():
        source_buffer = source_buffers[name]
        if target_buffer.is_floating_point() or target_buffer.is_complex():
            target_buffer.lerp_(source_buffer, tau)
        else:
            target_buffer.copy_(source_buffer)


class DDPGAgent:
    """DDPG networks, optimizers, replay storage, exploration, and checkpoints.

    ``config`` may be a nested mapping, a dataclass/namespace with ``network``
    and ``ddpg`` attributes, or a flat object using legacy field names.  No
    trainer or environment type is imported, keeping this class independently
    testable.
    """

    checkpoint_version = 1

    def __init__(
        self,
        config: Any = None,
        *,
        actor: Actor | None = None,
        critic: Critic | None = None,
        target_actor: Actor | None = None,
        target_critic: Critic | None = None,
        replay_buffer: ReplayBuffer | None = None,
        exploration_noise: OUNoise | UniformNoise | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        seed: int | None = None,
        target_init_mode: str | None = None,
    ) -> None:
        network = _lookup(config, "network", config)
        # Typed ReproductionConfig has a `ddpg` alias; a raw TOML mapping does
        # not. Both must resolve to the same NetworkConfig fields.
        ddpg = _lookup(config, "ddpg", network)
        economy = _lookup(config, "economy", config)
        runtime = _lookup(config, "runtime", config)
        compatibility = _lookup(config, "compatibility", None)
        requested_device = (
            device
            if device is not None
            else _first(runtime, ("device",), _first(ddpg, ("device",), "cpu"))
        )
        self.device = _resolve_device(requested_device)
        requested_dtype = dtype if dtype is not None else _first(runtime, ("dtype",), torch.float32)
        self.dtype = _resolve_dtype(requested_dtype)
        self.seed = int(
            seed
            if seed is not None
            else _first(
                runtime,
                ("seed", "random_seed"),
                _first(config, ("seed", "random_seed"), 0),
            )
        )
        self.state_dim = _int_dim(
            _first(network, ("state_size", "state_dim", "state_dims"), 4),
            "state_dim",
        )
        self.action_dim = _int_dim(
            _first(network, ("action_size", "action_dim", "action_dims"), 1),
            "action_dim",
        )
        self.action_bound = float(_first(network, ("action_bound",), 1.0))

        self.gamma = float(
            _first(
                ddpg,
                ("discount_factor", "gamma"),
                _first(economy, ("beta",), 0.8),
            )
        )
        self.tau = float(_first(ddpg, ("target_update_rate", "tau"), 0.01))
        self.batch_size = int(_first(ddpg, ("batch_size",), 1024))
        self.clip_exploration_action = bool(_first(ddpg, ("clip_exploration_action",), False))
        self.actor_gradient_clip = _first(ddpg, ("actor_gradient_clip",), None)
        self.critic_gradient_clip = _first(ddpg, ("critic_gradient_clip",), None)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("discount_factor must lie in [0, 1]")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("target_update_rate must lie in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        model_generator = _seeded_generator(self.seed)
        self.actor = (
            Actor.from_config(config, generator=model_generator) if actor is None else actor
        ).to(device=self.device, dtype=self.dtype)
        self.critic = (
            Critic.from_config(config, generator=model_generator) if critic is None else critic
        ).to(device=self.device, dtype=self.dtype)

        # Independent target initialization is intentionally available because
        # some historical variants applied one soft update at construction.
        self.target_actor = (
            Actor.from_config(config, generator=model_generator)
            if target_actor is None
            else target_actor
        ).to(device=self.device, dtype=self.dtype)
        self.target_critic = (
            Critic.from_config(config, generator=model_generator)
            if target_critic is None
            else target_critic
        ).to(device=self.device, dtype=self.dtype)

        active_method = getattr(compatibility, "active", None)
        if callable(active_method):
            legacy_soft_init = bool(active_method("use_soft_target_initialization_bug"))
        else:
            legacy_faithful = bool(_first(compatibility, ("legacy_faithful",), False))
            legacy_soft_init = legacy_faithful and bool(
                _first(
                    compatibility,
                    ("use_soft_target_initialization_bug",),
                    False,
                )
            )
        configured_hard_sync = bool(
            _first(
                ddpg,
                ("hard_sync_targets_on_init",),
                not legacy_soft_init,
            )
        )
        chosen_init_mode = target_init_mode or _first(
            ddpg,
            ("target_init_mode",),
            "hard" if configured_hard_sync else "legacy_soft",
        )
        self.target_init_mode = str(chosen_init_mode).lower().replace("-", "_")
        if self.target_init_mode in {"hard", "hard_sync", "standard"}:
            hard_update(self.target_actor, self.actor)
            hard_update(self.target_critic, self.critic)
            self.target_init_mode = "hard"
        elif self.target_init_mode in {"soft", "legacy", "legacy_soft"}:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
            self.target_init_mode = "legacy_soft"
        else:
            raise ValueError("target_init_mode must be 'hard' or 'legacy_soft'")

        self.target_actor.requires_grad_(False).eval()
        self.target_critic.requires_grad_(False).eval()
        self.actor.train()
        self.critic.train()

        actor_learning_rate = float(_first(ddpg, ("actor_learning_rate", "lr_actor"), 1.0e-5))
        critic_learning_rate = float(_first(ddpg, ("critic_learning_rate", "lr_critic"), 5.0e-5))
        weight_decay = float(_first(ddpg, ("l2_weight_decay",), 0.0))
        if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        # The TensorFlow implementation used a separate Adam instance for the
        # optional supervised warm start.  Keeping its moments separate avoids
        # leaking pretraining optimizer state into the DDPG policy updates.
        self.actor_pretrain_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate, weight_decay=weight_decay
        )

        self.replay = replay_buffer or ReplayBuffer.from_config(
            config,
            seed=self.seed + 1,
            dtype=self.dtype,
            storage_device="cpu",
        )
        if self.replay.state_dim != self.state_dim or self.replay.action_dim != self.action_dim:
            raise ValueError("replay dimensions do not match actor/critic dimensions")

        if exploration_noise is None:
            exploration = _lookup(config, "exploration", ddpg)
            noise_kind = (
                str(
                    _first(
                        exploration,
                        ("kind", "exploration_noise", "noise_type"),
                        _first(ddpg, ("exploration_noise", "noise_type"), "ou"),
                    )
                )
                .lower()
                .replace("-", "_")
            )
            if noise_kind in {"ou", "ornstein_uhlenbeck"}:
                exploration_noise = OUNoise.from_config(
                    config,
                    size=self.action_dim,
                    seed=self.seed + 2,
                    dtype=self.dtype,
                    device=self.device,
                )
            elif noise_kind in {"uniform", "uniform_noise"}:
                exploration_noise = UniformNoise.from_config(
                    config,
                    size=self.action_dim,
                    seed=self.seed + 2,
                    dtype=self.dtype,
                    device=self.device,
                )
            elif noise_kind not in {"none", "off"}:
                raise ValueError(f"unknown exploration noise type: {noise_kind!r}")
        self.exploration_noise = exploration_noise
        self.update_count = 0
        self.pretrain_update_count = 0

    @property
    def memory(self) -> ReplayBuffer:
        """Compatibility alias for the original agent's replay buffer."""

        return self.replay

    @property
    def noise(self) -> OUNoise | UniformNoise | None:
        return self.exploration_noise

    def reset_noise(self) -> None:
        if self.exploration_noise is not None:
            self.exploration_noise.reset()

    def pretrain_actor_step(
        self,
        states: Any,
        target_actions: Any | None = None,
        *,
        target_ratio: float = 0.9,
    ) -> dict[str, float]:
        """Apply one supervised actor warm-start step.

        When explicit targets are omitted, every state is assigned the same
        capital-saving ratio.  The reproduction profile uses the manuscript's
        stated ratio of ``0.9``; callers can override it when auditing the
        retained legacy script, which hard-coded a different value.
        """

        state_tensor = torch.as_tensor(states, dtype=self.dtype, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim != 2 or state_tensor.shape[1] != self.state_dim:
            raise ValueError(f"pretraining states must have shape (batch, {self.state_dim})")
        if len(state_tensor) == 0:
            raise ValueError("pretraining states must not be empty")
        if not torch.isfinite(state_tensor).all():
            raise ValueError("pretraining states must be finite")

        target_shape = (len(state_tensor), self.action_dim)
        if target_actions is None:
            target_tensor = torch.full(
                target_shape,
                float(target_ratio),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            target_tensor = torch.as_tensor(
                target_actions,
                dtype=self.dtype,
                device=self.device,
            )
            if target_tensor.ndim == 1 and self.action_dim == 1:
                target_tensor = target_tensor.unsqueeze(-1)
            try:
                target_tensor = torch.broadcast_to(target_tensor, target_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"pretraining targets must broadcast to shape {target_shape}"
                ) from exc
        if not torch.isfinite(target_tensor).all():
            raise ValueError("pretraining targets must be finite")
        if bool(torch.any(target_tensor < 0.0)) or bool(
            torch.any(target_tensor > self.action_bound)
        ):
            raise ValueError(f"pretraining targets must lie in [0, {self.action_bound}]")

        self.actor.train()
        self.actor_pretrain_optimizer.zero_grad(set_to_none=True)
        predictions = self.actor(state_tensor)
        loss = F.mse_loss(predictions, target_tensor)
        loss.backward()
        if self.actor_gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.actor_gradient_clip))
        self.actor_pretrain_optimizer.step()
        self.pretrain_update_count += 1
        return {
            "actor_pretrain_loss": float(loss.detach().cpu()),
            "prediction_mean": float(predictions.detach().mean().cpu()),
            "target_mean": float(target_tensor.detach().mean().cpu()),
            "batch_size": float(len(state_tensor)),
            "pretrain_update_count": float(self.pretrain_update_count),
        }

    pretrain_actor = pretrain_actor_step

    def select_action(
        self,
        state: Any,
        *,
        explore: bool = True,
        clip: bool | None = None,
        as_numpy: bool = False,
    ) -> Tensor | Any:
        states = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        single = states.ndim == 1
        if single:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(f"state must end in dimension {self.state_dim}")

        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
        self.actor.train(was_training)
        if explore and self.exploration_noise is not None:
            actions = actions + self.exploration_noise.sample().reshape(1, -1)
        should_clip = self.clip_exploration_action if clip is None else bool(clip)
        if should_clip:
            actions = actions.clamp(0.0, self.action_bound)
        result = actions[0] if single else actions
        if as_numpy:
            return result.detach().cpu().numpy()
        return result.detach()

    choose_action = select_action
    act = select_action

    def remember(
        self,
        state: Any,
        action: Any,
        reward: float | Tensor,
        next_state: Any,
        done: bool | float | Tensor,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    store_transition = remember

    def can_update(self, batch_size: int | None = None) -> bool:
        required = self.batch_size if batch_size is None else int(batch_size)
        # Match the thesis code: learning starts only after a full minibatch is
        # present, even though replacement sampling could technically start sooner.
        return len(self.replay) >= required

    def update(
        self,
        batch: TransitionBatch | None = None,
        *,
        batch_size: int | None = None,
        update_actor: bool = True,
        update_targets: bool = True,
    ) -> dict[str, float] | None:
        requested_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if batch is None:
            if not self.can_update(requested_batch_size):
                return None
            batch = self.replay.sample(requested_batch_size)
        batch = batch.to(self.device, dtype=self.dtype)

        self.actor.train()
        self.critic.train()
        with torch.no_grad():
            next_actions = self.target_actor(batch.next_states)
            next_values = self.target_critic(batch.next_states, next_actions)
            target_values = batch.rewards + self.gamma * batch.not_done * next_values

        self.critic_optimizer.zero_grad(set_to_none=True)
        current_values = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(current_values, target_values)
        critic_loss.backward()
        if self.critic_gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.critic_gradient_clip))
        self.critic_optimizer.step()

        actor_loss_value = float("nan")
        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            critic_was_training = self.critic.training
            # The critic supplies dQ/da but is not itself trained by the actor
            # objective. Eval mode also prevents a second BatchNorm running-
            # statistic update from the same minibatch.
            self.critic.eval()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            try:
                actor_loss = -self.critic(batch.states, self.actor(batch.states)).mean()
                actor_loss.backward()
                if self.actor_gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), float(self.actor_gradient_clip)
                    )
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.detach().cpu())
            finally:
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(True)
                self.critic.train(critic_was_training)

        if update_targets:
            self.soft_update_targets()
        self.update_count += 1
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(critic_loss.detach().cpu()),
            "q_mean": float(current_values.detach().mean().cpu()),
            "target_q_mean": float(target_values.detach().mean().cpu()),
            "batch_size": float(len(batch)),
            "update_count": float(self.update_count),
        }

    learn = update

    def hard_sync_targets(self) -> None:
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def soft_update_targets(self, tau: float | None = None) -> None:
        rate = self.tau if tau is None else float(tau)
        soft_update(self.target_actor, self.actor, rate)
        soft_update(self.target_critic, self.critic, rate)

    def update_target_parameters(self, first: bool = False) -> None:
        """Legacy-compatible target update method.

        The original final-thesis implementation used a hard copy for
        ``first=True`` and Polyak updates thereafter.
        """

        if first:
            self.hard_sync_targets()
        else:
            self.soft_update_targets()

    def reset_memory(self) -> None:
        self.replay.clear()

    def state_dict(
        self,
        *,
        include_replay: bool = False,
        include_noise: bool = True,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "checkpoint_version": self.checkpoint_version,
            "seed": self.seed,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "target_init_mode": self.target_init_mode,
            "update_count": self.update_count,
            "pretrain_update_count": self.pretrain_update_count,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_pretrain_optimizer": self.actor_pretrain_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if include_noise and self.exploration_noise is not None:
            state["noise"] = self.exploration_noise.state_dict()
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
        load_optimizers: bool = True,
        load_replay: bool = True,
        load_noise: bool = True,
    ) -> None:
        found_dims = (int(state["state_dim"]), int(state["action_dim"]))
        expected_dims = (self.state_dim, self.action_dim)
        if found_dims != expected_dims:
            raise ValueError(
                f"checkpoint dimensions {found_dims} do not match agent {expected_dims}"
            )
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic.load_state_dict(state["critic"], strict=strict)
        self.target_actor.load_state_dict(state["target_actor"], strict=strict)
        self.target_critic.load_state_dict(state["target_critic"], strict=strict)
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            if "actor_pretrain_optimizer" in state:
                self.actor_pretrain_optimizer.load_state_dict(state["actor_pretrain_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self._optimizer_to_device(self.actor_optimizer)
            self._optimizer_to_device(self.actor_pretrain_optimizer)
            self._optimizer_to_device(self.critic_optimizer)
        if load_replay and "replay" in state:
            self.replay.load_state_dict(state["replay"])
        if load_noise and "noise" in state and self.exploration_noise is not None:
            self.exploration_noise.load_state_dict(state["noise"])
        self.update_count = int(state.get("update_count", 0))
        self.pretrain_update_count = int(state.get("pretrain_update_count", 0))
        self.target_actor.requires_grad_(False).eval()
        self.target_critic.requires_grad_(False).eval()

    def _optimizer_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)

    def save(
        self,
        path: str | Path,
        *,
        include_replay: bool = False,
        include_noise: bool = True,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.state_dict(include_replay=include_replay, include_noise=include_noise),
            destination,
        )
        return destination

    save_checkpoint = save

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        load_optimizers: bool = True,
        load_replay: bool = True,
        load_noise: bool = True,
    ) -> None:
        try:
            state = torch.load(Path(path), map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch before the weights_only keyword.
            state = torch.load(Path(path), map_location=self.device)
        self.load_state_dict(
            state,
            strict=strict,
            load_optimizers=load_optimizers,
            load_replay=load_replay,
            load_noise=load_noise,
        )

    def save_actor(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), destination)
        return destination

    def load_actor(self, path: str | Path, *, strict: bool = True) -> None:
        try:
            state = torch.load(Path(path), map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state, strict=strict)


# Short compatibility alias used by some experiment scripts.
Agent = DDPGAgent


__all__ = [
    "Agent",
    "DDPGAgent",
    "hard_update",
    "soft_update",
]
