"""PyTorch actor and critic networks for the Krusell--Smith DDPG solver.

The original TensorFlow 1 graph called ``batch_normalization`` without a
training flag.  ``legacy_frozen`` reproduces that consequential detail: the
running moments remain zero and one, while the affine scale and bias are still
trainable.  ``batch_norm`` provides ordinary PyTorch batch normalization for
new experiments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_MISSING = object()


def _lookup(config: Any, name: str, default: Any = _MISSING) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
    elif hasattr(config, name):
        return getattr(config, name)
    return default


def _section(config: Any, name: str) -> Any:
    """Return a nested section, or the object itself for flat configurations."""

    return _lookup(config, name, config)


def _first(config: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        value = _lookup(config, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _as_dim(value: Any, *, name: str) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"{name} must be an integer or a one-item sequence")
        value = value[0]
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _as_hidden_dims(value: Any) -> tuple[int, int]:
    dims = tuple(int(item) for item in value)
    if len(dims) != 2 or any(item <= 0 for item in dims):
        raise ValueError("hidden_dims must contain exactly two positive integers")
    return dims[0], dims[1]


class LegacyFrozenBatchNorm1d(nn.BatchNorm1d):
    """TF1-compatible batch normalization with permanently frozen moments.

    TensorFlow 1.14's default ``training=False`` used running mean zero and
    variance one because the legacy code never executed moving-statistic update
    operations.  The affine parameters remain ordinary trainable parameters.
    """

    def __init__(self, num_features: int, *, eps: float = 1.0e-3) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        self._check_input_dim(inputs)
        return F.batch_norm(
            inputs,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


def make_normalization(
    mode: str,
    features: int,
    *,
    eps: float | None = None,
) -> nn.Module:
    """Build one of the supported one-dimensional normalization layers."""

    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode in {"legacy", "legacy_frozen", "frozen"}:
        return LegacyFrozenBatchNorm1d(features, eps=1.0e-3 if eps is None else eps)
    if normalized_mode in {
        "batch",
        "batch_norm",
        "batchnorm",
        "standard",
        "standard_batch_norm",
    }:
        return nn.BatchNorm1d(features, eps=1.0e-5 if eps is None else eps)
    if normalized_mode in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(
        f"normalization must be 'legacy_frozen', 'batch_norm', or 'none'; got {mode!r}"
    )


def _reset_linear(
    layer: nn.Linear,
    initialization: str,
    generator: torch.Generator | None,
) -> None:
    scheme = initialization.strip().lower().replace("-", "_")
    if scheme in {"legacy", "legacy_glorot_normal", "legacy_tf14"}:
        # TF's GlorotNormal was also (unusually) assigned to each bias vector.
        # A 1-D Glorot fan has fan_in=fan_out=len(bias), hence std=1/sqrt(n).
        nn.init.xavier_normal_(layer.weight, generator=generator)
        if layer.bias is not None:
            nn.init.normal_(
                layer.bias,
                mean=0.0,
                std=1.0 / math.sqrt(layer.bias.numel()),
                generator=generator,
            )
        return
    if scheme in {"glorot_normal", "xavier_normal"}:
        nn.init.xavier_normal_(layer.weight, generator=generator)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return
    if scheme in {"glorot_uniform", "xavier_uniform"}:
        nn.init.xavier_uniform_(layer.weight, generator=generator)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return
    if scheme in {"pytorch", "pytorch_default", "default"}:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0), generator=generator)
        if layer.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in else 0.0
            nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        return
    raise ValueError(f"unknown linear initialization scheme: {initialization!r}")


def capped_relu_action(raw_output: Tensor, *, scale: float, bound: float) -> Tensor:
    """Apply the exact capped-ReLU action transform used by the thesis code.

    ``bound * (M - relu(M - relu(x))) / M`` maps any real network output to
    ``[0, bound]``.  The legacy experiment used ``M=2`` and ``bound=1``.
    """

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if bound <= 0.0:
        raise ValueError("bound must be positive")
    positive = F.relu(raw_output)
    return bound * (scale - F.relu(scale - positive)) / scale


class Actor(nn.Module):
    """Deterministic policy network with the thesis's capped-ReLU output."""

    def __init__(
        self,
        state_dim: int | Sequence[int] = 4,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (200, 100),
        *,
        action_bound: float = 1.0,
        output_scale: float = 2.0,
        normalization: str = "legacy_frozen",
        normalisation: str | None = None,
        batch_norm_mode: str | None = None,
        normalization_eps: float | None = None,
        initialization: str = "legacy_glorot_normal",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = _as_dim(state_dim, name="state_dim")
        self.action_dim = _as_dim(action_dim, name="action_dim")
        self.hidden_dims = _as_hidden_dims(hidden_dims)
        self.action_bound = float(action_bound)
        self.output_scale = float(output_scale)
        self.normalization_mode = batch_norm_mode or normalisation or normalization
        self.initialization = initialization
        if self.action_bound <= 0.0 or self.output_scale <= 0.0:
            raise ValueError("action_bound and output_scale must be positive")

        hidden1, hidden2 = self.hidden_dims
        # nn.Linear constructors initialize from Torch's global RNG before our
        # named generator overwrites them. Restore that global state so merely
        # constructing a model cannot perturb unrelated experiment streams.
        with torch.random.fork_rng(devices=[]):
            self.input_norm = make_normalization(
                self.normalization_mode, self.state_dim, eps=normalization_eps
            )
            self.fc1 = nn.Linear(self.state_dim, hidden1)
            self.norm1 = make_normalization(self.normalization_mode, hidden1, eps=normalization_eps)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.norm2 = make_normalization(self.normalization_mode, hidden2, eps=normalization_eps)
            self.output_layer = nn.Linear(hidden2, self.action_dim)
        self.reset_parameters(generator=generator)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> Actor:
        network = _section(config, "network")
        return cls(
            state_dim=_first(network, ("state_size", "state_dim", "state_dims"), 4),
            action_dim=_first(network, ("action_size", "action_dim", "action_dims"), 1),
            hidden_dims=_first(network, ("hidden_sizes", "hidden_dims"), (200, 100)),
            action_bound=_first(network, ("action_bound",), 1.0),
            output_scale=_first(network, ("actor_output_scale", "output_scale"), 2.0),
            normalization=_first(
                network,
                ("batch_norm_mode", "normalization", "normalisation"),
                "legacy_frozen",
            ),
            normalization_eps=_first(network, ("normalization_eps",), None),
            initialization=_first(network, ("initialization",), "legacy_glorot_normal"),
            generator=generator,
        )

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        for layer in (self.fc1, self.fc2, self.output_layer):
            _reset_linear(layer, self.initialization, generator)
        for module in (self.input_norm, self.norm1, self.norm2):
            if isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                module.reset_parameters()

    def forward(self, states: Tensor) -> Tensor:
        squeeze = states.ndim == 1
        if squeeze:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(
                f"states must have shape (batch, {self.state_dim}) or ({self.state_dim},)"
            )
        hidden = F.relu(self.norm1(self.fc1(self.input_norm(states))))
        hidden = F.relu(self.norm2(self.fc2(hidden)))
        actions = capped_relu_action(
            self.output_layer(hidden),
            scale=self.output_scale,
            bound=self.action_bound,
        )
        return actions.squeeze(0) if squeeze else actions


class Critic(nn.Module):
    """State-action value network matching the legacy additive architecture."""

    def __init__(
        self,
        state_dim: int | Sequence[int] = 4,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (200, 100),
        *,
        normalization: str = "legacy_frozen",
        normalisation: str | None = None,
        batch_norm_mode: str | None = None,
        normalization_eps: float | None = None,
        initialization: str = "legacy_glorot_normal",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = _as_dim(state_dim, name="state_dim")
        self.action_dim = _as_dim(action_dim, name="action_dim")
        self.hidden_dims = _as_hidden_dims(hidden_dims)
        self.normalization_mode = batch_norm_mode or normalisation or normalization
        self.initialization = initialization

        hidden1, hidden2 = self.hidden_dims
        with torch.random.fork_rng(devices=[]):
            self.state_input_norm = make_normalization(
                self.normalization_mode, self.state_dim, eps=normalization_eps
            )
            self.state_fc1 = nn.Linear(self.state_dim, hidden1)
            self.state_norm1 = make_normalization(
                self.normalization_mode, hidden1, eps=normalization_eps
            )
            self.state_fc2 = nn.Linear(hidden1, hidden2)
            self.state_norm2 = make_normalization(
                self.normalization_mode, hidden2, eps=normalization_eps
            )

            self.action_input_norm = make_normalization(
                self.normalization_mode, self.action_dim, eps=normalization_eps
            )
            self.action_fc = nn.Linear(self.action_dim, hidden2)
            self.action_norm = make_normalization(
                self.normalization_mode, hidden2, eps=normalization_eps
            )
            self.output_layer = nn.Linear(hidden2, 1)
        self.reset_parameters(generator=generator)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> Critic:
        network = _section(config, "network")
        return cls(
            state_dim=_first(network, ("state_size", "state_dim", "state_dims"), 4),
            action_dim=_first(network, ("action_size", "action_dim", "action_dims"), 1),
            hidden_dims=_first(network, ("hidden_sizes", "hidden_dims"), (200, 100)),
            normalization=_first(
                network,
                ("batch_norm_mode", "normalization", "normalisation"),
                "legacy_frozen",
            ),
            normalization_eps=_first(network, ("normalization_eps",), None),
            initialization=_first(network, ("initialization",), "legacy_glorot_normal"),
            generator=generator,
        )

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        for layer in (
            self.state_fc1,
            self.state_fc2,
            self.action_fc,
            self.output_layer,
        ):
            _reset_linear(layer, self.initialization, generator)
        for module in (
            self.state_input_norm,
            self.state_norm1,
            self.state_norm2,
            self.action_input_norm,
            self.action_norm,
        ):
            if isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                module.reset_parameters()

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        squeeze = states.ndim == 1
        if squeeze:
            states = states.unsqueeze(0)
        if actions.ndim == 1:
            if squeeze and actions.numel() == self.action_dim:
                actions = actions.unsqueeze(0)
            elif self.action_dim == 1:
                actions = actions.unsqueeze(-1)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(f"states must have shape (batch, {self.state_dim})")
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape (batch, {self.action_dim})")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same batch size")

        state_hidden = F.relu(self.state_norm1(self.state_fc1(self.state_input_norm(states))))
        state_hidden = self.state_norm2(self.state_fc2(state_hidden))
        action_hidden = self.action_norm(self.action_fc(self.action_input_norm(actions)))
        values = self.output_layer(F.relu(state_hidden + action_hidden))
        return values.squeeze(0) if squeeze else values


__all__ = [
    "Actor",
    "Critic",
    "LegacyFrozenBatchNorm1d",
    "capped_relu_action",
    "make_normalization",
]
