"""DDPG inner loop and fictitious-play aggregate-law outer loop."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import DDPGAgent
from .artifacts import RunLayout, atomic_torch_save, atomic_write_json
from .config import ALMCoefficients, ReproductionConfig
from .models import Actor
from .runtime import RNGStream, configure_runtime
from .simulation import (
    ActorPolicy,
    EnsemblePolicy,
    KrusellSmithEnvironment,
    PopulationSimulator,
)
from .statistics import ALMRegression, RegimeFit, regress_alm
from .transitions import build_transition_matrices


def agent_config(config: ReproductionConfig) -> dict[str, Any]:
    """Translate the strict public config into the deliberately tolerant RL API."""

    return {
        "network": {
            "state_dim": config.network.state_size,
            "action_dim": config.network.action_size,
            "hidden_dims": config.network.hidden_sizes,
            "action_bound": config.network.action_bound,
            "actor_output_scale": config.network.actor_output_scale,
            "normalization": config.network.normalization,
            "initialization": config.network.initialization,
        },
        "exploration": {
            "kind": config.exploration.kind,
            "uniform_low": config.exploration.uniform_low,
            "uniform_high": config.exploration.uniform_high,
        },
        "ddpg": {
            "discount_factor": config.economy.beta,
            "target_update_rate": config.network.target_update_rate,
            "batch_size": config.network.batch_size,
            "replay_capacity": config.network.replay_capacity,
            "actor_learning_rate": config.network.actor_learning_rate,
            "critic_learning_rate": config.network.critic_learning_rate,
            "exploration_noise": config.exploration.kind,
            "ou_noise": {
                "mean": config.network.ou_mean,
                "sigma": config.network.ou_sigma,
                "theta": config.network.ou_theta,
                "dt": config.network.ou_dt,
                "initial": 0.0,
            },
            "uniform_noise": {
                "low": config.exploration.uniform_low,
                "high": config.exploration.uniform_high,
            },
            "replay_sample_with_replacement": config.network.replay_sample_with_replacement,
            "clip_exploration_action": False,
            "l2_weight_decay": 0.0,
            "target_init_mode": (
                "legacy_soft"
                if config.compatibility.active("use_soft_target_initialization_bug")
                else "hard"
            ),
        },
    }


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _load_actor_snapshot(
    state: Mapping[str, torch.Tensor],
    config: ReproductionConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Actor:
    actor = Actor.from_config(agent_config(config)).to(device=device, dtype=dtype)
    actor.load_state_dict(state)
    actor.eval().requires_grad_(False)
    return actor


def _fallback_regression(coefficients: ALMCoefficients) -> ALMRegression:
    bad = RegimeFit(
        coefficients.bad_intercept,
        coefficients.bad_slope,
        0.0,
        math.inf,
        0,
    )
    good = RegimeFit(
        coefficients.good_intercept,
        coefficients.good_slope,
        0.0,
        math.inf,
        0,
    )
    return ALMRegression(coefficients, bad, good)


class FictitiousPlayTrainer:
    checkpoint_version = 1

    def __init__(
        self,
        config: ReproductionConfig,
        layout: RunLayout,
        *,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.layout = layout.create(allow_existing=resume)
        self.streams, self.device, _ = configure_runtime(config.runtime)
        # TensorFlow 1 trained the networks in float32. Economics/statistics
        # modules explicitly use float64 regardless of this network dtype.
        self.network_dtype = torch.float32
        self.matrices = build_transition_matrices(config.economy, config.transitions)
        self.environment = KrusellSmithEnvironment(
            config,
            self.streams.numpy(RNGStream.TRAIN_TRANSITIONS),
            matrices=self.matrices,
            alm=config.initial_alm,
        )
        self.population = PopulationSimulator(config, matrices=self.matrices)
        self.agent = DDPGAgent(
            agent_config(config),
            device=self.device,
            dtype=self.network_dtype,
            seed=self.streams.seed_for(RNGStream.NETWORK_INITIALIZATION),
            target_init_mode=(
                "legacy_soft"
                if config.compatibility.active("use_soft_target_initialization_bug")
                else "hard"
            ),
        )
        self.completed_round = 0
        self.actor_snapshots: list[dict[str, torch.Tensor]] = []
        self.discounted_returns: list[float] = []
        self.undiscounted_returns: list[float] = []
        self.loss_history: list[dict[str, float]] = []
        self.pretrain_history: list[dict[str, float]] = []
        self.pretraining_completed = False
        self.alm_history: list[ALMCoefficients] = [config.initial_alm]
        self.current_policy_alm_history: list[ALMCoefficients] = [config.initial_alm]
        self.r2_history: list[tuple[float, float]] = [(0.0, 0.0)]
        self.current_policy_r2_history: list[tuple[float, float]] = [(0.0, 0.0)]
        self.round_seconds: list[float] = []
        if resume:
            self.resume()

    @property
    def log_path(self) -> Path:
        return self.layout.logs / "events.jsonl"

    def log(self, event: str, **payload: Any) -> None:
        record = {"time": time.time(), "event": event, **payload}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _episode(self) -> tuple[float, float]:
        state = self.environment.reset()
        if not self.config.compatibility.active("keep_ou_state_between_episodes"):
            self.agent.reset_noise()
        discounted_return = 0.0
        undiscounted_return = 0.0
        discount = 1.0
        while True:
            action = self.agent.select_action(state, explore=True, as_numpy=True)
            transition = self.environment.step(state, action)
            self.agent.remember(
                state,
                action,
                transition.reward,
                transition.state,
                transition.terminated,
            )
            losses = self.agent.update()
            if losses is not None and self.agent.update_count % 100 == 0:
                self.loss_history.append(losses)
            undiscounted_return += transition.reward
            discounted_return += discount * transition.reward
            discount *= self.config.economy.beta
            state = transition.state
            if transition.terminated:
                break
        return discounted_return, undiscounted_return

    def _sample_pretraining_states(self, batch_size: int) -> np.ndarray:
        """Draw the state distribution used by the retained warm-start code."""

        rng = self.streams.numpy(RNGStream.TRAIN_INITIAL_STATES)
        upper = 2.0 * self.config.economy.initial_aggregate_capital
        capital = rng.uniform(
            self.config.economy.minimum_positive,
            upper,
            size=(batch_size, 2),
        )
        regimes = rng.integers(0, 2, size=(batch_size, 2))
        return np.concatenate((capital, regimes), axis=1).astype(np.float32)

    def _pretrain_actor(self) -> None:
        settings = self.config.training
        self.log(
            "actor_pretraining_started",
            steps=settings.pretrain_steps,
            batch_size=settings.pretrain_batch_size,
            target_ratio=settings.pretrain_target_ratio,
            historical_source_target_ratio=0.7,
        )
        progress_interval = max(1, settings.pretrain_steps // 10)
        for step in range(1, settings.pretrain_steps + 1):
            states = self._sample_pretraining_states(settings.pretrain_batch_size)
            metrics = self.agent.pretrain_actor_step(
                states,
                target_ratio=settings.pretrain_target_ratio,
            )
            self.pretrain_history.append(metrics)
            if step % progress_interval == 0 or step == settings.pretrain_steps:
                self.log("actor_pretraining_progress", step=step, **metrics)
        # The historical warm start hard-synchronized its targets afterward,
        # even when the legacy soft-initialization quirk was otherwise active.
        self.agent.hard_sync_targets()
        self.pretraining_completed = True
        self.log(
            "actor_pretraining_completed",
            steps=settings.pretrain_steps,
            final_loss=self.pretrain_history[-1]["actor_pretrain_loss"],
        )

    def _snapshot_policy(self) -> Actor:
        state = _cpu_state_dict(self.agent.actor)
        self.actor_snapshots.append(state)
        return _load_actor_snapshot(state, self.config, self.device, self.network_dtype)

    def _ensemble_policy(self) -> EnsemblePolicy:
        actors = [
            ActorPolicy(_load_actor_snapshot(state, self.config, self.device, self.network_dtype))
            for state in self.actor_snapshots
        ]
        return EnsemblePolicy(actors)

    def fit_alm(self, policy: Any) -> ALMRegression:
        capital_paths: list[np.ndarray] = []
        state_paths: list[np.ndarray] = []
        rng = self.streams.numpy(RNGStream.ALM_SIMULATION)
        desired = self.config.training.alm_simulations
        maximum = max(desired, 2) * 10
        for _ in range(maximum):
            paths = self.population.simulate(
                policy,
                population=self.config.training.alm_population,
                horizon=self.config.training.horizon,
                rng=rng,
                initial_aggregate_state=(
                    self.config.training.alm_initial_aggregate_state
                ),
            )
            capital_paths.append(paths.aggregate_capital)
            state_paths.append(paths.aggregate_state)
            if len(capital_paths) >= desired:
                regimes = np.concatenate([path[:-1] for path in state_paths])
                if np.count_nonzero(regimes == 0) >= 2 and np.count_nonzero(regimes == 1) >= 2:
                    break
        try:
            return regress_alm(np.asarray(capital_paths), np.asarray(state_paths))
        except ValueError:
            return _fallback_regression(self.environment.alm)

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "config": self.config.to_dict(),
            "completed_round": self.completed_round,
            "agent": self.agent.state_dict(include_replay=True, include_noise=True),
            "rng_streams": self.streams.state_dict(),
            "actor_snapshots": self.actor_snapshots,
            "discounted_returns": self.discounted_returns,
            "undiscounted_returns": self.undiscounted_returns,
            "loss_history": self.loss_history,
            "pretrain_history": self.pretrain_history,
            "pretraining_completed": self.pretraining_completed,
            "alm_history": [item.to_bad_good() for item in self.alm_history],
            "current_policy_alm_history": [
                item.to_bad_good() for item in self.current_policy_alm_history
            ],
            "r2_history": self.r2_history,
            "current_policy_r2_history": self.current_policy_r2_history,
            "round_seconds": self.round_seconds,
        }

    def save_checkpoint(self) -> Path:
        destination = self.layout.checkpoint(self.completed_round)
        state = self._checkpoint_state()
        atomic_torch_save(destination, state)
        atomic_torch_save(self.layout.checkpoints / "last.pt", state)
        return destination

    def resume(self) -> None:
        path = self.layout.checkpoints / "last.pt"
        if not path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint exists at {path}")
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        if state.get("config") != self.config.to_dict():
            raise ValueError("checkpoint configuration differs from the requested configuration")
        self.completed_round = int(state["completed_round"])
        self.agent.load_state_dict(state["agent"], load_replay=True, load_noise=True)
        self.streams.load_state_dict(state["rng_streams"])
        self.actor_snapshots = list(state["actor_snapshots"])
        self.discounted_returns = list(state["discounted_returns"])
        self.undiscounted_returns = list(state["undiscounted_returns"])
        self.loss_history = list(state["loss_history"])
        self.pretrain_history = list(state.get("pretrain_history", []))
        self.pretraining_completed = bool(
            state.get("pretraining_completed", self.completed_round > 0)
        )
        self.alm_history = [ALMCoefficients.from_bad_good(item) for item in state["alm_history"]]
        self.current_policy_alm_history = [
            ALMCoefficients.from_bad_good(item) for item in state["current_policy_alm_history"]
        ]
        self.r2_history = [tuple(item) for item in state["r2_history"]]
        self.current_policy_r2_history = [
            tuple(item) for item in state["current_policy_r2_history"]
        ]
        self.round_seconds = list(state["round_seconds"])
        self.environment.set_alm(self.alm_history[-1])
        self.log("resumed", completed_round=self.completed_round)

    def _write_history(self) -> None:
        np.savez_compressed(
            self.layout.data / "training_history.npz",
            discounted_returns=np.asarray(self.discounted_returns),
            undiscounted_returns=np.asarray(self.undiscounted_returns),
            alm=np.asarray([item.to_bad_good() for item in self.alm_history]),
            current_policy_alm=np.asarray(
                [item.to_bad_good() for item in self.current_policy_alm_history]
            ),
            r2=np.asarray(self.r2_history),
            current_policy_r2=np.asarray(self.current_policy_r2_history),
            round_seconds=np.asarray(self.round_seconds),
            actor_pretrain_loss=np.asarray(
                [item["actor_pretrain_loss"] for item in self.pretrain_history]
            ),
        )

    def train(self) -> FictitiousPlayTrainer:
        if (
            self.completed_round == 0
            and self.config.training.pretrain_actor
            and not self.pretraining_completed
        ):
            self._pretrain_actor()

        for round_number in range(self.completed_round + 1, self.config.training.rounds + 1):
            started = time.perf_counter()
            self.agent.reset_memory()
            episodes = (
                self.config.training.first_round_episodes
                if round_number == 1
                else self.config.training.episodes_per_round
            )
            self.log("round_started", round=round_number, episodes=episodes)
            for episode in range(episodes):
                discounted, raw = self._episode()
                self.discounted_returns.append(discounted)
                self.undiscounted_returns.append(raw)
                if (episode + 1) % max(1, min(1000, episodes // 10 or 1)) == 0:
                    self.log(
                        "training_progress",
                        round=round_number,
                        episode=episode + 1,
                        discounted_return=discounted,
                        trailing_mean=float(np.mean(self.discounted_returns[-100:])),
                    )

            current_actor = self._snapshot_policy()
            current_fit = self.fit_alm(ActorPolicy(current_actor))
            equilibrium_fit = self.fit_alm(self._ensemble_policy())
            self.current_policy_alm_history.append(current_fit.coefficients)
            self.current_policy_r2_history.append(
                (current_fit.bad.r_squared, current_fit.good.r_squared)
            )
            self.alm_history.append(equilibrium_fit.coefficients)
            self.r2_history.append((equilibrium_fit.bad.r_squared, equilibrium_fit.good.r_squared))
            self.environment.set_alm(equilibrium_fit.coefficients)
            self.completed_round = round_number
            elapsed = time.perf_counter() - started
            self.round_seconds.append(elapsed)
            checkpoint = self.save_checkpoint()
            self._write_history()
            difference = max(
                abs(new - old)
                for new, old in zip(
                    self.alm_history[-1].to_bad_good(),
                    self.alm_history[-2].to_bad_good(),
                    strict=True,
                )
            )
            self.log(
                "round_completed",
                round=round_number,
                seconds=elapsed,
                alm=list(equilibrium_fit.coefficients.to_bad_good()),
                alm_max_change=difference,
                checkpoint=str(checkpoint),
            )
            if round_number > 1 and difference < self.config.training.alm_tolerance:
                self.log(
                    "alm_tolerance_reached",
                    round=round_number,
                    alm_max_change=difference,
                    diagnostic_only=True,
                )

        self._write_history()
        atomic_write_json(
            self.layout.root / "run.json",
            {
                "pipeline": {"stages": {"training": "completed"}},
                "completed_round": self.completed_round,
                "episodes": len(self.discounted_returns),
            },
        )
        return self
