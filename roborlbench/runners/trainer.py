"""The training loop.

One class handles all twenty baselines and all twelve environments: the only
branch is ``on_policy`` (rollout + epochs) versus ``off_policy`` (replay buffer +
gradient steps per environment step).  Everything else - evaluation,
checkpointing, logging - is shared.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..algos import build_agent
from ..envs import BenchEnv, Obs, make_env
from .config import TrainConfig
from .evaluate import evaluate_policy
from .logging import RunLogger

__all__ = ["Trainer"]


class Trainer:
    """Wire one algorithm to one environment and train it."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.env: BenchEnv = make_env(
            config.env,
            num_envs=config.num_envs,
            seed=config.seed,
            device=config.device,
            **config.env_kwargs,
        )
        self.agent = build_agent(
            config.algo,
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            num_agents=self.env.num_agents,
            state_dim=self.env.state_dim,
            num_envs=self.env.num_envs,
            device=config.device,
            seed=config.seed,
            **config.algo_kwargs,
        )
        self.rng = np.random.default_rng(config.seed)
        self.logger = RunLogger(config.run_dir(), config.tensorboard)
        self.logger.write_json(
            "config.json",
            {
                **config.to_dict(),
                "obs_dim": self.env.obs_dim,
                "act_dim": self.env.act_dim,
                "state_dim": self.env.state_dim,
                "num_agents": self.env.num_agents,
                "env_backend": self.env.backend,
                "algo_family": "multi" if self.agent.multi_agent else "single",
            },
        )
        self.history: List[Dict[str, float]] = []
        self.best_eval = float("-inf")
        self.last_eval: Optional[float] = None
        self._obs: Obs = self.env.reset()
        self._episode_return = np.zeros(self.env.num_envs, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, Any]:
        start = time.time()
        if self.agent.kind == "on_policy":
            steps = self._train_on_policy()
        else:
            steps = self._train_off_policy()
        elapsed = time.time() - start

        final_eval = self.evaluate() if self.config.eval_interval > 0 else None
        ckpt = os.path.join(self.config.run_dir(), "final.pt")
        self.agent.save(ckpt)

        summary = {
            "env": self.config.env,
            "algo": self.config.algo,
            "steps": int(steps),
            "num_envs": self.config.num_envs,
            "num_agents": self.env.num_agents,
            "seed": self.config.seed,
            "device": self.config.device,
            "wall_time_s": round(elapsed, 2),
            "steps_per_s": round(steps / max(elapsed, 1e-9), 1),
            "best_eval_return": float(self.best_eval) if np.isfinite(self.best_eval) else None,
            "final_eval_return": final_eval,
            "checkpoint": ckpt,
        }
        self.logger.write_json("summary.json", summary)
        self.logger.close()
        return summary

    # ------------------------------------------------------------------ #
    # On-policy
    # ------------------------------------------------------------------ #
    def _train_on_policy(self) -> int:
        agent = self.agent
        buffer = agent.make_buffer()
        steps = 0
        total = int(self.config.total_steps)
        while steps < total:
            for _ in range(agent.rollout_steps):
                # feed the observation normaliser, exactly like the off-policy
                # path does in ``store``; the buffer keeps raw values
                agent.observe(self._obs.obs, self._obs.state)
                obs_t = self._tensor(self._obs.obs)
                state_t = self._tensor(self._obs.state)
                action, logp, value = agent.act_batch(obs_t, state_t)
                action_np = action.detach().cpu().numpy().astype(np.float32)
                step = self.env.step(action_np)
                buffer.add(
                    self._obs.obs, self._obs.state, action_np,
                    logp.detach().cpu(), value.detach().cpu(),
                    step.reward, step.done, step.timeout,
                )
                self._track(step)
                self._obs = Obs(step.obs, step.state)
                steps += self.env.num_envs

            last_value = agent.value_of(
                self._tensor(self._obs.state), self._tensor(self._obs.obs)
            )
            last_done = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self._tensor_device()
            )
            buffer.compute_returns(last_value, last_done)
            metrics = agent.update(buffer)
            buffer.reset()

            metrics = dict(metrics)
            metrics["train_return"] = float(self._episode_return.mean())
            self._after_iteration(steps, metrics)
        return steps

    # ------------------------------------------------------------------ #
    # Off-policy
    # ------------------------------------------------------------------ #
    def _train_off_policy(self) -> int:
        agent = self.agent
        steps = 0
        total = int(self.config.total_steps)
        while steps < total:
            if agent.ready():
                action = agent.act(self._obs.obs, self._obs.state)
            else:
                action = self.rng.uniform(
                    -1.0, 1.0,
                    size=(self.env.num_envs, self.env.num_agents, self.env.act_dim),
                ).astype(np.float32)
            step = self.env.step(action)
            agent.store(
                self._obs.obs, self._obs.state, action, step.reward,
                step.obs, step.state, step.done, step.timeout,
            )
            self._track(step)
            self._obs = Obs(step.obs, step.state)
            steps += self.env.num_envs

            metrics: Dict[str, float] = {}
            if agent.ready():
                for _ in range(max(1, agent.grad_steps)):
                    metrics = agent.update()
            metrics = dict(metrics or {})
            metrics["train_return"] = float(self._episode_return.mean())
            self._after_iteration(steps, metrics)
        return steps

    # ------------------------------------------------------------------ #
    # Evaluation / logging
    # ------------------------------------------------------------------ #
    def evaluate(self) -> float:
        """Run ``eval_episodes`` deterministic episodes and return their mean return."""
        result = evaluate_policy(
            self.agent, self.env, episodes=self.config.eval_episodes
        )
        self._obs = self.env.reset()
        self._episode_return[:] = 0.0
        value = float(result["eval_return"])
        self.last_eval = value
        self.best_eval = max(self.best_eval, value)
        return value

    def _after_iteration(self, steps: int, metrics: Dict[str, float]) -> None:
        row = dict(metrics)
        row["steps_per_s"] = 0.0
        if self.config.eval_interval > 0 and steps % self.config.eval_interval < self.env.num_envs:
            row["eval_return"] = self.evaluate()
        if self.config.save_interval > 0 and steps % self.config.save_interval < self.env.num_envs:
            self.agent.save(os.path.join(self.config.run_dir(), f"step_{steps}.pt"))
        self.history.append({"step": steps, **row})
        self.logger.log(steps, row)

    def _track(self, step) -> None:
        self._episode_return += step.reward.mean(axis=1)
        done = np.nonzero(step.done)[0]
        if done.size:
            self._episode_return[done] = 0.0

    # ------------------------------------------------------------------ #
    def _tensor_device(self) -> torch.device:
        return torch.device(self.config.device)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32,
                               device=self._tensor_device())
