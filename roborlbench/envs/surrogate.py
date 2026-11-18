"""Built-in analytic environments (pure NumPy, CPU only).

These two environments exist for three reasons:

1. **They run anywhere.**  No Isaac Gym, no Isaac Lab, no MuJoCo - a laptop
   without a GPU can execute the complete benchmark (all 20 algorithms on both
   a single-agent and a multi-agent task) in minutes.
2. **They are unit-test fixtures.**  ``tests/`` uses them to prove that every
   algorithm actually performs gradient updates and improves the return.
3. **They are research sandboxes.**  ``PointReach`` isolates the optimisation
   problem (dense quadratic reward, 2 actuators) and ``Hunt2D`` is a 2-D
   abstraction of the quadruped wolf-hunt task: ``A`` agents have to encircle a
   moving prey at a prescribed radius without colliding.  Ideas can be screened
   here before touching a GPU simulator.

Both environments are fully vectorised over ``num_envs`` task copies.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import BenchEnv, Obs, Step

__all__ = ["PointReachEnv", "Hunt2DEnv"]


class _AnalyticEnv(BenchEnv):
    """Shared bookkeeping: per-environment RNG, counters, auto-reset."""

    backend = "analytic"

    def __init__(self, num_envs: int = 1, seed: int = 0) -> None:
        super().__init__()
        self.num_envs = int(num_envs)
        self._seeds = np.arange(self.num_envs, dtype=np.int64) + int(seed) * 10_007
        self._rng = np.random.default_rng(int(seed))
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._success = np.zeros(self.num_envs, dtype=bool)
        self._first_reset = True

    # ------------------------------------------------------------------ #
    # Sub-class hooks
    # ------------------------------------------------------------------ #
    def _reset_idx(self, idx: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _obs(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _state(self) -> np.ndarray:
        return self._obs().reshape(self.num_envs, -1)

    def _step_idx(self, idx: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Advance the selected envs, return ``(len(idx), num_agents)`` rewards."""
        raise NotImplementedError

    def _is_done(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # BenchEnv API
    # ------------------------------------------------------------------ #
    def reset(self, mask: Optional[np.ndarray] = None) -> Obs:
        if mask is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.flatnonzero(mask)
        if idx.size:
            self._reset_idx(idx)
            self._t[idx] = 0
            self._done[idx] = False
            self._success[idx] = False
        self._first_reset = False
        return self._make_obs(self._obs(), self._state())

    def step(self, actions: np.ndarray) -> Step:
        actions = self.check_actions(actions)
        rewards = self._step_idx(np.arange(self.num_envs), actions)
        self._t += 1

        done = self._is_done()
        timeout = (self._t >= self.max_episode_steps) & ~done
        done = done | timeout

        info: Dict[str, object] = {"timeout": timeout}
        if self._success.any():
            info["metrics"] = {"success_rate": float(self._success.mean())}

        if done.any():
            # auto-reset the finished copies; the returned obs are post-reset
            self.reset(mask=done)

        return Step(
            obs=self._obs(),
            state=self._state(),
            reward=rewards.astype(np.float32),
            done=done,
            info=info,
        )


# --------------------------------------------------------------------------- #
# Single agent task
# --------------------------------------------------------------------------- #
class PointReachEnv(_AnalyticEnv):
    """A point mass has to reach a randomly placed goal.

    * observation: ``[pos(2), vel(2), goal(2), prev_action(2)]`` -> dim 8
    * action: 2-D acceleration in ``[-1, 1]``
    * reward: ``-||pos - goal||`` plus a ``+10`` bonus on success
    * horizon: 100 steps, success radius ``0.08``

    The optimal policy is a bang-bang controller, so any correctly implemented
    actor-critic should reach a return close to ``-2.0`` within ~30k steps.
    """

    obs_dim = 8
    act_dim = 2
    max_episode_steps = 100

    SUCCESS_RADIUS = 0.08
    DT = 0.1
    DAMPING = 0.9
    MAX_SPEED = 1.0
    ARENA = 1.5

    def __init__(self, num_envs: int = 1, seed: int = 0) -> None:
        super().__init__(num_envs=num_envs, seed=seed)
        self.state_dim = self.obs_dim
        self.pos = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.vel = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.goal = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.prev_action = np.zeros((self.num_envs, 2), dtype=np.float32)

    def _reset_idx(self, idx: np.ndarray) -> None:
        n = idx.size
        self.pos[idx] = self._rng.uniform(-self.ARENA, self.ARENA, size=(n, 2)).astype(np.float32)
        self.vel[idx] = 0.0
        self.goal[idx] = self._rng.uniform(-self.ARENA, self.ARENA, size=(n, 2)).astype(np.float32)
        self.prev_action[idx] = 0.0

    def _obs(self) -> np.ndarray:
        obs = np.concatenate([self.pos, self.vel, self.goal, self.prev_action], axis=-1)
        return obs[:, None, :].astype(np.float32)

    def _step_idx(self, idx: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = np.clip(act[idx, 0], -1.0, 1.0).astype(np.float32)
        self.vel[idx] = self.DAMPING * self.vel[idx] + a * self.DT * 10.0
        speed = np.linalg.norm(self.vel[idx], axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.MAX_SPEED / np.maximum(speed, 1e-6))
        self.vel[idx] = self.vel[idx] * scale
        self.pos[idx] = np.clip(self.pos[idx] + self.vel[idx] * self.DT,
                                -self.ARENA, self.ARENA)
        self.prev_action[idx] = a

        dist = np.linalg.norm(self.pos[idx] - self.goal[idx], axis=-1)
        rew = -dist + 10.0 * (dist < self.SUCCESS_RADIUS)
        self._success[idx] = dist < self.SUCCESS_RADIUS
        return rew[:, None].astype(np.float32)

    def _is_done(self) -> np.ndarray:
        return np.linalg.norm(self.pos - self.goal, axis=-1) < self.SUCCESS_RADIUS


# --------------------------------------------------------------------------- #
# Multi agent task
# --------------------------------------------------------------------------- #
class Hunt2DEnv(_AnalyticEnv):
    """Cooperative encirclement of a moving prey - the 2-D twin of the wolf hunt.

    ``A`` agents (default 3) must occupy the vertices of a regular polygon of
    radius ``R`` around a prey that drifts inside the arena.  Each agent has

    * observation: ``[pos(2), vel(2), prey_rel(2), target_rel(2), prev_action(2)]``
      plus ``(A-1)`` neighbour blocks ``[rel_pos(2), rel_vel(2)]``
      -> ``obs_dim = 10 + 4 * (A - 1)``
    * action: 2-D acceleration in ``[-1, 1]``
    * reward (shared): ``-sum_i ||p_i - t_i||`` penalty, a per-step
      ``-0.02 * n_collisions`` term and ``+A`` when *every* agent is within
      ``0.12`` of its ring slot (task success)
    * global state: the concatenation of all observations plus the prey pose
      -> ``state_dim = A * obs_dim + 4``

    The environment is a *cooperative* MARL problem with a shared reward, which
    is exactly the setting of the legged wolf-hunt benchmark.
    """

    act_dim = 2
    max_episode_steps = 150

    RING_RADIUS = 1.0
    SUCCESS_TOL = 0.12
    COLLISION_DIST = 0.25
    DT = 0.1
    DAMPING = 0.88
    MAX_SPEED = 1.0
    ARENA = 2.0
    PREY_SPEED = 0.22

    def __init__(self, num_envs: int = 1, seed: int = 0, num_agents: int = 3) -> None:
        super().__init__(num_envs=num_envs, seed=seed)
        self.num_agents = int(num_agents)
        self.obs_dim = 10 + 4 * (self.num_agents - 1)
        self.state_dim = self.num_agents * self.obs_dim + 4

        A = self.num_agents
        self.pos = np.zeros((self.num_envs, A, 2), dtype=np.float32)
        self.vel = np.zeros((self.num_envs, A, 2), dtype=np.float32)
        self.prev_action = np.zeros((self.num_envs, A, 2), dtype=np.float32)
        self.prey_pos = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.prey_vel = np.zeros((self.num_envs, 2), dtype=np.float32)
        # fixed slot assignment: agent i owns angle 2*pi*i/A on the ring
        angles = 2.0 * np.pi * np.arange(A) / A
        self._slot_dir = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _reset_idx(self, idx: np.ndarray) -> None:
        n, A = idx.size, self.num_agents
        self.prey_pos[idx] = self._rng.uniform(-0.6, 0.6, size=(n, 2)).astype(np.float32)
        angle = self._rng.uniform(0.0, 2.0 * np.pi, size=n)
        self.prey_vel[idx] = self.PREY_SPEED * np.stack(
            [np.cos(angle), np.sin(angle)], axis=-1
        ).astype(np.float32)
        # spawn agents on a slightly larger ring so they must close in
        radius = self.RING_RADIUS * 1.6
        offset = angle[:, None]
        for i in range(A):
            ang = 2.0 * np.pi * i / A + offset[:, 0]
            self.pos[idx, i, 0] = self.prey_pos[idx, 0] + radius * np.cos(ang)
            self.pos[idx, i, 1] = self.prey_pos[idx, 1] + radius * np.sin(ang)
        self.pos[idx] = np.clip(self.pos[idx], -self.ARENA, self.ARENA)
        self.vel[idx] = 0.0
        self.prev_action[idx] = 0.0

    def _targets(self) -> np.ndarray:
        """``(N, A, 2)`` ring slot for every agent."""
        return self.prey_pos[:, None, :] + self.RING_RADIUS * self._slot_dir[None, :, :]

    def _obs(self) -> np.ndarray:
        N, A = self.num_envs, self.num_agents
        prey_rel = self.prey_pos[:, None, :] - self.pos                 # (N, A, 2)
        target_rel = self._targets() - self.pos                        # (N, A, 2)
        blocks = [prey_rel, target_rel]
        if A > 1:
            rel_pos = self.pos[:, None, :, :] - self.pos[:, :, None, :]      # (N, A, A, 2)
            rel_vel = self.vel[:, None, :, :] - self.vel[:, :, None, :]
            eye = np.eye(A, dtype=bool)
            neigh = rel_pos[~np.broadcast_to(eye, (N, A, A))].reshape(N, A, A - 1, 2)
            neigh_v = rel_vel[~np.broadcast_to(eye, (N, A, A))].reshape(N, A, A - 1, 2)
            blocks += [neigh.reshape(N, A, -1), neigh_v.reshape(N, A, -1)]
        obs = np.concatenate(
            [self.pos, self.vel] + blocks + [self.prev_action], axis=-1
        )
        return obs.astype(np.float32)

    def _state(self) -> np.ndarray:
        obs = self._obs().reshape(self.num_envs, -1)
        return np.concatenate([obs, self.prey_pos, self.prey_vel], axis=-1).astype(np.float32)

    def _step_idx(self, idx: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = np.clip(act[idx], -1.0, 1.0).astype(np.float32)
        # ---- agents ---------------------------------------------------- #
        self.vel[idx] = self.DAMPING * self.vel[idx] + a * self.DT * 10.0
        speed = np.linalg.norm(self.vel[idx], axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.MAX_SPEED / np.maximum(speed, 1e-6))
        self.vel[idx] = self.vel[idx] * scale
        self.pos[idx] = np.clip(self.pos[idx] + self.vel[idx] * self.DT,
                                -self.ARENA, self.ARENA)
        self.prev_action[idx] = a

        # ---- prey: constant velocity that bounces off the arena walls -- #
        p = self.prey_pos[idx] + self.prey_vel[idx] * self.DT
        bounce = np.abs(p) > (self.ARENA - 0.5)
        self.prey_vel[idx] = np.where(bounce, -self.prey_vel[idx], self.prey_vel[idx])
        self.prey_pos[idx] = np.clip(p, -(self.ARENA - 0.5), self.ARENA - 0.5)

        # ---- reward ---------------------------------------------------- #
        err = np.linalg.norm(self.pos[idx] - self._targets()[idx], axis=-1)   # (n, A)
        rew = -err.sum(axis=-1)

        dist_ij = np.linalg.norm(
            self.pos[idx][:, :, None, :] - self.pos[idx][:, None, :, :], axis=-1
        )
        if self.num_agents > 1:
            iu = np.triu_indices(self.num_agents, k=1)
            collisions = (dist_ij[:, iu[0], iu[1]] < self.COLLISION_DIST).sum(axis=-1)
            rew = rew - 0.02 * collisions
        inside = (err < self.SUCCESS_TOL).all(axis=-1)
        rew = rew + self.num_agents * inside
        self._success[idx] = inside

        return np.repeat(rew[:, None], self.num_agents, axis=-1).astype(np.float32)

    def _is_done(self) -> np.ndarray:
        err = np.linalg.norm(self.pos - self._targets(), axis=-1)
        return (err < self.SUCCESS_TOL).all(axis=-1)


# --------------------------------------------------------------------------- #
# Factories used by the environment registry
# --------------------------------------------------------------------------- #
def make_point_reach(num_envs: int = 1, seed: int = 0, device: str = "cpu", **_) -> PointReachEnv:
    """``point-reach`` factory."""
    return PointReachEnv(num_envs=num_envs, seed=seed)


def make_multi_hunt(
    num_envs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    num_agents: int = 3,
    **_,
) -> Hunt2DEnv:
    """``multi-hunt-2d`` factory."""
    return Hunt2DEnv(num_envs=num_envs, seed=seed, num_agents=num_agents)
