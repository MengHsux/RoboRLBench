"""TD7 - A State-of-the-Art Off-Policy RL Algorithm (Fujimoto et al., ICML 2023).

TD7 is TD3 plus four components, each of which is cheap to implement and easy
to ablate - which is exactly why it makes a good benchmark entry:

* **Encoder** - a learned embedding ``Z(s, a)`` shared by the actor and the
  critics (the state-only branch feeds a zero action).
* **SALE** (Self-predictive *Averaged* *L*atent *E*ncoder) - snapshots of the
  encoder are averaged; the averaged encoder is used at evaluation time and is
  markedly more robust than any single snapshot.
* **LAP** (Latent *A*ction *P*lanning) - the replay buffer is sampled with a
  probability proportional to the transition's exploration value.  We use the
  absolute TD error as a tractable proxy for that value and document the
  substitution in ``docs/ALGORITHMS.md``.
* **LASER** - a self-supervised auxiliary loss that pulls ``Z(s', a')`` towards
  ``Z(s)`` and ``Z(s, a)``, turning the encoder into a multi-step predictor.

The target networks are TD3's (actor + critic), plus a target encoder.

Reference: https://arxiv.org/abs/2302.09419
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OffPolicyAgent
from ..nets import soft_update

__all__ = ["TD7"]


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class _Encoder(nn.Module):
    """``(obs | obs, act) -> z`` with LayerNorm (TD7's ``Z``)."""

    def __init__(self, in_dim: int, z_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ActorHead(nn.Module):
    def __init__(self, z_dim: int, act_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim, act_dim)
        nn.init.orthogonal_(self.fc.weight, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(z))


class _QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, 1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


class _TwinQ(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.q1 = _QNet(in_dim, hidden)
        self.q2 = _QNet(in_dim, hidden)

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        return self.q1(z, a), self.q2(z, a)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class TD7(OffPolicyAgent):
    name = "td7"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        hidden: int = 256,
        z_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        warmup: int = 5_000,
        grad_steps: int = 1,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        laser_coef: float = 1.0,
        checkpoint_interval: int = 250_000,
        max_checkpoints: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau,
            warmup=warmup, grad_steps=grad_steps, **kwargs,
        )
        assert self.num_agents == 1, "TD7 is a single-agent algorithm"
        self.hidden = int(hidden)
        self.z_dim = int(z_dim)
        self.policy_delay = int(policy_delay)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.laser_coef = float(laser_coef)
        self.checkpoint_interval = int(checkpoint_interval)
        self.max_checkpoints = int(max_checkpoints)

        self.encoder = _Encoder(self.obs_dim + self.act_dim, self.z_dim, self.hidden).to(self.device)
        self.actor = _ActorHead(self.z_dim, self.act_dim).to(self.device)
        self.critic = _TwinQ(self.z_dim + self.act_dim, self.hidden).to(self.device)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        # SALE: running average of the encoder snapshots (evaluation only)
        self.sale_encoder = copy.deepcopy(self.encoder)
        for net in (self.target_encoder, self.target_actor, self.target_critic, self.sale_encoder):
            for p in net.parameters():
                p.requires_grad_(False)

        # the encoder is trained by both objectives (critic + LASER, actor)
        self.actor_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr
        )
        self.critic_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # LAP: sample transitions proportional to their exploration value
        self.buffer.use_priority = True
        self._update_count = 0
        self._sale_count = 0
        self._env_steps = 0
        self._last_checkpoint = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _cat(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action is None:
            action = torch.zeros(state.shape[0], self.act_dim, device=state.device)
        return torch.cat([state, action], dim=-1)

    def _sale(self) -> None:
        """Fold the current encoder into the SALE running average."""
        self._sale_count += 1
        n = float(self._sale_count)
        with torch.no_grad():
            for avg, cur in zip(self.sale_encoder.parameters(), self.encoder.parameters()):
                avg.mul_((n - 1.0) / n).add_(cur, alpha=1.0 / n)
            for avg, cur in zip(self.sale_encoder.buffers(), self.encoder.buffers()):
                avg.mul_((n - 1.0) / n).add_(cur, alpha=1.0 / n)

    def _maybe_checkpoint(self) -> None:
        if self.checkpoint_interval <= 0 or self._sale_count >= self.max_checkpoints:
            return
        if self._env_steps - self._last_checkpoint >= self.checkpoint_interval:
            self._last_checkpoint = self._env_steps
            self._sale()

    # ------------------------------------------------------------------ #
    def store(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().store(*args, **kwargs)
        self._env_steps += self.num_envs

    def sample(self) -> Dict[str, torch.Tensor]:
        return self.buffer.sample(self.batch_size, generator=self._gen, return_index=True)

    # ------------------------------------------------------------------ #
    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b = obs.shape[0]
        o = obs.reshape(b, -1)
        encoder = self.sale_encoder if (deterministic and self._sale_count > 0) else self.encoder
        z = encoder(self._cat(o, None))
        return self.actor(z).reshape(b, self.num_agents, self.act_dim)

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        batch = self.sample()
        batch = self.normalize_batch(batch, self.obs_rms, self.state_rms)
        metrics = self._update_critic(batch)
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            metrics.update(self._update_actor(batch))
            self._soft_update_targets()
        self._maybe_checkpoint()
        return metrics

    # ------------------------------------------------------------------ #
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        obs = batch["obs"].reshape(b, -1)
        next_obs = batch["next_obs"].reshape(b, -1)
        action = batch["action"].reshape(b, -1)

        with torch.no_grad():
            z_next = self.target_encoder(self._cat(next_obs, None))
            noise = torch.randn(b, self.act_dim, device=self.device) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(z_next) + noise).clamp(-1.0, 1.0)
            z_next_action = self.target_encoder(self._cat(next_obs, next_action))
            q1_t, q2_t = self.target_critic(z_next_action, next_action)
            q_next = torch.min(q1_t, q2_t)
            target = batch["reward"].reshape(-1) + self.gamma * batch["discount"].reshape(-1) * q_next

        z_state = self.encoder(self._cat(obs, None))
        z_action = self.encoder(self._cat(obs, action))
        z_next_live = self.encoder(self._cat(next_obs, next_action))

        q1, q2 = self.critic(z_action, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # LASER: Z(s', a') predicts both Z(s) and Z(s, a) (targets detached)
        laser = F.mse_loss(z_next_live, z_state.detach()) + F.mse_loss(
            z_next_live.detach(), z_action
        )
        loss = critic_loss + self.laser_coef * laser

        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        # LAP: refresh the exploration priority of the sampled transitions
        td = 0.5 * ((q1.detach() - target).abs() + (q2.detach() - target).abs())
        self.buffer.set_priority(batch["row"], batch["env"], td)
        return {
            "critic_loss": float(critic_loss.detach()),
            "laser_loss": float(laser.detach()),
            "td_error": float(td.mean()),
        }

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        obs = batch["obs"].reshape(b, -1)
        z_state = self.encoder(self._cat(obs, None))
        action = self.actor(z_state)
        z_action = self.encoder(self._cat(obs, action))
        q1, _ = self.critic(z_action, action)
        actor_loss = -q1.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        self.actor_optim.step()
        return {"actor_loss": float(actor_loss.detach())}

    def _soft_update_targets(self) -> None:
        soft_update(self.target_encoder, self.encoder, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {
            "encoder": self.encoder,
            "actor": self.actor,
            "critic": self.critic,
            "target_encoder": self.target_encoder,
            "target_actor": self.target_actor,
            "target_critic": self.target_critic,
            "sale_encoder": self.sale_encoder,
        }

    def extra_state(self) -> Dict:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_count": self._update_count,
            "sale_count": self._sale_count,
            "env_steps": self._env_steps,
            "last_checkpoint": self._last_checkpoint,
        }

    def load_extra_state(self, state: Dict) -> None:
        if "actor_optim" in state:
            self.actor_optim.load_state_dict(state["actor_optim"])
        if "critic_optim" in state:
            self.critic_optim.load_state_dict(state["critic_optim"])
        self._update_count = int(state.get("update_count", 0))
        self._sale_count = int(state.get("sale_count", 0))
        self._env_steps = int(state.get("env_steps", 0))
        self._last_checkpoint = int(state.get("last_checkpoint", 0))
