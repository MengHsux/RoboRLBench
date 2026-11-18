"""MADDPG - Multi-Agent Deep Deterministic Policy Gradient
(Lowe et al., NeurIPS 2017).

The canonical CTDE baseline: every agent has a deterministic actor that only
sees its own observation, and a **centralised critic** ``Q_i(s, a_1..a_N)`` that
sees the global state and the joint action.  Training is off-policy with target
networks and soft updates.

Parameter sharing is used for the actors (one shared actor) and the critics are
one network with an output dimension of ``num_agents`` - i.e. the ``A`` critics
share their trunk.  This is the standard implementation when the agents are
homogeneous, which is the case for all four benchmark environments.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import DeterministicActor, soft_update
from .base import MARLOffPolicy, joint_action
from .nets_marl import CentralQ

__all__ = ["MADDPG"]


class MADDPG(MARLOffPolicy):
    name = "maddpg"
    policy_delay = 1

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        hidden=(256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        warmup: int = 2_000,
        grad_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau,
            warmup=warmup, grad_steps=grad_steps, **kwargs,
        )
        self.hidden = tuple(hidden)

        self.actor = self.build_actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = self.build_critic()
        self.critic_target = copy.deepcopy(self.critic)
        for net in (self.actor_target, self.critic_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self._update_count = 0

    # ------------------------------------------------------------------ #
    def build_actor(self) -> nn.Module:
        return DeterministicActor(self.obs_dim, self.act_dim, self.hidden).to(self.device)

    def build_critic(self) -> nn.Module:
        return CentralQ(
            self.state_dim, self.num_agents, self.act_dim, self.hidden
        ).to(self.device)

    def _agent_act(self, flat_obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return self.actor(flat_obs)

    # ------------------------------------------------------------------ #
    def _target_actions(self, next_obs: torch.Tensor) -> torch.Tensor:
        b = next_obs.shape[0]
        flat = next_obs.reshape(b * self.num_agents, -1)
        return self.actor_target(flat).reshape(b, self.num_agents, self.act_dim)

    def _target_q(self, next_state: torch.Tensor, next_joint: torch.Tensor) -> torch.Tensor:
        return self.critic_target(next_state, next_joint)

    def _q(self, state: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        return self.critic(state, joint)

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
        return metrics

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            next_action = self._target_actions(batch["next_obs"])
            q_next = self._target_q(batch["next_state"], joint_action(next_action))
            target = batch["reward"] + self.gamma * batch["discount"].unsqueeze(-1) * q_next

        q = self._q(batch["state"], joint_action(batch["action"]))
        critic_loss = F.mse_loss(q, target)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # agent i's loss differentiates through a_i only (the other actions are
        # the ones stored in the buffer); the per-agent terms share one graph.
        actor_loss = torch.zeros((), device=self.device)
        for i in range(self.num_agents):
            own = self.actor(batch["obs"][:, i])
            joint = torch.cat(
                [batch["action"][:, :i], own.unsqueeze(1), batch["action"][:, i + 1:]], dim=1
            )
            q = self._q(batch["state"], joint_action(joint))[:, i]
            actor_loss = actor_loss - q.mean()
        actor_loss = actor_loss / float(self.num_agents)

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()
        return {"actor_loss": float(actor_loss.detach())}

    def _soft_update_targets(self) -> None:
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
        }

    def extra_state(self) -> Dict:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_count": self._update_count,
        }

    def load_extra_state(self, state: Dict) -> None:
        if "actor_optim" in state:
            self.actor_optim.load_state_dict(state["actor_optim"])
        if "critic_optim" in state:
            self.critic_optim.load_state_dict(state["critic_optim"])
        self._update_count = int(state.get("update_count", 0))
