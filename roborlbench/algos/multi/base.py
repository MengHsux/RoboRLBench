"""Shared machinery for the ten multi-agent baselines.

Every algorithm here follows the **CTDE** pattern (centralised training with
decentralised execution):

* actors are *decentralised* - they only see the agent's own observation - and
  are shared across agents (parameter sharing), which is what lets the same
  baseline run on a 2-agent point-mass and on a 3-wolf pack;
* critics are *centralised* - they see the global state from
  :class:`~roborlbench.envs.base.BenchEnv` - and output **one value per agent**
  so that both shared (team) and individual rewards are supported.

Two families:

``MARLOnPolicy``
    IPPO / MAPPO / HAPPO / HATRPO / MAT.  ``centralized=False`` gives the
    independent critic of IPPO, ``centralized=True`` the global-state critic of
    MAPPO and the sequential methods.
``MARLOffPolicy``
    MADDPG / MATD3 / MASAC / HASAC / FACMAC.  The critic consumes the global
    state *and* the joint action.
"""

from __future__ import annotations

import abc
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import OffPolicyAgent, OnPolicyAgent
from ..buffers import RolloutBuffer
from ..nets import GaussianActor, mlp

__all__ = ["MARLOnPolicy", "MARLOffPolicy", "joint_action", "split_action"]


def joint_action(action: torch.Tensor) -> torch.Tensor:
    """``(B, A, d) -> (B, A*d)``."""
    return action.reshape(action.shape[0], -1)


def split_action(joint: torch.Tensor, num_agents: int, act_dim: int) -> torch.Tensor:
    """``(B, A*d) -> (B, A, d)``."""
    return joint.reshape(joint.shape[0], num_agents, act_dim)


# --------------------------------------------------------------------------- #
# On-policy family
# --------------------------------------------------------------------------- #
class MARLOnPolicy(OnPolicyAgent):
    """Clustered PPO for cooperative multi-agent control."""

    multi_agent = True
    #: ``True`` -> critic sees the global state (MAPPO), ``False`` -> own obs (IPPO)
    centralized: bool = True

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        actor_hidden=(256, 128),
        critic_hidden=(256, 128),
        lr: float = 3e-4,
        critic_lr: Optional[float] = None,
        clip_range: float = 0.2,
        value_clip: float = 0.0,
        entropy_coef: float = 0.0,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )
        self.clip_range = float(clip_range)
        self.value_clip = float(value_clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.actor = GaussianActor(
            self.obs_dim, self.act_dim, tuple(actor_hidden), "elu"
        ).to(self.device)
        if self.centralized:
            self.critic = mlp(self.state_dim, tuple(critic_hidden), self.num_agents, "elu").to(self.device)
        else:
            self.critic = mlp(self.obs_dim, tuple(critic_hidden), 1, "elu").to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr or lr
        )
        self.lr = float(lr)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b, n = obs.shape[0], obs.shape[1]
        action = self.actor.act(obs.reshape(b * n, -1), deterministic)
        return action.reshape(b, n, self.act_dim)

    def value_of(self, state: torch.Tensor, obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-agent value, ``(B, A)``."""
        if self.centralized:
            return self.critic(state)
        if obs is None:
            raise ValueError(
                f"{self.name}: an independent critic needs the observations, "
                "call value_of(state, obs)"
            )
        b, n = obs.shape[0], obs.shape[1]
        return self.critic(obs.reshape(b * n, -1)).reshape(b, n)

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor, state: torch.Tensor):
        # normalise exactly like ``update`` does, otherwise the clipped ratio is
        # computed on a different input distribution than the one we acted on
        b, n = obs.shape[0], obs.shape[1]
        o = self.norm_obs_t(obs)
        s = self.norm_state_t(state)
        dist = self.actor.distribution(o.reshape(b * n, -1))
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1).reshape(b, n)
        value = self.value_of(s, o)
        return action.reshape(b, n, self.act_dim), logp, value

    # ------------------------------------------------------------------ #
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        total = buffer.steps * self.num_envs
        a_dim, o_dim = self.num_agents, self.obs_dim
        obs_all = self.norm_obs_t(buffer.obs).reshape(total, a_dim, o_dim)
        state_all = self.norm_state_t(buffer.state).reshape(total, self.state_dim)
        act_all = buffer.action.reshape(total, a_dim, self.act_dim)
        logp_all = buffer.logp.reshape(total, a_dim)
        value_all = buffer.value.reshape(total, a_dim)
        ret = buffer.ret.reshape(total, a_dim)
        adv = buffer.adv.reshape(total, a_dim).clone()
        adv = (adv - adv.mean(0, keepdim=True)) / (adv.std(0, keepdim=True) + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        nb = 0
        for _ in range(self.num_epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, self.minibatch_size):
                idx = perm[start: start + self.minibatch_size]
                if idx.numel() < 2:
                    continue
                o, s, a = obs_all[idx], state_all[idx], act_all[idx]
                b = idx.numel()
                dist = self.actor.distribution(o.reshape(b * a_dim, -1))
                logp = dist.log_prob(a.reshape(b * a_dim, self.act_dim)).sum(-1).reshape(b, a_dim)
                entropy = dist.entropy().sum(-1).reshape(b, a_dim)
                old_logp, old_value = logp_all[idx], value_all[idx]
                adv_mb, ret_mb = adv[idx], ret[idx]

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.value_of(s, o)
                if self.value_clip > 0:
                    clipped = old_value + torch.clamp(
                        values - old_value, -self.value_clip, self.value_clip
                    )
                    value_loss = torch.max((values - ret_mb) ** 2, (clipped - ret_mb) ** 2).mean()
                else:
                    value_loss = ((values - ret_mb) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                with torch.no_grad():
                    kl = ((ratio - 1.0) - (logp - old_logp)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy.mean())
                    stats["approx_kl"] += float(kl)
                    stats["clip_frac"] += float(clip_frac)
                nb += 1

        return {k: v / max(nb, 1) for k, v in stats.items()}

    # ------------------------------------------------------------------ #
    def extra_state(self) -> Dict:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

    def load_extra_state(self, state: Dict) -> None:
        if "actor_optim" in state:
            self.actor_optim.load_state_dict(state["actor_optim"])
        if "critic_optim" in state:
            self.critic_optim.load_state_dict(state["critic_optim"])


# --------------------------------------------------------------------------- #
# Off-policy family
# --------------------------------------------------------------------------- #
class MARLOffPolicy(OffPolicyAgent):
    """Off-policy multi-agent actor-critic with a centralised critic."""

    multi_agent = True

    # ------------------------------------------------------------------ #
    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b, n = obs.shape[0], obs.shape[1]
        action = self._agent_act(obs.reshape(b * n, -1), deterministic)
        return action.reshape(b, n, self.act_dim)

    @abc.abstractmethod
    def _agent_act(self, flat_obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Action for a flattened ``(B * A, obs_dim)`` batch of observations."""

    # ------------------------------------------------------------------ #
    def q_values(self, critic: nn.Module, state: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        """``(B, A)`` Q values of the joint action."""
        return critic(state, joint_action(action))

    def q_min(self, critic: nn.Module, state: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        """Clipped double-Q of the joint action, ``(B, A)``."""
        return critic.q(state, joint_action(action))

    def batch_actions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["action"]
