"""MAT - Multi-Agent Transformer (Wen et al., ICML 2022).

MAT casts cooperative MARL as a **sequence modelling** problem: the ``A``
observations form a sequence that a transformer encoder maps to a joint
representation, and a causal transformer decoder emits the joint action
autoregressively,

``pi(a_1..a_A | o_1..o_A) = prod_i pi(a_i | o_1..o_A, a_1..a_{i-1})``.

Because the factorisation is sequential, the policy gradient of the whole team
uses a *single* joint importance ratio - no per-agent ratio, no joint-ratio
product - which is what makes MAT so sample efficient in the paper.  Training is
PPO on that joint ratio, with a centralised value function that outputs one
value per agent.

Reference: https://arxiv.org/abs/2205.14953
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..base import OnPolicyAgent
from ..buffers import RolloutBuffer
from ..nets import mlp
from .nets_marl import MultiAgentTransformer

__all__ = ["MAT"]


class MAT(OnPolicyAgent):
    name = "mat"
    multi_agent = True

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        critic_hidden=(256, 128),
        lr: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.0,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )
        self.clip_range = float(clip_range)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.transformer = MultiAgentTransformer(
            self.obs_dim,
            self.act_dim,
            self.num_agents,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.critic = mlp(
            self.state_dim, tuple(critic_hidden), self.num_agents, "elu"
        ).to(self.device)

        self.actor_optim = torch.optim.Adam(self.transformer.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {"actor": self.transformer, "critic": self.critic}

    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        action, _ = self.transformer.sample(obs, deterministic)
        return action

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor, state: torch.Tensor):
        # MAT normalises the rollout in ``update``; act on the same inputs
        action, logp = self.transformer.sample(self.norm_obs_t(obs))
        value = self.critic(self.norm_state_t(state))
        return action, logp, value

    @torch.no_grad()
    def value_of(self, state: torch.Tensor, obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.critic(state)

    # ------------------------------------------------------------------ #
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        total = buffer.steps * self.num_envs
        a_dim, o_dim = self.num_agents, self.obs_dim
        obs = self.norm_obs_t(buffer.obs).reshape(total, a_dim, o_dim)
        state = self.norm_state_t(buffer.state).reshape(total, self.state_dim)
        act = buffer.action.reshape(total, a_dim, self.act_dim)
        logp_old = buffer.logp.reshape(total, a_dim).sum(-1)
        ret = buffer.ret.reshape(total, a_dim)

        adv = buffer.adv.reshape(total, a_dim).sum(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        nb = 0
        for _ in range(self.num_epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, self.minibatch_size):
                idx = perm[start: start + self.minibatch_size]
                if idx.numel() < 2:
                    continue
                o, s, a = obs[idx], state[idx], act[idx]
                logp, entropy = self.transformer.evaluate(o, a)
                logp = logp.sum(-1)

                ratio = torch.exp(logp - logp_old[idx])
                adv_mb = adv[idx]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(s)
                value_loss = ((values - ret[idx]) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                with torch.no_grad():
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy.mean())
                    stats["approx_kl"] += float(((ratio - 1.0) - (logp - logp_old[idx])).mean())
                    stats["clip_frac"] += float(((ratio - 1.0).abs() > self.clip_range).float().mean())
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
