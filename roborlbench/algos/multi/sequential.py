"""Scaffolding shared by HAPPO and HATRPO.

Both algorithms are *sequential* policy-gradient methods: at every iteration the
agents are updated one after the other in a random order and each update is
weighted by the joint importance ratio

``M_i = prod_{k < i} pi_k^{new}(a_k | o_k) / pi_k^{old}(a_k | o_k)``

which makes the monotonic-improvement guarantee of the multi-agent policy
gradient theorem hold.  The only difference between the two is the *trust
region* used for the single-agent step:

* **HAPPO** - PPO's clipped surrogate,
* **HATRPO** - a KL-constrained natural gradient with a backtracking line
  search (Fisher-vector product + conjugate gradients).

The critic is a standard centralised value function trained with a few epochs of
mean-squared error on the rollout returns.
"""

from __future__ import annotations

import abc
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..buffers import RolloutBuffer
from .base import MARLOnPolicy

__all__ = [
    "SequentialOnPolicy",
    "flatten_params",
    "set_flat_params",
    "conjugate_gradient",
    "gaussian_kl",
]


def gaussian_kl(mu1: torch.Tensor, std1: torch.Tensor,
                mu2: torch.Tensor, std2: torch.Tensor) -> torch.Tensor:
    """Element-wise ``KL(N(mu1, std1) || N(mu2, std2))`` summed over the last axis."""
    var1, var2 = std1.pow(2), std2.pow(2)
    kl = (var1 / var2).log().mul(-0.5) + (var1 + (mu1 - mu2).pow(2)) / (2.0 * var2) - 0.5
    return kl.sum(-1)


def flatten_params(params) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in params])


@torch.no_grad()
def set_flat_params(params, flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].reshape_as(p))
        offset += n


def conjugate_gradient(fvp, b: torch.Tensor, iters: int = 10, tol: float = 1e-10) -> torch.Tensor:
    """Solve ``A x = b`` with the conjugate gradient method (no matrix storage)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r.dot(r)
    for _ in range(iters):
        ap = fvp(p)
        denom = p.dot(ap)
        if denom <= 0:
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = r.dot(r)
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


class SequentialOnPolicy(MARLOnPolicy):
    """Common data preparation and bookkeeping for HAPPO / HATRPO."""

    num_critic_epochs: int = 5

    # ------------------------------------------------------------------ #
    def _prepare(self, buffer: RolloutBuffer) -> Dict[str, torch.Tensor]:
        total = buffer.steps * self.num_envs
        n, d = self.num_agents, self.obs_dim
        data = {
            "obs": self.norm_obs_t(buffer.obs).reshape(total, n, d),
            "state": self.norm_state_t(buffer.state).reshape(total, self.state_dim),
            "action": buffer.action.reshape(total, n, self.act_dim),
            "logp": buffer.logp.reshape(total, n),
            "value": buffer.value.reshape(total, n),
            "ret": buffer.ret.reshape(total, n),
            "adv": buffer.adv.reshape(total, n).clone(),
        }
        adv = data["adv"]
        data["adv"] = (adv - adv.mean(0, keepdim=True)) / (adv.std(0, keepdim=True) + 1e-8)
        return data

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        data = self._prepare(buffer)
        n = self.num_agents
        total = data["obs"].shape[0]

        order = torch.randperm(n, device=self.device).tolist()
        joint_ratio = torch.ones(total, device=self.device)
        policy_loss = 0.0
        for i in order:
            policy_loss += self._agent_step(data, i, joint_ratio)
            joint_ratio = joint_ratio * self._agent_ratio(data, i)

        value_loss = self._update_value(data)
        with torch.no_grad():
            dist = self.actor.distribution(data["obs"].reshape(total * n, -1))
            entropy = float(dist.entropy().sum(-1).mean())
        return {
            "policy_loss": policy_loss / float(max(n, 1)),
            "value_loss": value_loss,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _agent_step(self, data: Dict[str, torch.Tensor], agent: int,
                    joint_ratio: torch.Tensor) -> float:
        """Perform the single-agent update and return its loss value."""

    @torch.no_grad()
    def _agent_ratio(self, data: Dict[str, torch.Tensor], agent: int) -> torch.Tensor:
        """``pi_new(a_i | o_i) / pi_old(a_i | o_i)`` for the stored action."""
        dist = self.actor.distribution(data["obs"][:, agent])
        logp = dist.log_prob(data["action"][:, agent]).sum(-1)
        return torch.exp(logp - data["logp"][:, agent])

    def _update_value(self, data: Dict[str, torch.Tensor]) -> float:
        total = data["obs"].shape[0]
        last = 0.0
        for _ in range(self.num_critic_epochs):
            values = self.value_of(data["state"], data["obs"])
            loss = ((values - data["ret"]) ** 2).mean()
            self.critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            last = float(loss.detach())
        return last
