"""TQC - Controlling Overestimation Bias with Truncated Quantile Critics
(Kuznetsov et al., ICML 2020).

TQC replaces the two scalar critics of SAC with an ensemble of critics that
each predict a *distribution* over returns through ``N_tau`` quantiles.  Two
mechanisms give the algorithm its name:

* **Truncated quantiles** - the Bellman target is built from the
  ``N_tau - drop`` smallest quantiles of the target ensemble, which removes the
  most optimistic estimates and therefore controls overestimation bias.
* **Quantile Huber loss** - each predicted quantile is regressed onto the target
  distribution with the asymmetric quantile (pinball) weighting.

Everything else (squashed Gaussian actor, entropy temperature, soft target
updates) is inherited from :class:`~roborlbench.algos.single.sac.SAC`.

Reference: https://arxiv.org/abs/2005.01578
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..nets import QuantileCritic
from .sac import SAC

__all__ = ["TQC", "quantile_huber_loss"]


def quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss used by TQC / IQN.

    Parameters
    ----------
    pred:
        ``(N_q, B, N_tau)`` predicted quantiles.
    target:
        ``(B, N_keep)`` truncated target quantiles.
    taus:
        ``(N_tau,)`` quantile levels in ``(0, 1)``.
    """
    u = target.unsqueeze(0).unsqueeze(2) - pred.unsqueeze(-1)    # (N_q, B, N_tau, N_keep)
    huber = torch.where(
        u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa)
    )
    weight = (taus.view(1, 1, -1, 1) - (u < 0).to(pred.dtype)).abs()
    return (weight * huber).mean() / kappa


class TQC(SAC):
    name = "tqc"

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
        num_critics: int = 5,
        num_quantiles: int = 25,
        top_quantiles_to_drop: int = 2,
        kappa: float = 1.0,
        grad_steps: int = 1,
        **kwargs,
    ) -> None:
        # build_actor / build_critic are called from SAC.__init__, so the
        # quantile configuration has to be available before super().__init__.
        self.num_critics = int(num_critics)
        self.num_quantiles = int(num_quantiles)
        self.top_quantiles_to_drop = int(top_quantiles_to_drop)
        self.kappa = float(kappa)
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            hidden=hidden, grad_steps=grad_steps, **kwargs,
        )
        self.taus = (
            torch.arange(self.num_quantiles, device=self.device, dtype=torch.float32) + 0.5
        ) / self.num_quantiles
        assert 0 <= self.top_quantiles_to_drop < self.num_quantiles, (
            "top_quantiles_to_drop must be smaller than num_quantiles"
        )

    # ------------------------------------------------------------------ #
    def build_critic(self) -> nn.Module:
        return QuantileCritic(
            self.state_dim + self.act_dim,
            self.hidden,
            num_critics=self.num_critics,
            num_quantiles=self.num_quantiles,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.rsample(batch["next_obs"].reshape(b, -1))
            next_q = self.critic_target(batch["next_state"], next_action)   # (N_q, B, N_tau)
            next_q = next_q.permute(1, 0, 2).reshape(b, -1)                 # (B, N_q*N_tau)
            keep = self.num_quantiles - self.top_quantiles_to_drop
            target_q = next_q.sort(dim=1).values[:, :keep]                  # (B, N_keep)
            target = batch["reward"].reshape(-1, 1) + self.gamma * batch["discount"].reshape(-1, 1) * (
                target_q - self.alpha.detach() * next_logp.unsqueeze(-1)
            )

        pred = self.critic(batch["state"], batch["action"].reshape(b, -1))  # (N_q, B, N_tau)
        critic_loss = torch.zeros((), device=self.device)
        for i in range(pred.shape[0]):
            critic_loss = critic_loss + quantile_huber_loss(pred[i], target, self.taus, self.kappa)
        critic_loss = critic_loss / float(pred.shape[0])

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        self._soft_update_targets()
        return {"critic_loss": float(critic_loss.detach())}

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = batch["state"].shape[0]
        action, logp, _ = self.actor.rsample(batch["obs"].reshape(b, -1))
        # the actor maximises the mean over all quantiles of all critics
        q = self.critic(batch["state"], action).mean(dim=(0, 2))             # (B,)
        actor_loss = (self.alpha.detach() * logp - q).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        out = {"actor_loss": float(actor_loss.detach()), "alpha": float(self.alpha.detach())}
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            out["alpha_loss"] = float(alpha_loss.detach())
        return out
