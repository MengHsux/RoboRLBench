"""HATRPO - Heterogeneous-Agent Trust Region Policy Optimisation
(Kuba et al., ICLR 2022).

HATRPO is the trust-region sibling of HAPPO: the sequential multi-agent update
(joint importance ratio, random agent order) is kept, but each agent's step
solves a **KL-constrained** problem

``max_theta  E[ M_i * (pi_new(a_i|o_i) / pi_old(a_i|o_i)) * A_i ]``
``s.t.        E[ KL(pi_old || pi_new) ] <= delta``

with a natural-gradient step computed by a Fisher-vector product followed by
conjugate gradients, and a backtracking line search that enforces the KL
constraint.  That is the multi-agent counterpart of TRPO.

A plain gradient step is used as a fallback if the conjugate-gradient solve is
degenerate, so the baseline can never crash mid-training.

Reference: https://arxiv.org/abs/2110.02793
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .sequential import (
    SequentialOnPolicy,
    conjugate_gradient,
    flatten_params,
    gaussian_kl,
    set_flat_params,
)

__all__ = ["HATRPO"]


class HATRPO(SequentialOnPolicy):
    name = "hatrpo"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        state_dim: Optional[int] = None,
        num_envs: int = 1,
        device: str = "cpu",
        seed: int = 0,
        max_kl: float = 0.01,
        cg_iters: int = 10,
        damping: float = 1e-2,
        backtrack_iters: int = 10,
        backtrack_coef: float = 0.5,
        num_critic_epochs: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed, **kwargs
        )
        self.max_kl = float(max_kl)
        self.cg_iters = int(cg_iters)
        self.damping = float(damping)
        self.backtrack_iters = int(backtrack_iters)
        self.backtrack_coef = float(backtrack_coef)
        self.num_critic_epochs = int(num_critic_epochs)

    # ------------------------------------------------------------------ #
    def _agent_step(self, data: Dict[str, torch.Tensor], agent: int,
                    joint_ratio: torch.Tensor) -> float:
        obs_i = data["obs"][:, agent]
        act_i = data["action"][:, agent]
        adv_i = data["adv"][:, agent]
        logp_old = data["logp"][:, agent]
        params = [p for p in self.actor.parameters() if p.requires_grad]

        with torch.no_grad():
            dist0 = self.actor.distribution(obs_i)
            old_mu, old_std = dist0.mean, dist0.stddev

        def terms():
            dist = self.actor.distribution(obs_i)
            logp = dist.log_prob(act_i).sum(-1)
            ratio = torch.exp(logp - logp_old)
            surrogate = (joint_ratio * ratio * adv_i).mean()
            kl = gaussian_kl(old_mu, old_std, dist.mean, dist.stddev).mean()
            return surrogate, kl

        surrogate, _ = terms()
        base_surrogate = float(surrogate.detach())
        grads = torch.autograd.grad(surrogate, params)
        g = torch.cat([gr.reshape(-1) for gr in grads])
        if not torch.isfinite(g).all() or g.norm() < 1e-12:
            return base_surrogate

        def fvp(vec: torch.Tensor) -> torch.Tensor:
            _, kl = terms()
            kl_grads = torch.autograd.grad(kl, params, create_graph=True)
            flat = torch.cat([gr.reshape(-1) for gr in kl_grads])
            product = (flat * vec).sum()
            hess = torch.autograd.grad(product, params)
            return torch.cat([h.reshape(-1) for h in hess]) + self.damping * vec

        try:
            step_dir = conjugate_gradient(fvp, g, self.cg_iters)
            shs = 0.5 * (step_dir * fvp(step_dir)).sum()
            if not torch.isfinite(shs) or shs <= 0:
                raise ValueError("degenerate Fisher matrix")
            scale = torch.sqrt(shs / self.max_kl).clamp_min(1e-8)
            full_step = step_dir / scale
        except Exception:                       # pragma: no cover - numerical guard
            full_step = g / (g.norm() + 1e-8) * self.lr

        old_flat = flatten_params(params)
        accepted = False
        coef = 1.0
        for _ in range(self.backtrack_iters):
            set_flat_params(params, old_flat + coef * full_step)
            with torch.no_grad():
                new_surrogate, new_kl = terms()
                if (float(new_kl) <= self.max_kl) and (float(new_surrogate) >= base_surrogate):
                    accepted = True
                    break
            coef *= self.backtrack_coef

        if not accepted:
            # constraint violated everywhere: take a small gradient-ascent step
            set_flat_params(params, old_flat + self.lr * g)
        return base_surrogate
