"""Proximal Policy Optimisation (Schulman et al., 2017).

The reference on-policy baseline for every robot-learning benchmark.  The
implementation follows the ``rsl_rl`` recipe that the legged-robot papers use -
Gaussian policy with a *state independent* learnable ``log_std``, generalised
advantage estimation, clipped surrogate objective, value clipping, gradient
clipping and an optional adaptive learning rate driven by the measured KL.

Reference: https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..base import OnPolicyAgent
from ..buffers import RolloutBuffer
from ..nets import GaussianActor, mlp

__all__ = ["PPO"]


class PPO(OnPolicyAgent):
    name = "ppo"

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
        clip_range: float = 0.2,
        value_clip: float = 0.0,
        entropy_coef: float = 0.0,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        target_kl: float = 0.0,
        desired_kl: float = 0.01,
        adaptive_lr: bool = True,
        rollout_steps: int = 24,
        num_minibatches: int = 4,
        num_epochs: int = 5,
        gamma: float = 0.99,
        lam: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim, act_dim, num_agents, state_dim, num_envs, device, seed,
            rollout_steps=rollout_steps, gamma=gamma, lam=lam,
            num_minibatches=num_minibatches, num_epochs=num_epochs, **kwargs,
        )
        self.clip_range = float(clip_range)
        self.value_clip = float(value_clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.desired_kl = float(desired_kl)
        self.adaptive_lr = bool(adaptive_lr)

        self.actor = GaussianActor(self.obs_dim, self.act_dim, tuple(actor_hidden)).to(self.device)
        # the critic predicts one value per agent from the centralised state
        self.critic = mlp(self.state_dim, tuple(critic_hidden), self.num_agents, "elu").to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.lr = float(lr)

    # ------------------------------------------------------------------ #
    def modules(self) -> Dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b = obs.shape[0]
        action = self.actor.act(obs.reshape(b, -1), deterministic)
        return action.reshape(b, self.num_agents, self.act_dim)

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor, state: torch.Tensor):
        # ``update`` normalises the rollout with the same running statistics, so
        # the log-prob ratio is always computed on the inputs we acted on.
        b = obs.shape[0]
        dist = self.actor.distribution(self.norm_obs_t(obs).reshape(b, -1))
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(self.norm_state_t(state))
        return (
            action.reshape(b, self.num_agents, self.act_dim),
            logp.unsqueeze(-1).expand(b, self.num_agents),
            value,
        )

    @torch.no_grad()
    def value_of(self, state: torch.Tensor, obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """State value ``(B, num_agents)`` used for bootstrapping."""
        return self.critic(state)

    # ------------------------------------------------------------------ #
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        obs_all = self.norm_obs_t(buffer.obs)
        state_all = self.norm_state_t(buffer.state)
        action_all = buffer.action.reshape(buffer.steps, self.num_envs, -1)
        old_logp = buffer.logp
        old_value = buffer.value
        adv = buffer.adv.clone()
        # normalise the advantage per agent (zero mean / unit std)
        adv = (adv - adv.mean(dim=(0, 1), keepdim=True)) / (adv.std(dim=(0, 1), keepdim=True) + 1e-8)
        ret = buffer.ret

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        nb = 0
        epoch_kl = []
        total = buffer.steps * self.num_envs
        for epoch in range(self.num_epochs):
            perm = torch.randperm(total, device=self.device, generator=self._gen)
            kl_epoch, nb_epoch = 0.0, 0
            for start in range(0, total, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                if idx.numel() < 2:
                    continue
                flat_obs = obs_all.reshape(total, self.num_agents, self.obs_dim)[idx]
                flat_state = state_all.reshape(total, self.state_dim)[idx]
                flat_action = action_all.reshape(total, self.num_agents, self.act_dim)[idx]
                flat_logp = old_logp.reshape(total, self.num_agents)[idx]
                flat_adv = adv.reshape(total, self.num_agents)[idx]
                flat_ret = ret.reshape(total, self.num_agents)[idx]
                flat_value = old_value.reshape(total, self.num_agents)[idx]

                b = flat_obs.shape[0]
                dist = self.actor.distribution(flat_obs.reshape(b, -1))
                a = flat_action.reshape(b, self.num_agents, self.act_dim)
                logp = dist.log_prob(a).sum(-1)                  # (b, A)
                entropy = dist.entropy().sum(-1).unsqueeze(-1).expand_as(logp)
                ratio = torch.exp(logp - flat_logp)

                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * flat_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(flat_state)
                if self.value_clip > 0:
                    clipped = flat_value + torch.clamp(
                        values - flat_value, -self.value_clip, self.value_clip
                    )
                    value_loss = torch.max(
                        (values - flat_ret) ** 2, (clipped - flat_ret) ** 2
                    ).mean()
                else:
                    value_loss = ((values - flat_ret) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                with torch.no_grad():
                    kl = ((ratio - 1.0) - (logp - flat_logp)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy.mean())
                    stats["approx_kl"] += float(kl)
                    stats["clip_frac"] += float(clip_frac)
                    kl_epoch += float(kl)
                    nb_epoch += 1
                nb += 1

            if nb_epoch:
                epoch_kl.append(kl_epoch / nb_epoch)
                # stop once the policy has moved far enough from the rollout
                if self.target_kl > 0 and epoch_kl[-1] > self.target_kl:
                    break

        out = {k: v / max(nb, 1) for k, v in stats.items()}
        # the schedule must look at the *last* epoch: the first one always has
        # ratio == 1 and would dilute the measured KL towards zero
        out["lr"] = self.lr
        self._maybe_adjust_lr(epoch_kl[-1] if epoch_kl else 0.0)
        return out

    # ------------------------------------------------------------------ #
    def _maybe_adjust_lr(self, kl: float) -> None:
        if not self.adaptive_lr or kl <= 0:
            return
        if kl > 2.0 * self.desired_kl:
            self.lr = max(self.lr / 1.5, 1e-5)
        elif kl < 0.5 * self.desired_kl:
            self.lr = min(self.lr * 1.5, 1e-2)
        else:
            return
        for opt in (self.actor_optim, self.critic_optim):
            for group in opt.param_groups:
                group["lr"] = self.lr

    # ------------------------------------------------------------------ #
    def extra_state(self) -> Dict:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "lr": self.lr,
        }

    def load_extra_state(self, state: Dict) -> None:
        if "actor_optim" in state:
            self.actor_optim.load_state_dict(state["actor_optim"])
        if "critic_optim" in state:
            self.critic_optim.load_state_dict(state["critic_optim"])
        self.lr = float(state.get("lr", self.lr))
        for opt in (self.actor_optim, self.critic_optim):
            for group in opt.param_groups:
                group["lr"] = self.lr
