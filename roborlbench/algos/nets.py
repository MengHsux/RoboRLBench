"""Shared neural-network building blocks.

Every algorithm in the benchmark is written with these primitives so that the
*only* differences between two baselines are the algorithmic ones.

Conventions
-----------
* Actors output actions in ``[-1, 1]``.  Environment adapters map that box onto
  the robot's physical action range, which is what lets the same network drive a
  Go2 (12 joints), an SO-101 (6 joints) or a 2-D point mass (2 actuators).
* Critics take the *centralised* state (single robot: the observation) and,
  depending on the algorithm, the joint action of all agents.
* Every module is shape-agnostic in the agent axis: they operate on
  ``(..., obs_dim)`` and never assume a particular batch layout.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

__all__ = [
    "mlp",
    "activation",
    "DeterministicActor",
    "GaussianActor",
    "SquashedGaussianActor",
    "Critic",
    "TwinCritic",
    "EnsembleCritic",
    "QuantileCritic",
    "SimbaMLP",
    "SimbaSquashedGaussianActor",
    "SimbaTwinCritic",
    "SimbaDeterministicActor",
    "LayerNormMLP",
    "BatchRenorm1d",
    "logprob_from_normal",
    "tanh_logprob_correction",
    "get_activation",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def get_activation(name: str) -> nn.Module:
    table = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in table:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(table)}")
    return table[name]()


def activation(name: str) -> nn.Module:  # backwards compatible alias
    return get_activation(name)


def _as_tuple(layer_sizes: Iterable[int]) -> Sequence[int]:
    sizes = tuple(int(s) for s in layer_sizes)
    if len(sizes) < 1:
        raise ValueError("layer_sizes must contain at least the input size")
    return sizes


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: str = "silu",
    *,
    output_activation: Optional[str] = None,
    use_layer_norm: bool = False,
    dropout: float = 0.0,
    init: str = "orthogonal",
) -> nn.Sequential:
    """A plain MLP ``in -> hidden... -> out``."""
    layers = []
    last = in_dim
    for width in hidden:
        layers.append(nn.Linear(last, int(width)))
        if use_layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(get_activation(act))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = int(width)
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    if init == "orthogonal":
        orthogonal_init(net)
    elif init == "xavier":
        xavier_init(net)
    return net


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def xavier_init(module: nn.Module, gain: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# SimBa / LayerNorm blocks
# --------------------------------------------------------------------------- #
class SimbaBlock(nn.Module):
    """Residual block from *SimBa* (Lee et al., NeurIPS 2024).

    ``x + Linear(SiLU(LayerNorm(Linear(SiLU(x)))))`` - the point of the paper is
    that this minimal architecture scales with network width and depth much
    better than the usual 2-layer ReLU MLP.
    """

    def __init__(self, dim: int, expansion: int = 2, act: str = "silu") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, expansion * dim)
        self.fc2 = nn.Linear(expansion * dim, dim)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.fc1(self.act(h))
        h = self.fc2(self.act(h))
        return x + h


class SimbaMLP(nn.Module):
    """Encoder used by the SimBa baseline: ``Linear -> [SimbaBlock] * n``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        out_dim: Optional[int] = None,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[SimbaBlock(hidden_dim, act=act) for _ in range(num_blocks)])
        self.out = nn.Linear(hidden_dim, out_dim) if out_dim is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.blocks(self.inp(x)))


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm on every hidden layer (used by DroQ / CrossQ)."""

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = mlp(
            in_dim, hidden, out_dim, act,
            use_layer_norm=True, dropout=dropout, init="none",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Actors
# --------------------------------------------------------------------------- #
class DeterministicActor(nn.Module):
    """``tanh`` actor used by DDPG / TD3 / MATD3 / FACMAC and friends."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        act: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = mlp(
            obs_dim, hidden, act_dim, act,
            use_layer_norm=use_layer_norm, output_activation=None,
        )
        # small final-layer gain keeps the initial policy close to zero
        final = [m for m in self.trunk if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(final.weight, 0.01)
        nn.init.zeros_(final.bias)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.trunk(obs))


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with a state-independent ``log_std``.

    This is the PPO actor used by ``rsl_rl`` and by every legged-robot
    locomotion paper; the learnable ``log_std`` vector is shared across the
    batch and initialised to zero (``std = 1``).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 128),
        act: str = "elu",
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean = mlp(obs_dim, hidden, act_dim, act)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        final = [m for m in self.mean if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(final.weight, 0.01)
        nn.init.zeros_(final.bias)

    def distribution(self, obs: Tensor) -> Normal:
        mu = self.mean(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        dist = self.distribution(obs)
        return dist.mean if deterministic else dist.sample()


class SquashedGaussianActor(nn.Module):
    """SAC-style ``tanh(N(mu, sigma))`` policy.

    ``forward`` returns ``(action, log_prob)`` where ``log_prob`` already
    contains the change-of-variables correction for the ``tanh`` squashing.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        act: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = mlp(
            obs_dim, hidden, 2 * act_dim, act, use_layer_norm=use_layer_norm
        )
        self.act_dim = act_dim
        final = [m for m in self.trunk if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(final.weight, 0.01)
        nn.init.zeros_(final.bias)

    def distribution(self, obs: Tensor) -> tuple[Normal, Tensor]:
        h = self.trunk(obs)
        mu, log_std = h[..., : self.act_dim], h[..., self.act_dim:]
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp()), mu

    def rsample(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reparameterised sample -> ``(action, log_prob, deterministic_action)``."""
        dist, mu = self.distribution(obs)
        pre = dist.rsample()
        action = torch.tanh(pre)
        # change of variables, elementwise over the action axis, then summed
        log_prob = dist.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1), torch.tanh(mu)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        action, log_prob, _ = self.rsample(obs)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        _, mu = self.distribution(obs)
        if deterministic:
            return torch.tanh(mu)
        dist, _ = self.distribution(obs)
        return torch.tanh(dist.sample())


# --------------------------------------------------------------------------- #
# Critics
# --------------------------------------------------------------------------- #
class Critic(nn.Module):
    """``Q(s, a)`` MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (256, 256),
        act: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = mlp(
            in_dim, hidden, 1, act,
            use_layer_norm=use_layer_norm, dropout=dropout,
            init="none",
        )
        for m in reversed(list(self.net)):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
                break

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent critics (clipped double-Q)."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 256), act: str = "relu", **kw) -> None:
        super().__init__()
        self.q1 = Critic(in_dim, hidden, act, **kw)
        self.q2 = Critic(in_dim, hidden, act, **kw)

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(s, a), self.q2(s, a)

    def q(self, s: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)

    def both(self, s: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(s, a)
        return torch.stack([q1, q2], dim=0)


class EnsembleCritic(nn.Module):
    """``N`` independent critics, stacked along a leading dimension.

    Used by REDQ / DroQ / FastTD3 where the target is the minimum over a random
    subset of the ensemble.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (256, 256),
        num_critics: int = 5,
        act: str = "relu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_critics = int(num_critics)
        self.members = nn.ModuleList(
            [Critic(in_dim, hidden, act, use_layer_norm=use_layer_norm, dropout=dropout)
             for _ in range(self.num_critics)]
        )

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        return torch.stack([m(s, a) for m in self.members], dim=0)   # (N_q, B)

    def q_mean(self, s: Tensor, a: Tensor) -> Tensor:
        return self.forward(s, a).mean(0)


class QuantileCritic(nn.Module):
    """TQC critic: every member predicts ``num_quantiles`` quantiles."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 256),
                 num_critics: int = 5, num_quantiles: int = 25, act: str = "relu") -> None:
        super().__init__()
        self.num_critics = int(num_critics)
        self.num_quantiles = int(num_quantiles)
        # each member outputs ``num_quantiles`` values for a single (s, a)
        self.members = nn.ModuleList(
            [mlp(in_dim, hidden, self.num_quantiles, act, init="none")
             for _ in range(self.num_critics)]
        )

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([s, a], dim=-1)
        return torch.stack([m(x) for m in self.members], dim=0)      # (N_q, B, n_tau)


# --------------------------------------------------------------------------- #
# Probability helpers
# --------------------------------------------------------------------------- #
def tanh_logprob_correction(pre: Tensor, eps: float = 1e-6) -> Tensor:
    """``log(1 - tanh(x)^2 + eps)`` summed over the last axis."""
    return torch.log(1.0 - torch.tanh(pre).pow(2) + eps).sum(-1)


def logprob_from_normal(mu: Tensor, std: Tensor, action: Tensor, squashed: bool = False) -> Tensor:
    """Log-density of ``action`` under ``N(mu, std)``, summed over the last axis."""
    dist = Normal(mu, std)
    if squashed:
        action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre = torch.atanh(action)
        # tanh(atanh(a)) == a, so the correction is exactly log(1 - a^2)
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (dist.log_prob(pre) - correction).sum(-1)
    return dist.log_prob(action).sum(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


# --------------------------------------------------------------------------- #
# BatchRenormalisation (CrossQ)
# --------------------------------------------------------------------------- #
class BatchRenorm1d(nn.Module):
    """Batch renormalisation layer used by *CrossQ* (Bhatt et al., ICLR 2024).

    CrossQ removes the target networks and relies on BatchRenorm to keep the
    critic's input statistics stable while the replay buffer distribution
    drifts.  In ``eval`` mode it falls back to the running statistics, exactly
    like :class:`torch.nn.BatchNorm1d`.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.01,
        eps: float = 1e-5,
        rmax: float = 1.0,
        dmax: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.rmax = float(rmax)
        self.dmax = float(dmax)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x3 = x.unsqueeze(-1)          # (B, C, 1)
            squeeze = True
        else:
            x3 = x
            squeeze = False
        out = self._renorm(x3)
        return out.squeeze(-1) if squeeze else out

    def _renorm(self, x: Tensor) -> Tensor:
        if self.training:
            dims = [d for d in range(x.dim()) if d != 1]
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            # stop-gradient warm-up of the renormalisation coefficients
            with torch.no_grad():
                r = (var.sqrt() / self.running_var.sqrt().clamp_min(self.eps)).clamp(
                    1.0 / self.rmax, self.rmax
                )
                d = ((mean - self.running_mean) / self.running_var.sqrt().clamp_min(self.eps)).clamp(
                    -self.dmax, self.dmax
                )
            shape = [1] * x.dim()
            shape[1] = -1
            r, d = r.reshape(shape), d.reshape(shape)
            out = (x - mean.reshape(shape)) / var.clamp_min(self.eps).sqrt().reshape(shape) * r + d
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
                self.step += 1
        else:
            shape = [1] * x.dim()
            shape[1] = -1
            out = (x - self.running_mean.reshape(shape)) / self.running_var.clamp_min(self.eps).sqrt().reshape(shape)
        shape = [1] * x.dim()
        shape[1] = -1
        return out * self.weight.reshape(shape) + self.bias.reshape(shape)


# --------------------------------------------------------------------------- #
# SimBa variants (Lee et al., NeurIPS 2024)
# --------------------------------------------------------------------------- #
class SimbaSquashedGaussianActor(nn.Module):
    """Same contract as :class:`SquashedGaussianActor` but with SimBa blocks."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.act_dim = int(act_dim)
        self.trunk = SimbaMLP(obs_dim, hidden_dim, num_blocks, 2 * act_dim)
        final = self.trunk.out
        nn.init.orthogonal_(final.weight, 0.01)
        nn.init.zeros_(final.bias)

    def distribution(self, obs: Tensor) -> tuple[Normal, Tensor]:
        h = self.trunk(obs)
        mu, log_std = h[..., : self.act_dim], h[..., self.act_dim:]
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp()), mu

    def rsample(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dist, mu = self.distribution(obs)
        pre = dist.rsample()
        action = torch.tanh(pre)
        # change of variables, elementwise over the action axis, then summed
        log_prob = dist.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1), torch.tanh(mu)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        action, log_prob, _ = self.rsample(obs)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        dist, mu = self.distribution(obs)
        return torch.tanh(mu) if deterministic else torch.tanh(dist.sample())


class SimbaTwinCritic(nn.Module):
    """Two SimBa-encoded critics with the :class:`TwinCritic` interface."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_blocks: int = 2) -> None:
        super().__init__()
        self.q1 = SimbaMLP(in_dim, hidden_dim, num_blocks, 1)
        self.q2 = SimbaMLP(in_dim, hidden_dim, num_blocks, 1)
        for net in (self.q1, self.q2):
            nn.init.orthogonal_(net.out.weight, 1.0)
            nn.init.zeros_(net.out.bias)

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q(self, s: Tensor, a: Tensor) -> Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class SimbaDeterministicActor(nn.Module):
    """``tanh`` actor with SimBa blocks (used by TD7 / FastTD3)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.trunk = SimbaMLP(obs_dim, hidden_dim, num_blocks, act_dim)
        nn.init.orthogonal_(self.trunk.out.weight, 0.01)
        nn.init.zeros_(self.trunk.out.bias)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.trunk(obs))
