"""Networks that are specific to the multi-agent baselines.

``CentralQ`` / ``TwinCentralQ``
    Centralised critics used by MADDPG, MATD3, MASAC and HASAC.  They map the
    global state and the **joint** action to one Q value *per agent*, i.e.
    ``Q_i(s, a_1, ..., a_N)`` with parameter sharing (the agent axis is an
    output dimension instead of a separate set of networks).

``FactoredTwinQ``
    FACMAC's factored critic: ``Q_i(s, a_i, phi_i(a_{-i}))`` where ``phi_i``
    aggregates the embeddings of the other agents' actions.  The aggregation is
    a (weighted) sum, which is the monotonic factorisation FACMAC uses to keep
    the joint greedy action tractable.

``MultiAgentTransformer``
    The sequence model of MAT (Multi-Agent Transformer): a transformer encoder
    over the per-agent observations and a causal transformer decoder that
    produces the joint action autoregressively.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..nets import mlp, orthogonal_init

__all__ = ["CentralQ", "TwinCentralQ", "FactoredTwinQ", "MultiAgentTransformer"]


# --------------------------------------------------------------------------- #
# Centralised critics
# --------------------------------------------------------------------------- #
class CentralQ(nn.Module):
    """``(state, joint_action) -> Q_i for every agent``."""

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        act: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.net = mlp(
            state_dim + num_agents * act_dim,
            hidden,
            num_agents,
            act,
            use_layer_norm=use_layer_norm,
            init="none",
        )
        final = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(final.weight, 1.0)
        nn.init.zeros_(final.bias)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, joint_action], dim=-1))


class TwinCentralQ(nn.Module):
    """Two :class:`CentralQ` critics with clipped double-Q access."""

    def __init__(self, state_dim: int, num_agents: int, act_dim: int, **kw) -> None:
        super().__init__()
        self.q1 = CentralQ(state_dim, num_agents, act_dim, **kw)
        self.q2 = CentralQ(state_dim, num_agents, act_dim, **kw)

    def forward(self, state, joint_action):
        return self.q1(state, joint_action), self.q2(state, joint_action)

    def q(self, state, joint_action):
        q1, q2 = self.forward(state, joint_action)
        return torch.min(q1, q2)


class _FactoredCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        num_agents: int,
        embed_dim: int = 32,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.act_embed = nn.Sequential(
            nn.Linear(act_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.agent_embed = nn.Embedding(num_agents, embed_dim)
        in_dim = state_dim + act_dim + embed_dim + embed_dim
        self.net = mlp(in_dim, hidden, 1, "relu", init="none")
        nn.init.orthogonal_(self.net[-1].weight, 1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        # accept both (B, A, act_dim) and the flattened (B, A * act_dim)
        if joint_action.dim() == 2:
            joint_action = joint_action.reshape(joint_action.shape[0], self.num_agents, -1)
        b, n, _ = joint_action.shape
        emb = self.act_embed(joint_action).reshape(b, n, -1)          # (B, A, E)
        outs = []
        for i in range(n):
            others = [emb[:, j] for j in range(n) if j != i]
            if others:
                phi = torch.stack(others, dim=1).mean(dim=1)          # monotone aggregation
            else:
                phi = torch.zeros(b, emb.shape[-1], device=state.device)
            x = torch.cat(
                [state, joint_action[:, i], phi, self.agent_embed.weight[i].expand(b, -1)],
                dim=-1,
            )
            outs.append(self.net(x).squeeze(-1))
        return torch.stack(outs, dim=-1)                              # (B, A)


class FactoredTwinQ(nn.Module):
    """Twin factored critics (FACMAC)."""

    def __init__(self, state_dim: int, num_agents: int, act_dim: int, **kw) -> None:
        super().__init__()
        self.q1 = _FactoredCritic(state_dim, act_dim, num_agents, **kw)
        self.q2 = _FactoredCritic(state_dim, act_dim, num_agents, **kw)

    def forward(self, state, joint_action):
        return self.q1(state, joint_action), self.q2(state, joint_action)

    def q(self, state, joint_action):
        q1, q2 = self.forward(state, joint_action)
        return torch.min(q1, q2)


# --------------------------------------------------------------------------- #
# Multi-Agent Transformer (MAT)
# --------------------------------------------------------------------------- #
class MultiAgentTransformer(nn.Module):
    """Encoder-decoder transformer that maps ``obs`` to a joint action.

    * The encoder consumes the ``A`` observations as a sequence, so every agent
      sees what its team-mates observe (this is the centralised training part).
    * The decoder is causal: the action of agent ``i`` is conditioned on the
      actions of agents ``0 .. i-1``.  This factorisation
      ``pi(a | o) = prod_i pi(a_i | o, a_{<i})`` is what makes the joint policy
      tractable and allows a single joint importance ratio.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_agents = int(num_agents)
        self.d_model = int(d_model)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.act_embed = nn.Linear(act_dim, d_model)
        # index 0 is the <start> token, index j is agent j-1
        self.agent_embed = nn.Parameter(torch.zeros(num_agents + 1, d_model))
        nn.init.normal_(self.agent_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # ``enable_nested_tensor=False`` keeps the fast path off (it is
        # incompatible with ``norm_first``) and silences a PyTorch warning
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.head = nn.Linear(d_model, 2 * act_dim)
        orthogonal_init(self.head, gain=0.01)

    # ------------------------------------------------------------------ #
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        b, n, _ = obs.shape
        x = self.obs_embed(obs) + self.agent_embed[1: n + 1].unsqueeze(0)
        return self.encoder(x)

    def _decode(self, memory: torch.Tensor, tgt: torch.Tensor):
        n = tgt.shape[1]
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=tgt.device), diagonal=1
        )
        h = self.decoder(tgt, memory, tgt_mask=mask)
        out = self.head(h)
        mu, log_std = out[..., : self.act_dim], out[..., self.act_dim:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def _tgt_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed ``[<start>, a_0, ..., a_{n-2}]`` (teacher forcing)."""
        b, n, _ = actions.shape
        start = self.agent_embed[0].view(1, 1, -1).expand(b, 1, -1)
        if n == 1:
            return start
        embedded = self.act_embed(actions[:, :-1]) + self.agent_embed[1:n].unsqueeze(0)
        return torch.cat([start, embedded], dim=1)

    # ------------------------------------------------------------------ #
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Teacher-forced ``(log_prob, entropy)``, both ``(B, A)``."""
        memory = self.encode(obs)
        mu, log_std = self._decode(memory, self._tgt_from_actions(actions))
        dist = Normal(mu, log_std.exp())
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Conditional log-probability of an existing joint action, ``(B, A)``."""
        logp, _ = self.evaluate(obs, actions)
        return logp

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Autoregressive joint action sampling -> ``(action, log_prob)``."""
        b, n, _ = obs.shape
        memory = self.encode(obs)
        actions = []
        logps = []
        for i in range(n):
            if i == 0:
                tgt = self.agent_embed[0].view(1, 1, -1).expand(b, 1, -1)
                mu, log_std = self._decode(memory, tgt)
                mu_i, std_i = mu[:, 0], log_std[:, 0].exp()
            else:
                prefix = torch.stack(actions, dim=1)                   # (B, i, d)
                mu, log_std = self._decode(memory, self._tgt_from_actions(prefix))
                mu_i, std_i = mu[:, -1], log_std[:, -1].exp()
            dist = Normal(mu_i, std_i)
            a = mu_i if deterministic else dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a).sum(-1))
        return torch.stack(actions, dim=1), torch.stack(logps, dim=1)
