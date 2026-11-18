"""Registry of the twenty baselines.

The registry is the single place the CLI, the tests and the documentation read
from, so adding a new algorithm is a one-line registration plus the module.

Usage
-----
>>> from roborlbench.algos import build_agent, list_algos
>>> list_algos()
['crossq', 'droq', ...]
>>> agent = build_agent("sac", obs_dim=8, act_dim=2, state_dim=8)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import AgentBase, OffPolicyAgent, OnPolicyAgent
from .multi import (
    FACMAC,
    HASAC,
    HAPPO,
    HATRPO,
    IPPO,
    MADDPG,
    MAPPO,
    MASAC,
    MAT,
    MATD3,
)
from .single import (
    PPO,
    REDQ,
    SAC,
    TQC,
    TD3,
    TD7,
    CrossQ,
    DroQ,
    FastTD3,
    SimBa,
)

__all__ = [
    "AlgoSpec",
    "ALGOS",
    "register_algo",
    "get_algo_spec",
    "list_algos",
    "build_agent",
]


@dataclass
class AlgoSpec:
    """Metadata plus a factory for one baseline."""

    name: str
    cls: Callable[..., AgentBase]
    family: str                 # "single" | "multi"
    venue: str
    year: int
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return getattr(self.cls, "kind", "off_policy")

    @property
    def multi_agent(self) -> bool:
        return self.family == "multi"


ALGOS: Dict[str, AlgoSpec] = {}


def register_algo(
    name: str,
    cls: Callable[..., AgentBase],
    family: str,
    venue: str,
    year: int,
    description: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> AlgoSpec:
    if family not in ("single", "multi"):
        raise ValueError(f"family must be 'single' or 'multi', got {family!r}")
    if name in ALGOS:
        raise KeyError(f"algorithm {name!r} is already registered")
    spec = AlgoSpec(
        name=name,
        cls=cls,
        family=family,
        venue=venue,
        year=int(year),
        description=description,
        defaults=dict(defaults or {}),
    )
    ALGOS[name] = spec
    return spec


def get_algo_spec(name: str) -> AlgoSpec:
    try:
        return ALGOS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {sorted(ALGOS)}"
        ) from exc


def list_algos(family: Optional[str] = None) -> List[str]:
    if family is None:
        return sorted(ALGOS)
    return sorted(n for n, s in ALGOS.items() if s.family == family)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
# -- single-agent (single-robot environments) ------------------------------- #
register_algo(
    "ppo", PPO, "single", "arXiv 2017", 2017,
    "Clipped-surrogate on-policy policy gradient with a Gaussian policy (rsl_rl recipe)",
)
register_algo(
    "sac", SAC, "single", "ICML 2018", 2018,
    "Maximum-entropy off-policy actor-critic with automatic temperature tuning",
)
register_algo(
    "td3", TD3, "single", "ICML 2018", 2018,
    "Deterministic actor-critic with clipped double-Q and delayed policy updates",
)
register_algo(
    "tqc", TQC, "single", "ICML 2020", 2020,
    "SAC with truncated quantile critics (distributional control of overestimation)",
)
register_algo(
    "redq", REDQ, "single", "ICLR 2021", 2021,
    "SAC with a 10-critic ensemble, random target subset and a high UTD",
    defaults={"grad_steps": 4},
)
register_algo(
    "droq", DroQ, "single", "arXiv 2021", 2021,
    "SAC with dropout + LayerNorm critics and a high UTD",
    defaults={"grad_steps": 4},
)
register_algo(
    "td7", TD7, "single", "ICML 2023", 2023,
    "TD3 with a learned encoder, SALE averaging, LAP replay and LASER self-supervision",
)
register_algo(
    "crossq", CrossQ, "single", "ICLR 2024", 2024,
    "SAC with BatchRenorm critics and no target networks",
)
register_algo(
    "simba", SimBa, "single", "NeurIPS 2024", 2024,
    "SAC with residual LayerNorm/SiLU blocks (SimBa architecture)",
)
register_algo(
    "fasttd3", FastTD3, "single", "arXiv 2025", 2025,
    "TD3 with a critic ensemble, LayerNorm, large batches and a high UTD",
    defaults={"grad_steps": 8, "batch_size": 1024},
)

# -- multi-agent (multi-robot environments) --------------------------------- #
register_algo(
    "ippo", IPPO, "multi", "arXiv 2020", 2020,
    "Independent PPO: every agent has its own policy and its own critic",
)
register_algo(
    "maddpg", MADDPG, "multi", "NeurIPS 2017", 2017,
    "Deterministic actors with centralised critics over the joint action",
)
register_algo(
    "matd3", MATD3, "multi", "AAMAS 2019", 2019,
    "MADDPG with clipped double-Q, target policy smoothing and delayed updates",
)
register_algo(
    "mappo", MAPPO, "multi", "NeurIPS 2021", 2021,
    "Shared PPO policies with a centralised value function and clipped value loss",
)
register_algo(
    "masac", MASAC, "multi", "arXiv 2021", 2021,
    "Multi-agent SAC with a centralised entropy-augmented twin critic",
)
register_algo(
    "facmac", FACMAC, "multi", "ICLR 2021", 2021,
    "MATD3 with factored critics (monotonic aggregation of the other agents' actions)",
)
register_algo(
    "happo", HAPPO, "multi", "ICLR 2022", 2022,
    "Sequential PPO updates with the joint importance ratio (heterogeneous agents)",
)
register_algo(
    "hatrpo", HATRPO, "multi", "ICLR 2022", 2022,
    "Sequential TRPO updates with the joint importance ratio and a KL line search",
)
register_algo(
    "mat", MAT, "multi", "ICML 2022", 2022,
    "Transformer encoder-decoder over the agent sequence, trained with PPO",
)
register_algo(
    "hasac", HASAC, "multi", "ICLR 2023", 2023,
    "Sequential maximum-entropy updates with the joint importance ratio",
)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _accepted_params(cls: Callable[..., AgentBase]) -> set:
    """Every keyword a baseline ultimately accepts.

    Derived baselines forward ``**kwargs`` to their parent (``TQC`` -> ``SAC`` ->
    ``OffPolicyAgent``), so the whole MRO contributes to the accepted set.
    """
    names = set()
    for klass in cls.__mro__:
        if "__init__" not in vars(klass):
            continue
        names |= set(inspect.signature(klass.__init__).parameters)
    return names


def build_agent(
    name: str,
    obs_dim: int,
    act_dim: int,
    num_agents: int = 1,
    state_dim: Optional[int] = None,
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
    **overrides: Any,
) -> AgentBase:
    """Instantiate a baseline, validating the hyper-parameters and the shape.

    Raises
    ------
    KeyError
        Unknown algorithm name.
    ValueError
        A single-agent baseline was requested for a multi-agent environment.
    TypeError
        A hyper-parameter that the algorithm does not accept.
    """
    spec = get_algo_spec(name)
    if spec.family == "single" and int(num_agents) != 1:
        raise ValueError(
            f"{name!r} is a single-agent baseline but the environment has "
            f"num_agents={num_agents}; use one of the multi-agent baselines: "
            f"{list_algos('multi')}"
        )

    params = _accepted_params(spec.cls)
    unknown = sorted(set(overrides) - params)
    if unknown:
        raise TypeError(
            f"{name}: unknown hyper-parameter(s) {unknown}. "
            f"Accepted: {sorted(params - {'self'})}"
        )

    config = dict(spec.defaults)
    config.update(overrides)
    return spec.cls(
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        num_agents=int(num_agents),
        state_dim=int(state_dim) if state_dim is not None else int(obs_dim),
        num_envs=int(num_envs),
        device=device,
        seed=int(seed),
        **config,
    )
