# Algorithms

The twenty baselines in RoboRLBench, what each one changes relative to its
parent, and the hyper-parameters that ship with it.

Every algorithm is implemented from the paper and shares the same primitives
(`algos/nets.py`, `algos/buffers.py`, `algos/normalizer.py`), so a difference in
the reported return comes from the algorithm, not from a different MLP.

---

## Contents

- [The two agent families](#the-two-agent-families)
- [Single-agent baselines](#single-agent-baselines)
- [Multi-agent baselines](#multi-agent-baselines)
- [Hyper-parameters](#hyper-parameters)
- [Implementation notes and deviations](#implementation-notes-and-deviations)
- [Adding a baseline](#adding-a-baseline)

---

## The two agent families

Everything derives from `roborlbench.algos.base.AgentBase`, which owns the
observation normaliser, the checkpoint format and the `act(...)` entry point.
There are only two sub-families, and the trainer has exactly one branch on
`agent.kind`:

| | `OnPolicyAgent` (`kind = "on_policy"`) | `OffPolicyAgent` (`kind = "off_policy"`) |
|---|---|---|
| Data | `RolloutBuffer`, `rollout_steps x num_envs` per iteration | `ReplayBuffer`, circular, `buffer_size / num_envs` rows |
| Step | `act_batch(obs, state) -> (action, logp, value)` | `act(obs, state)`, then `store(...)`, then `update()` |
| Learning | `update(buffer)`: `num_epochs` passes over the rollout, GAE-lambda advantages | `update()`: `grad_steps` gradient steps per vectorised env step |
| Members | PPO, IPPO, MAPPO, HAPPO, HATRPO, MAT | SAC/TD3 family (14 baselines) |

Both families record `reward (N, A)` and `done (N,)` with a separate
`timeout (N,)` flag, so a time-limit truncation is bootstrapped while a true
terminal state is not.

### `grad_steps`, UTD and `num_envs`

One `env.step` on a vectorised environment adds `num_envs` transitions to the
replay buffer, and the runner then performs `grad_steps` gradient updates. The
update-to-data ratio is therefore

```
UTD = grad_steps / num_envs
```

so `grad_steps` has to grow with `num_envs` to keep the same UTD. The defaults
(`grad_steps = 1`, or 4/8 for REDQ / DroQ / FastTD3) target the `num_envs = 1`
analytic and MuJoCo backends, where `grad_steps == UTD`. Recommended values:

| `num_envs` | `grad_steps` for UTD ~ 0.1 | for UTD = 1 |
|---|---|---|
| 1 | 1 | 1 |
| 8 | 1 | 8 |
| 4096 | 512 | 4096 |

A UTD of 1 is what SAC / CrossQ / FastTD3 report on GPU; it means one update per
collected transition and is expensive. `configs/go2-flat-crossq.yaml` uses 512
updates on 4096 robots (UTD ~ 0.125) as a compromise on a single GPU.

---

## Single-agent baselines

Use these on the single-robot environments: `point-reach`, `go2-flat`,
`go2-mixed`, `go2-flat-isaacgym`, `hunt-single-isaacgym`, `so101-mujoco`,
`so101-lift`, `so101-pickplace`, `so101-stack`.

| Key | Paper | Venue | One-line contribution |
|---|---|---|---|
| `ppo` | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | arXiv 2017 | Clipped surrogate objective; the `rsl_rl` recipe used by essentially every legged-locomotion paper |
| `sac` | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) | ICML 2018 | Maximum-entropy actor-critic with automatic temperature tuning |
| `td3` | [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) | ICML 2018 | Clipped double-Q, target policy smoothing, delayed actor updates |
| `tqc` | [arXiv:2005.01578](https://arxiv.org/abs/2005.01578) | ICML 2020 | Truncated quantile critics control overestimation through a return distribution |
| `redq` | [arXiv:2101.05982](https://arxiv.org/abs/2101.05982) | ICLR 2021 | Critic ensemble + random target subset + high UTD |
| `droq` | [arXiv:2110.02034](https://arxiv.org/abs/2110.02034) | arXiv 2021 | LayerNorm + 1% dropout in the critics, making a high UTD stable |
| `td7` | [arXiv:2306.01105](https://arxiv.org/abs/2306.01105) | ICML 2023 | Learned encoder, SALE state averaging, LAP replay, LASER self-supervision |
| `crossq` | [OpenReview](https://openreview.net/forum?id=2VXyy9mIyU) | ICLR 2024 | BatchRenorm critics and **no target networks** at all |
| `simba` | [arXiv:2410.09754](https://arxiv.org/abs/2410.09754) | NeurIPS 2024 | Residual LayerNorm/SiLU blocks; architecture-only change to SAC |
| `fasttd3` | [arXiv:2502.06718](https://arxiv.org/abs/2502.06718) | arXiv 2025 | TD3 + critic ensemble + LayerNorm + batch 1024 + high UTD |

**Subclass chain** (each level only overrides the component its paper changes):

```
OffPolicyAgent
└── SAC ──── TQC            (quantile critics)
         ├── REDQ           (ensemble critic, random target subset)
         ├── DroQ           (dropout + LayerNorm critics)
         ├── CrossQ         (BatchRenorm critics, no target nets)
         └── SimBa          (SimBa actor + SimBa twin critic)
    TD3 ──── MATD3*         (*multi-agent variant)
         ├── FastTD3        (ensemble critics, LayerNorm, large batch)
         └── TD7            (encoder + SALE + LAP + LASER)
    PPO                     (on-policy)
```

## Multi-agent baselines

Use these on the multi-robot environments: `multi-hunt-2d`,
`hunt-multi-mujoco`, `hunt-multi-isaacgym`. They also accept a single-robot
environment (`num_agents == 1`) - useful to check that a CTDE method degrades
to its single-agent limit - and `tests/test_algos.py` does exactly that.

All ten follow **CTDE**: decentralised, parameter-shared actors that see only
the agent's own observation, and a centralised critic that sees the global
`state` and returns **one value per agent**. That design supports both a shared
team reward and per-agent rewards without changing the algorithm.

| Key | Paper | Venue | One-line contribution |
|---|---|---|---|
| `maddpg` | [arXiv:1706.02275](https://arxiv.org/abs/1706.02275) | NeurIPS 2017 | Deterministic actors with a critic over the joint action |
| `matd3` | [arXiv:1910.01465](https://arxiv.org/abs/1910.01465) | AAMAS 2019 | MADDPG + clipped double-Q, smoothing, delayed updates |
| `ippo` | [arXiv:2011.09533](https://arxiv.org/abs/2011.09533) | arXiv 2020 | Independent PPO: no information sharing (control condition) |
| `mappo` | [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) | NeurIPS 2021 | Shared PPO policy + centralised value function |
| `masac` | [arXiv:2103.13516](https://arxiv.org/abs/2103.13516) | arXiv 2021 | Multi-agent SAC with a centralised entropy-augmented twin critic |
| `facmac` | [arXiv:2006.14171](https://arxiv.org/abs/2006.14171) | ICLR 2021 | Factored (monotonic) critics over an embedding of the other agents' actions |
| `happo` | [arXiv:2110.02793](https://arxiv.org/abs/2110.02793) | ICLR 2022 | Sequential PPO with the joint importance ratio |
| `hatrpo` | [arXiv:2110.02793](https://arxiv.org/abs/2110.02793) | ICLR 2022 | The KL-constrained (TRPO) version of the same sequential update |
| `mat` | [arXiv:2205.14953](https://arxiv.org/abs/2205.14953) | ICML 2022 | Transformer encoder-decoder, autoregressive joint policy, PPO on the joint ratio |
| `hasac` | [arXiv:2203.11487](https://arxiv.org/abs/2203.11487) | ICLR 2023 | Sequential maximum-entropy updates with the joint importance ratio |

**Subclass chain:**

```
MARLOnPolicy (clustered PPO)
├── IPPO        centralized = False
├── MAPPO       centralized = True, clipped value loss
└── SequentialOnPolicy   (joint importance ratio, random agent order)
    ├── HAPPO   clipped surrogate per agent
    └── HATRPO  natural gradient + KL line search

MARLOffPolicy
├── MASAC ──── HASAC        (sequential entropy updates)
└── MADDPG ─── MATD3 ─── FACMAC   (factored critic)
```

---

## Hyper-parameters

Defaults are the paper values (or the values used by the reference
implementation) unless stated otherwise. Every keyword in these tables is
accepted through `--algo-kwargs` or `algo_kwargs:` in a config file; anything
else is rejected with the list of accepted names.

### Shared: `OnPolicyAgent`

| Parameter | Default | Meaning |
|---|---|---|
| `rollout_steps` | 24 | env steps collected per iteration (per parallel env) |
| `gamma`, `lam` | 0.99, 0.95 | discount and GAE-lambda |
| `num_epochs` | 5 | passes over the rollout |
| `num_minibatches` | 4 | minibatch size = `rollout_steps*num_envs/num_minibatches` |

### Shared: `OffPolicyAgent`

| Parameter | Default | Meaning |
|---|---|---|
| `batch_size` | 256 | transitions per gradient step |
| `buffer_size` | 1e6 | transitions (divided by `num_envs` for the ring buffer) |
| `warmup` | 2000 | transitions before the first gradient step |
| `grad_steps` | 1 | gradient steps per vectorised env step (see above) |
| `gamma`, `tau` | 0.99, 0.005 | discount and soft target update rate |

### `ppo`

| Parameter | Default |
|---|---|
| `actor_hidden`, `critic_hidden` | `(256, 128)`, `(256, 128)` |
| `lr` | 3e-4 |
| `clip_range` | 0.2 |
| `value_clip` | 0.0 (disabled) |
| `entropy_coef`, `value_coef` | 0.0, 1.0 |
| `max_grad_norm` | 1.0 |
| `adaptive_lr`, `desired_kl` | `True`, 0.01 (rsl_rl schedule) |
| `target_kl` | 0.0 (disabled) |

The adaptive learning rate follows `rsl_rl`: after each update the **last
epoch's** approximate KL is compared against `desired_kl` and the learning rate
is divided or multiplied by 1.5. (Averaging the KL over all epochs would be
wrong here - the first epoch always has ratio 1 and dilutes the estimate
towards zero, which makes the learning rate run away.)

### SAC family

| | `sac` | `tqc` | `redq` | `droq` | `crossq` | `simba` | `td3` | `td7` | `fasttd3` |
|---|---|---|---|---|---|---|---|---|---|
| `hidden` | (256,256) | (256,256) | (256,256) | (256,256) | (256,256) | - | (256,256) | - | (256,256) |
| `num_critics` | - | 5 | 10 | 2 | 2 | 2 | 2 | 2 | 10 |
| `num_target_subset` | - | - | 2 | - | - | - | - | - | 2 |
| `num_quantiles` | - | 25 | - | - | - | - | - | - | - |
| `top_quantiles_to_drop` | - | 2 | - | - | - | - | - | - | - |
| `dropout` | - | - | - | 0.01 | - | - | - | - | - |
| `grad_steps` | 1 | 1 | 4 | 4 | 1 | 1 | 1 | 1 | 8 |
| `batch_size` | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 1024 |
| `warmup` | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | **5000** | 2000 |
| `autotune` (`init_alpha`) | True (0.2) | True (0.2) | True (0.2) | True (0.2) | True (0.2) | True (0.2) | - | - | - |
| other | | `kappa=1.0` | | | no target nets | `hidden_dim=256, num_blocks=2` | `policy_delay=2, policy_noise=0.2, noise_clip=0.5` | `z_dim=256, laser_coef=1.0` | `policy_delay=2, policy_noise=0.2, noise_clip=0.5` |

### Multi-agent

| | `ippo` | `mappo` | `happo` | `hatrpo` | `mat` |
|---|---|---|---|---|---|
| critic | per-agent `V(o_i)` | centralised `V(s)` | centralised | centralised | centralised |
| `value_clip` | 0.0 | **0.2** | 0.0 | 0.0 | - |
| specific | - | - | `num_critic_epochs=5` | `max_kl=0.01, cg_iters=10, damping=0.01, backtrack_iters=10, backtrack_coef=0.5` | `d_model=64, nhead=4, enc/dec layers=2/2, ff=128` |

| | `maddpg` | `matd3` | `facmac` | `masac` | `hasac` |
|---|---|---|---|---|---|
| `hidden` | (256,256) | (256,256) | (256,256) | (256,256) | (256,256) |
| `tau` | 0.01 | 0.01 | 0.01 | 0.005 | 0.005 |
| specific | centralised `Q(s,a_1..a_A)` | `policy_delay=2, policy_noise=0.2, noise_clip=0.5` | + `embed_dim=32` | autotune entropy | + `ratio_clip=5.0` |

---

## Implementation notes and deviations

Documented so that a reviewer can see exactly where this benchmark follows the
paper and where it makes a deliberate, explicit choice.

* **One value per agent.** The centralised critics emit `A` values instead of
  one. With a shared team reward all `A` targets are identical, so this reduces
  exactly to the paper's `V(s)`; with per-agent rewards it is the only correct
  formulation. `MAPPO`, `HAPPO`, `HATRPO`, `MAT` and all off-policy MARL
  baselines use it.
* **Parameter sharing.** Actors are always shared across agents (the standard
  cooperative-MARL setup). HAPPO/HATRPO/HASAC are named "heterogeneous agent"
  because of their *update rule*, not because this benchmark instantiates
  different networks per agent; the sequential update with the joint importance
  ratio is implemented faithfully, and sharing is the `A = 1` special case of
  it.
* **`FACMAC`** keeps the factored monotone critic and the MATD3 actor, but uses
  the standard per-agent deterministic policy gradient instead of the paper's
  one-step joint gradient, so that it is comparable with `MATD3` and `MADDPG` on
  the same footing.
* **`TQC`** drops the `top_quantiles_to_drop` most optimistic quantiles from the
  target ensemble and uses the quantile Huber loss with `kappa = 1`; the
  temperature is SAC's.
* **`REDQ`/`FastTD3`** take the **minimum / mean** over a random subset of
  `num_target_subset` ensemble members per update (REDQ min, FastTD3 mean), both
  as in the respective papers.
* **`CrossQ`** has no target network at all: `_soft_update_targets` is a no-op
  and the current and next transitions go through the critics in a single
  forward pass so that both share the same BatchRenorm statistics.
* **`SimBa`** is architecture-only: the SAC objective, the temperature and the
  buffers are unchanged, only the actor and critic are built from
  `SimbaMLP` (residual LayerNorm/SiLU blocks).
* **`TD7`** implements all four components: the encoder with SALE state
  averaging, LAP (prioritised replay with an exploration bonus) and LASER
  (latent self-prediction). It is the only baseline that sets
  `use_priority = True` on the replay buffer.
* **`MAT`** uses a causal decoder so the joint policy factorises
  autoregressively; the PPO ratio is therefore computed on the **joint**
  log-probability (`sum_i log pi(a_i)`), not per agent.
* **`HATRPO`** solves the KL-constrained problem with a conjugate-gradient
  Fisher-vector product and a backtracking line search, and falls back to a
  plain gradient step if the CG solve degenerates - so it can never crash
  mid-training.
* **Observation normalisation** (`normalize_obs=True`) is on for every
  baseline and uses a running mean/std with a clip of 10. The normaliser is fed
  during action selection as well as during the update, so the log-probability
  ratio is always computed on the inputs the policy actually acted on.
  `normalize_state` is off by default (the global state already contains the
  per-agent observations).
* **Advantage normalisation** (zero mean, unit std, per agent) is applied in
  every on-policy baseline.

---

## Adding a baseline

An off-policy baseline is a class with `update()`; an on-policy one needs
`act_batch()` and `update(buffer)`. Both inherit checkpointing, normalisation,
`act()` and the `train()/eval()` mode switch.

```python
# roborlbench/algos/single/my_algo.py
from typing import Dict
import torch
from ..base import OffPolicyAgent
from ..nets import SquashedGaussianActor, TwinCritic

class MyAlgo(OffPolicyAgent):
    name = "myalgo"

    def __init__(self, *args, hidden=(256, 256), lr=3e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, hidden)
        self.critic = TwinCritic(self.state_dim + self.act_dim, hidden)

    def modules(self) -> Dict[str, torch.nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    def _act_impl(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        b = obs.shape[0]
        return self.actor.act(obs.reshape(b, -1), deterministic).reshape(
            b, self.num_agents, self.act_dim)

    def update(self) -> Dict[str, float]:
        batch = self.normalize_batch(self.sample(), self.obs_rms, self.state_rms)
        ...
        return {"critic_loss": float(loss)}
```

Then register it (one call, no other file changes):

```python
from roborlbench.algos import register_algo
register_algo("myalgo", MyAlgo, "single", "ICML 2026", 2026, "one-line summary")
```

`build_agent`, `roborlbench list`, `roborlbench train`, `roborlbench smoke` and
`tests/test_algos.py` pick it up automatically.

House rules for a pull request:

1. the module docstring must state the paper, the venue and **what the
   algorithm changes relative to its parent class**;
2. subclass an existing baseline when only one component changes (that is what
   makes the difference auditable);
3. keep shapes as documented: `obs (B, A, obs_dim)`, `action (B, A, act_dim)`,
   `reward (B, A)`, `discount (B, A)`;
4. return a dict of finite floats from `update()`;
5. `pytest -m "not slow"` and `roborlbench smoke` must pass.
