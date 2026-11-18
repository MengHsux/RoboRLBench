# RoboRLBench

A reproducible deep-reinforcement-learning benchmark for **legged locomotion**,
**multi-robot cooperative hunting** and **arm manipulation**, with **20 SOTA
baselines** that all run on the same four robot simulators.

RoboRLBench wraps four existing simulation projects into one uniform
vectorised API, implements ten single-agent and ten multi-agent deep-RL
algorithms on top of that API, and ships a CLI, configs, tests and a
CPU-only smoke mode so that a result is always reproducible.

```
                                 ┌──────────────┐
   Go2_control        ─┐         │              │        ┌─ 10 single-agent baselines
   legged_hunt-main    ├─ BenchEnv ─ Runner ─ Trainer ──┤    PPO SAC TD3 TQC REDQ
   legged_hunt-single  ├─  (N,A,obs)                       │    DroQ TD7 CrossQ SimBa
   OpenSO-101-main    ─┘         │              │        └─ 10 multi-agent baselines
                                 └──────────────┘           IPPO MAPPO HAPPO HATRPO MAT
                                                            MADDPG MATD3 MASAC HASAC FACMAC
```

---

## Table of contents

- [Highlights](#highlights)
- [Benchmark at a glance](#benchmark-at-a-glance)
- [Installation](#installation)
- [Quickstart (CPU, 2 minutes)](#quickstart-cpu-2-minutes)
- [Command line interface](#command-line-interface)
- [Configuring a run](#configuring-a-run)
- [Wiring the four upstream simulators](#wiring-the-four-upstream-simulators)
- [Verified status](#verified-status)
- [Repository layout](#repository-layout)
- [Extending the benchmark](#extending-the-benchmark)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation and licence](#citation-and-licence)

---

## Highlights

| | |
|---|---|
| **One API, four simulators** | `BenchEnv` exposes `obs (N, A, obs_dim)`, `state (N, state_dim)`, `reward (N, A)`, `done (N,)` for MuJoCo, Isaac Gym and Isaac Lab alike. Algorithm code never touches a simulator. |
| **20 SOTA baselines** | 10 single-agent (2017-2025) and 10 multi-agent (2017-2023) algorithms, all written from the paper, all sharing the same network/buffer primitives so a comparison is a fair comparison. |
| **CTDE-ready for any number of robots** | Actors are decentralised and parameter-shared, critics are centralised and emit **one value per agent**, so the *same* baseline handles 1 Go2 and a 3-wolf pack, shared and individual rewards. |
| **Runs on a laptop** | Two analytically solvable environments (`point-reach`, `multi-hunt-2d`) are pure NumPy. All 20 baselines train on a CPU in about a minute (`roborlbench smoke`). |
| **Honest vectorisation** | Auto-reset inside `step`, post-reset observations, truncation flags for correct bootstrapping, replay buffers that keep the agent axis explicit. |
| **No vendored simulators** | The four upstream repositories are imported by path (`simpaths.yaml` / env vars), so this repository stays small and licence-clean on GitHub. |
| **Tested** | 111 automated tests: registry contracts, env shape contracts, one training test per baseline, checkpoint round-trips, CLI end-to-end, and learning-sanity checks that a baseline *improves* the return. |

---

## Benchmark at a glance

### Environments (12 tasks, 4 simulator families)

| Environment | Robots | Backend | Task |
|---|---|---|---|
| `point-reach` | 1 | analytic (NumPy) | 2-D point mass reaching a random goal, dense reward |
| `multi-hunt-2d` | 3 | analytic (NumPy) | 2-D cooperative prey encirclement |
| `hunt-multi-mujoco` | 3 | MuJoCo | 3 x Go2 wolf-hunt (`legged_hunt-main`) |
| `so101-mujoco` | 1 | MuJoCo | SO-101 reach / lift / pick / stack (`OpenSO-101-main`) |
| `hunt-multi-isaacgym` | 3 | Isaac Gym | 3 x Go2 wolf-hunt (`legged_hunt-main`) |
| `hunt-single-isaacgym` | 1 | Isaac Gym | single Go2 hunting (`legged_hunt-main-single`) |
| `go2-flat-isaacgym` | 1 | Isaac Gym | Go2 flat-ground velocity tracking (`legged_gym`) |
| `go2-flat` | 1 | Isaac Lab | Go2 flat-ground velocity tracking (`Go2_control`) |
| `go2-mixed` | 1 | Isaac Lab | Go2 mixed rough terrain (`Go2_control`) |
| `so101-lift` | 1 | Isaac Lab | SO-101 lift (`OpenSO-101-main`) |
| `so101-pickplace` | 1 | Isaac Lab | SO-101 pick-and-place (`OpenSO-101-main`) |
| `so101-stack` | 1 | Isaac Lab | SO-101 cube stacking (`OpenSO-101-main`) |

`roborlbench list` prints the live list with the number of robots and the
backend of every entry.

### Baselines (20)

| Single-agent | Venue / year | Multi-agent | Venue / year |
|---|---|---|---|
| `ppo` | arXiv 2017 (rsl_rl recipe) | `maddpg` | NeurIPS 2017 |
| `sac` | ICML 2018 | `matd3` | AAMAS 2019 |
| `td3` | ICML 2018 | `ippo` | arXiv 2020 |
| `tqc` | ICML 2020 | `mappo` | NeurIPS 2021 |
| `redq` | ICLR 2021 | `masac` | arXiv 2021 |
| `droq` | arXiv 2021 | `facmac` | ICLR 2021 |
| `td7` | ICML 2023 | `happo` | ICLR 2022 |
| `crossq` | ICLR 2024 | `hatrpo` | ICLR 2022 |
| `simba` | NeurIPS 2024 | `mat` | ICML 2022 |
| `fasttd3` | arXiv 2025 | `hasac` | ICLR 2023 |

Full references, the algorithmic difference of each entry and the
hyper-parameter table live in [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

**Rule of thumb:** a single-robot environment (`point-reach`, `go2-*`,
`so101-*`, `hunt-single-*`) takes the 10 single-agent baselines; a multi-robot
environment (`multi-hunt-2d`, `hunt-multi-*`) takes the 10 multi-agent ones.
`build_agent` enforces this and tells you which baselines are valid.

---

## Installation

Requires **Python >= 3.9**, `torch >= 2.0` and `numpy >= 1.21`.

```bash
git clone <this-repository> RoboRLBench
cd RoboRLBench

# CPU-only path: analytic environments, all 20 algorithms, tests
pip install -e ".[dev]"

# optional extras
pip install -e ".[dev,mujoco,log]"     # + MuJoCo environments + TensorBoard
```

Verify the installation:

```bash
roborlbench doctor          # backends, upstream repos, env shape probes
roborlbench list            # 20 baselines + 12 environments
roborlbench smoke           # train all 20 baselines for a few hundred steps
```

`smoke` is the recommended first command: it exercises every algorithm's
gradient update path and prints a table of the resulting episode returns.

The GPU backends (Isaac Gym, Isaac Lab) are **not** pip installable; see
[Wiring the four upstream simulators](#wiring-the-four-upstream-simulators) and
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

---

## Quickstart (CPU, 2 minutes)

No simulator required - two analytically solvable environments ship with the
package so the whole pipeline can be verified on any machine.

```bash
# 1. which algorithms and environments exist?
roborlbench list --family single

# 2. train SAC on the 2-D reaching task (about a minute on a laptop CPU)
roborlbench train --algo sac --env point-reach --num-envs 4 --total-steps 20000

# 3. score the checkpoint that was just written
roborlbench eval --checkpoint runs/sac-point-reach-seed0/final.pt --episodes 20

# 4. the multi-agent counterpart
roborlbench train --algo mappo --env multi-hunt-2d --num-envs 4 --total-steps 20000
```

A uniform random policy scores about **-196** on `point-reach` and about
**-814** on `multi-hunt-2d`; after 20k steps SAC and MAPPO are expected to be
clearly better than that (that is exactly what `tests/test_learning.py`
asserts).

From Python, without the CLI:

```python
from roborlbench.runners import TrainConfig, Trainer

config = TrainConfig(env="point-reach", algo="sac", num_envs=8,
                     total_steps=200_000, eval_interval=20_000)
summary = Trainer(config).train()
print(summary["final_eval_return"])
```

The two pieces the trainer wires together are public, so a custom loop is short:

```python
from roborlbench.algos import build_agent
from roborlbench.envs import make_env

env = make_env("point-reach", num_envs=8, seed=0)
agent = build_agent("sac", obs_dim=env.obs_dim, act_dim=env.act_dim,
                    num_agents=env.num_agents, state_dim=env.state_dim,
                    num_envs=env.num_envs, seed=0)

obs = env.reset()
action = agent.act(obs.obs, obs.state)      # (num_envs, num_agents, act_dim)
step = env.step(action)                     # Step(obs, state, reward, done, info)
```

---

## Command line interface

```
roborlbench <command> [options]

  train    train one baseline on one environment
  eval     evaluate a saved checkpoint
  smoke    train every baseline for a few hundred steps
  doctor   check installed backends and probe every available environment
  list     list registered baselines and environments
```

| Command | Useful flags |
|---|---|
| `train` | `--config`, `--algo`, `--env`, `--num-envs`, `--total-steps`, `--seed`, `--device`, `--log-dir`, `--run-name`, `--eval-interval`, `--eval-episodes`, `--save-interval`, `--tensorboard`, `--algo-kwargs`, `--env-kwargs` |
| `eval` | `--checkpoint` (required), `--env`, `--algo`, `--episodes`, `--num-envs`, `--seed`, `--device`, `--algo-kwargs`, `--env-kwargs` |
| `smoke` | `--algos`, `--env`, `--env-single`, `--env-multi`, `--steps`, `--num-envs`, `--device`, `--log-dir` |
| `doctor` | `--analytic-only`, `--steps` |
| `list` | `--family {single,multi}`, `--env-kind {single,multi}` |

`eval` reads `config.json` from the checkpoint's directory, so `--env` and
`--algo` are usually unnecessary. Hyper-parameter overrides are JSON:

```bash
roborlbench train --algo crossq --env point-reach --total-steps 50000 \
    --algo-kwargs '{"batch_size": 512, "grad_steps": 2}'
```

Every run writes to `runs/<run-name>/`:

```
runs/sac-point-reach-seed0/
  config.json     # the full resolved configuration + env shapes
  metrics.csv     # one row per iteration (train_return, losses, eval_return)
  summary.json    # wall time, steps/s, best and final evaluation return
  final.pt        # checkpoint understood by `roborlbench eval`
  step_*.pt       # optional periodic checkpoints
```

---

## Configuring a run

A run is described by `roborlbench.runners.TrainConfig` and can come from a
YAML/JSON file, CLI flags, or the Python API (CLI flags override the file):

```yaml
# configs/point-reach-sac.yaml
env: point-reach
algo: sac

num_envs: 4
total_steps: 20000
seed: 0
device: cpu

eval_interval: 5000
eval_episodes: 10
save_interval: 0

log_dir: runs
run_name: sac-point-reach

env_kwargs: {}          # forwarded to the environment factory
algo_kwargs:            # forwarded to the algorithm constructor
  batch_size: 256
  warmup: 2000
  grad_steps: 1
```

```bash
roborlbench train --config configs/point-reach-sac.yaml
roborlbench train --config configs/point-reach-sac.yaml --total-steps 200000
```

Four ready-to-use configs ship in `configs/`:

| Config | Demonstrates |
|---|---|
| `point-reach-sac.yaml` | CPU-only off-policy baseline |
| `go2-flat-crossq.yaml` | Isaac Lab locomotion, 4096 envs, scaled `grad_steps` |
| `hunt-multi-happo.yaml` | MuJoCo 3-robot hunt, on-policy sequential MARL |
| `so101-td7.yaml` | MuJoCo arm manipulation, model-based-free SOTA |

### Budgets that work

`num_envs` multiplies both the transitions per second and the number of rows a
replay buffer gains per step. Off-policy baselines perform `grad_steps` updates
after every vectorised step, so the update-to-data ratio is
`UTD = grad_steps / num_envs`: keep `grad_steps` at 1 on the analytic and MuJoCo
backends, and scale it with `num_envs` on GPU (512 for `num_envs = 4096` is
UTD ~ 0.125; 4096 would be UTD = 1).

| Backend | `num_envs` | `total_steps` | notes |
|---|---|---|---|
| analytic | 1-8 | 20k - 200k | `point-reach` / `multi-hunt-2d` sanity checks |
| MuJoCo | 1-8 | 2M - 20M | CPU, one process, ~130-160 steps/s on `hunt-multi-mujoco` |
| Isaac Gym / Isaac Lab | 1024-4096 | 20M - 200M | GPU, `device: cuda:0` |

On-policy baselines (PPO family) are far more sample hungry than off-policy
ones: expect `point-reach` to need ~200k steps for PPO against ~20k for SAC.
This is the usual trade-off and is not a bug in either implementation.

`hunt-multi-mujoco` inherits the upstream task's horizon of
`episode_length_s = 200` (10 000 control steps per episode), so evaluation - not
training - dominates a short run. Override it while iterating:

```bash
roborlbench train --algo mappo --env hunt-multi-mujoco --num-envs 2 \
    --env-kwargs '{"episode_length_s": 20}' --total-steps 200000
```

---

## Wiring the four upstream simulators

RoboRLBench does **not** vendor the simulator projects. Each adapter imports
the upstream package by name; `roborlbench/envs/_simpaths.py` decides where to
look, in this order:

1. **Environment variable** (highest priority):
   `ROBORLBENCH_LEGGED_HUNT`, `ROBORLBENCH_LEGGED_HUNT_SINGLE`,
   `ROBORLBENCH_GO2_RL_LAB`, `ROBORLBENCH_OPENSO101`.
2. **`simpaths.yaml`** in the working directory or in `~/.roborlbench/`
   (copy `simpaths.example.yaml` and edit the paths).
3. **A sibling directory** found by walking up from the package, so the default
   layout works with no configuration at all:

```
workspace/
├── RoboRLBench/                 # this repository
├── legged_hunt-main/
├── legged_hunt-main-single/
├── Go2_control/
│   └── go2_rl_lab-lab3.0/
└── OpenSO-101-main/
```

Check the result with `roborlbench doctor`, which prints the resolved path of
every upstream repository and probes every environment it can instantiate.

| Upstream project | Provides | Backend | Extra requirement |
|---|---|---|---|
| `legged_hunt-main` | 3 x Go2 wolf-hunt | Isaac Gym | `legged_gym`, `isaacgym` |
| `legged_hunt-main-single` | single Go2 hunt | Isaac Gym | `legged_gym`, `isaacgym` |
| `Go2_control` | Go2 flat / mixed terrain | Isaac Lab 3.0 | `isaaclab`, `isaacsim` |
| `OpenSO-101-main` | SO-101 reach / lift / pick / stack | Isaac Lab + MuJoCo | `isaaclab` and/or `mujoco` |

Both GPU backends create their own app / simulator instance, so run **one
training process per GPU** and keep them in separate processes.

---

## Verified status

Everything below is produced by the test-suite and the CLI in this repository;
no numbers are copied from the papers.

`roborlbench smoke` (512 steps per baseline, CPU, ~60 s):

* **20/20 baselines trained successfully**, i.e. every algorithm performed at
  least one gradient update and produced finite metrics.

The same check against the **real MuJoCo backends** - `so101-mujoco` for the ten
single-agent baselines, `hunt-multi-mujoco` (3 x Go2) for the ten multi-agent
ones - also reports **20/20**, and deterministic `train` + `eval` runs write
loadable checkpoints on both. The Isaac Gym and Isaac Lab adapters are covered by
contract tests but cannot be exercised here (no GPU / Linux simulator on this
machine); `roborlbench doctor` reports their status on yours.

`pytest` (111 tests):

* registry contracts: 20 baselines, 10 per family, complete metadata, no
  duplicate names, unknown hyper-parameters rejected, single-agent baselines
  rejected on multi-agent environments;
* environment contracts: shapes, reset determinism, auto-reset, timeout flags,
  wrong-action-shape rejection, for `num_envs` 1 and 4;
* one training test per baseline (all 20) asserting that a parameter actually
  changed and that every logged metric is finite;
* checkpoint round-trip for four representative baselines;
* CLI end-to-end (`train` -> `eval`), orphan-checkpoint handling, `doctor`,
  `list`, `smoke` on a subset;
* learning sanity (marked `slow`): SAC beats random by > 80 return points on
  `point-reach`, MAPPO beats random by > 200 on `multi-hunt-2d`.

```bash
pytest                 # everything (a few minutes)
pytest -m "not slow"   # fast subset
pytest -m slow         # only the learning-sanity tests
```

---

## Repository layout

```
RoboRLBench/
├── roborlbench/
│   ├── envs/
│   │   ├── base.py              # BenchEnv / Obs / Step / registry
│   │   ├── surrogate.py         # point-reach, multi-hunt-2d (pure NumPy)
│   │   ├── _simpaths.py         # locate the four upstream repositories
│   │   ├── mujoco_backend.py    # legged_hunt + OpenSO-101 (MuJoCo)
│   │   ├── isaacgym_backend.py  # legged_hunt / legged_gym (Isaac Gym)
│   │   └── isaaclab_backend.py  # Go2_control / OpenSO-101 (Isaac Lab)
│   ├── algos/
│   │   ├── base.py              # AgentBase / OnPolicyAgent / OffPolicyAgent
│   │   ├── nets.py              # actors, critics, SimBa blocks, BatchRenorm
│   │   ├── buffers.py           # ReplayBuffer, RolloutBuffer (agent-aware)
│   │   ├── normalizer.py        # RunningMeanStd
│   │   ├── registry.py          # the 20 baselines + build_agent validation
│   │   ├── single/              # ppo sac td3 tqc redq droq td7 crossq simba fasttd3
│   │   └── multi/               # ippo mappo happo hatrpo mat + maddpg matd3 masac hasac facmac
│   ├── runners/                 # TrainConfig, Trainer, evaluate_policy, RunLogger
│   ├── cli/                     # train / eval / smoke / doctor / list
│   └── registry.py              # re-exports envs + algos registration
├── configs/                     # ready-to-run YAML configurations
├── docs/
│   ├── ALGORITHMS.md
│   └── ENVIRONMENTS.md
├── scripts/                     # verification helpers
├── tests/
├── simpaths.example.yaml
├── LICENSE
└── pyproject.toml
```

Design rules the code follows everywhere:

* one environment interface, one agent interface - the runner has exactly one
  branch (`on_policy` vs `off_policy`) and nothing else;
* the agent axis is explicit in every tensor, so `num_agents == 1` is not a
  special case;
* every algorithm is a subclass of a shared base and only overrides the
  component that its paper changes;
* simulator imports are lazy, so `import roborlbench` never requires a
  simulator.

---

## Extending the benchmark

### Add an algorithm

```python
# roborlbench/algos/single/my_algo.py
from ..base import OffPolicyAgent

class MyAlgo(OffPolicyAgent):
    name = "myalgo"

    def update(self):
        batch = self.normalize_batch(self.sample(), self.obs_rms, self.state_rms)
        ...            # your loss
        return {"critic_loss": float(loss)}
```

Register it once:

```python
from roborlbench.algos import register_algo
register_algo("myalgo", MyAlgo, "single", "ICML 2026", 2026, "one-line summary")
```

`build_agent`, the CLI, `smoke`, the tests and `roborlbench list` pick it up
automatically.

### Add an environment

```python
# roborlbench/envs/my_backend.py
from .base import BenchEnv, Obs, Step

class MyEnv(BenchEnv):
    obs_dim, act_dim, num_agents = 31, 12, 1

    def reset(self, mask=None) -> Obs: ...
    def step(self, actions) -> Step: ...      # auto-reset + info["timeout"]
```

```python
from roborlbench.envs import register_env
register_env("my-env", "roborlbench.envs.my_backend:make_my_env",
             kind="single", backend="mujoco")
```

The contract is documented in detail in
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

---

## Testing

```bash
pytest -m "not slow"    # ~2 min: contracts, all 20 training smoke tests, CLI
pytest                  # + learning sanity (a few minutes)
roborlbench smoke       # same as the training smoke test, as a CLI table
```

Test markers (declared in `pyproject.toml`):

* `slow` - learning-sanity runs that train a policy to a threshold;
* `sim` - tests that need a GPU simulator (skipped by default).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `environment 'go2-flat' needs backend 'isaaclab', but ... could not be imported` | The upstream repository is not on the path or Isaac Lab is not installed. Run `roborlbench doctor` and read [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md). |
| `'sac' is a single-agent baseline but the environment has num_agents=3` | Use one of the multi-agent baselines: `roborlbench list --family multi`. |
| `unknown hyper-parameter(s) [...]` | The algorithm does not accept that keyword; the error message lists the accepted ones. |
| `reading YAML configs requires pyyaml` | `pip install pyyaml`. |
| Training is slow on CPU | Increase `num_envs` (vectorised envs), reduce `grad_steps`, or use the analytic environments for algorithm development. |
| `UnicodeEncodeError` in a Windows console | Already handled: the CLI reconfigures stdout to replace unencodable paths. |
| A GPU run crashes at startup | Only one Isaac Gym / Isaac Lab process per GPU; make sure the `PYTHONPATH` set up by `_simpaths.py` is not shadowed by a global `isaacgym` install. |

---

## Citation and licence

If you use this benchmark, cite the algorithms you actually ran from
[docs/ALGORITHMS.md](docs/ALGORITHMS.md); each entry links to its paper.

The upstream simulators keep their own licences and are not redistributed
here. RoboRLBench itself is released under the MIT licence - see
[LICENSE](LICENSE).
