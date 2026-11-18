# Environments

How RoboRLBench wraps the four simulation projects, the exact tensor contract
every adapter must satisfy, and the installation steps per backend.

---

## Contents

- [The `BenchEnv` contract](#the-benchenv-contract)
- [Environment catalogue](#environment-catalogue)
- [The two analytic environments](#the-two-analytic-environments)
- [Wiring the upstream repositories](#wiring-the-upstream-repositories)
- [Backend: MuJoCo](#backend-mujoco)
- [Backend: Isaac Gym](#backend-isaac-gym)
- [Backend: Isaac Lab](#backend-isaac-lab)
- [Diagnostics](#diagnostics)
- [Adding a new environment](#adding-a-new-environment)

---

## The `BenchEnv` contract

One interface for every simulator. Algorithms never import a simulator module.

```python
from roborlbench.envs import make_env

env  = make_env("hunt-multi-mujoco", num_envs=8, seed=0)
obs  = env.reset()                        # Obs(obs=(8, 3, obs_dim), state=(8, state_dim))
step = env.step(actions)                  # actions: (8, 3, act_dim) in [-1, 1]
# step.obs     (8, 3, obs_dim)   next observations, already auto-reset
# step.state   (8, state_dim)    centralised state for the critics
# step.reward  (8, 3)            one reward per agent
# step.done    (8,)              episode boundary mask
# step.info    {"timeout": (8,)}
```

| Element | Shape | Meaning |
|---|---|---|
| `obs` | `(num_envs, num_agents, obs_dim)` | decentralised actor input |
| `state` | `(num_envs, state_dim)` | centralised critic input; equals `obs` for single-robot tasks |
| `action` | `(num_envs, num_agents, act_dim)` | **always** normalised to `[-1, 1]` per actuator |
| `reward` | `(num_envs, num_agents)` | per-agent reward (broadcast for a shared reward) |
| `done` | `(num_envs,)` | episode ended in this step (terminal **or** truncated) |
| `info["timeout"]` | `(num_envs,)` | subset of `done` that is a **time-limit truncation** |

Rules the adapters guarantee, and that `probe_env()` checks:

1. **Auto-reset inside `step`.** Environments that terminate are reset before
   the observation is returned, exactly like `rsl_rl` and Isaac Lab. The
   returned `obs` are therefore *post-reset* for the copies whose `done` is
   true.
2. **Truncation is not termination.** A time-limit end sets the bit in both
   `done` and `info["timeout"]`. Off-policy agents bootstrap from truncations
   (`discount = 1`) and do not bootstrap from real terminals. This removes the
   classical "time-limit-is-terminal" bias.
3. **Actions are always in `[-1, 1]`.** Each adapter maps that box onto the
   simulator's own action space (torque limits, joint targets, velocity
   commands). A policy trained on one robot therefore *runs* on another without
   rescaling - only the action dimension and the observation differ.
4. **`obs` and `state` contain only finite values.**
5. **`reset(mask=...)` is supported** and resets only the selected copies.

`Obs.squeeze_agents()` returns plain `(num_envs, obs_dim)` views for the
single-agent baselines; it raises if called on a multi-agent observation, which
catches an accidental family mix-up.

---

## Environment catalogue

| Registry key | Robots | `obs_dim` | `act_dim` | `state_dim` | Backend |
|---|---|---|---|---|---|
| `point-reach` | 1 | 8 | 2 | 8 | analytic |
| `multi-hunt-2d` | 3 | 18 | 2 | 58 | analytic |
| `hunt-multi-mujoco` | 3 | `num_obs` | `num_actions` | `3 x num_privileged_obs` | MuJoCo (`legged_hunt-main`) |
| `so101-mujoco` | 1 | `OBS_DIM` | `ACTION_DIM` | = `obs_dim` | MuJoCo (`OpenSO-101-main`) |
| `hunt-multi-isaacgym` | 3 | task | 12 | task | Isaac Gym (`legged_hunt-main`) |
| `hunt-single-isaacgym` | 1 | task | 12 | task | Isaac Gym (`legged_hunt-main-single`) |
| `go2-flat-isaacgym` | 1 | task | 12 | task | Isaac Gym (`legged_gym`) |
| `go2-flat` | 1 | task | 12 | task | Isaac Lab (`Go2_control`) |
| `go2-mixed` | 1 | task | 12 | task | Isaac Lab (`Go2_control`) |
| `so101-lift` | 1 | task | 6 | task | Isaac Lab (`OpenSO-101-main`) |
| `so101-pickplace` | 1 | task | 6 | task | Isaac Lab (`OpenSO-101-main`) |
| `so101-stack` | 1 | task | 6 | task | Isaac Lab (`OpenSO-101-main`) |

The simulator rows are **not** hard-coded: every adapter reads the dimensions
from the upstream task configuration when the environment is constructed
(`num_obs`, `num_actions`, `num_privileged_obs`, `OBS_DIM`, ...). Run
`roborlbench doctor` to print the real shape of every environment your
installation can instantiate.

`roborlbench list --env-kind multi` shows the multi-robot tasks only.

---

## The two analytic environments

Pure NumPy, no simulator, deterministic given a seed. They exist for three
reasons: the whole benchmark runs on a laptop, they are the fixtures of the
test-suite, and they are fast research sandboxes for a new idea.

### `point-reach` - single agent

* observation: `[pos(2), vel(2), goal(2), prev_action(2)]` -> 8
* action: 2-D acceleration in `[-1, 1]`
* dynamics: `vel <- 0.9*vel + action`, speed clipped to 1.0, arena `[-1.5, 1.5]`
* reward: `-||pos - goal||`, plus `+10` on success
* horizon 100 steps, success radius 0.08

Reference numbers (10 random episodes, seed 0): uniform random policy
**-196.4**; SAC after 20k steps **-2.7**.

### `multi-hunt-2d` - 3 agents

* observation per agent: `[pos(2), vel(2), prey_rel(2), target_rel(2),
  prev_action(2)]` + `(A-1)` neighbour blocks `[rel_pos(2), rel_vel(2)]`
  -> `10 + 4(A-1)`, i.e. **18** for `A = 3`
* global state: all observations concatenated + the prey pose -> `A*obs_dim + 4`
* action: 2-D acceleration in `[-1, 1]`
* reward: shared; `-sum_i ||p_i - t_i||` for the ring slots, `-0.02` per
  collision, `+A` when every agent is within 0.12 of its slot
* horizon 150 steps

Reference numbers: uniform random **-813.6**; MAPPO after 20k steps
**-157.6**.

Both are fully vectorised over `num_envs` and share the same auto-reset and
truncation logic as the real backends.

---

## Wiring the upstream repositories

RoboRLBench does not vendor the simulators. `roborlbench/envs/_simpaths.py`
resolves each of them, first hit wins:

| Priority | Source |
|---|---|
| 1 | environment variable `ROBORLBENCH_LEGGED_HUNT`, `ROBORLBENCH_LEGGED_HUNT_SINGLE`, `ROBORLBENCH_GO2_RL_LAB`, `ROBORLBENCH_OPENSO101` |
| 2 | `simpaths.yaml` in the working directory or `~/.roborlbench/simpaths.yaml` |
| 3 | a sibling directory, searched upwards from the package |

```yaml
# simpaths.yaml  (copy from simpaths.example.yaml)
legged_hunt:        /data/code/legged_hunt-main
legged_hunt_single: /data/code/legged_hunt-main-single
go2_rl_lab:         /data/code/Go2_control/go2_rl_lab-lab3.0
openso101:          /data/code/OpenSO-101-main
```

```bash
# equivalent, per shell
export ROBORLBENCH_LEGGED_HUNT=/data/code/legged_hunt-main
export ROBORLBENCH_GO2_RL_LAB=/data/code/Go2_control/go2_rl_lab-lab3.0
```

The default sibling layout needs no configuration at all:

```
workspace/
├── RoboRLBench/
├── legged_hunt-main/
├── legged_hunt-main-single/
├── Go2_control/go2_rl_lab-lab3.0/
└── OpenSO-101-main/
```

Each repository contributes specific importable roots (`source/go2_rl_lab` for
Isaac Lab extensions, `src` for OpenSO-101, ...). The adapters insert those
roots into `sys.path` on demand, so **nothing is imported - and no path is
touched - until you instantiate an environment of that backend**.

If a repository cannot be found, the error message names the environment
variable to set.

---

## Backend: MuJoCo

| Upstream | Adapter | Registry keys |
|---|---|---|
| `legged_hunt-main/mujoco_tools/wolf_env.py` -> `WolfMuJoCoEnv` | `WolfMuJoCoAdapter` | `hunt-multi-mujoco` |
| `OpenSO-101-main/src/openso101/offpolicy/env.py` -> `VecSO101Env` | `SO101MuJoCoAdapter` | `so101-mujoco` |

Install: `pip install -e ".[mujoco]"` plus the upstream repository's own
requirements. Runs on CPU.

**`WolfMuJoCoAdapter`** wraps `num_fields` parallel arenas x `num_wolfs` Go2
quadrupeds per arena. The upstream environment exposes `obs (F, W, 52)`,
`privileged_obs (F, W, 55)`, `rewards (F*W,)` and `dones (F*W,)` with per-robot
auto-reset. The adapter

* folds `F*W` into `(num_envs, num_agents)` so rewards become `(F, W)`;
* sets the episode mask to `any` over the wolves of an arena (an arena ends
  when any wolf is done, which is how the upstream task is scored);
* builds the centralised state from the concatenated privileged observations of
  all wolves -> `state_dim = W * 55`.

The upstream task is **long-horizon**: `episode_length_s = 200` at a 0.02 s
control period, i.e. `max_episode_steps = 10000`. One evaluation episode
therefore costs about 10 000 environment steps, which dominates the wall time
of a short run. For quick iteration override it:

```bash
roborlbench train --algo mappo --env hunt-multi-mujoco \
    --env-kwargs '{"episode_length_s": 20}' --num-envs 2 --total-steps 200000
```

Measured on a laptop CPU (3 Go2 per field): 6.1 ms per vectorised step with
`num_envs=1`, 6.8 ms with 2 and 7.8 ms with 4, i.e. 130-160 steps/s.

**`SO101MuJoCoAdapter`** exposes the SO-101 arm with four tasks
(`task: reach | lift | pick | stack`), `act_dim = 6`. The global state is the
observation (single robot). Episodes are 200 steps.

```bash
roborlbench train --algo td7 --env so101-mujoco \
    --env-kwargs '{"task": "lift"}' --num-envs 8 --total-steps 5000000
```

## Backend: Isaac Gym

| Upstream | Registry keys |
|---|---|
| `legged_hunt-main` (`legged_gym` task `wolf`) | `hunt-multi-isaacgym` |
| `legged_hunt-main-single` (`wolf`) | `hunt-single-isaacgym` |
| `legged_gym` (`go2`) | `go2-flat-isaacgym` |

Requires a working `isaacgym` installation (NVIDIA GPU, `python` bindings) and
`legged_gym` on the path. `legged_gym` environments already auto-reset and
expose `(obs, privileged_obs, rew, dones, extras)`, so the adapter
(`LeggedGymAdapter`) only re-shapes the per-robot tensors into
`(num_envs, num_agents, dim)`.

One implementation detail worth knowing: `legged_gym.utils.get_args()` parses
`sys.argv`, which would swallow RoboRLBench's own flags. The adapter builds an
equivalent `argparse.Namespace` by hand, so `roborlbench train ...` and a
standalone `legged_gym` script can coexist.

## Backend: Isaac Lab

| Upstream | Gym id | Registry key |
|---|---|---|
| `Go2_control` (`go2_rl_lab-lab3.0`) | `go2-flat` | `go2-flat` |
| `Go2_control` (`go2_rl_lab-lab3.0`) | `go2-mixed` | `go2-mixed` |
| `OpenSO-101-main` | `OpenSO101-Lift-v0` | `so101-lift` |
| `OpenSO-101-main` | `OpenSO101-PickPlace-v0` | `so101-pickplace` |
| `OpenSO-101-main` | `OpenSO101-Stack-v0` | `so101-stack` |

Requires Isaac Lab 3.0 / Isaac Sim. Isaac Lab environments are `gymnasium`
environments returning `(obs, reward, terminated, truncated, extras)` and reset
terminated copies internally. Observations are **dicts**; the adapter

* uses the `"policy"` observation group as the actor input,
* concatenates every other group as the centralised critic state,
* maps `terminated` / `truncated` onto `done` / `info["timeout"]`.

Isaac Sim is started **once per process** (headless by default) and reused, so a
`num_envs` sweep inside one Python process does not pay the startup cost twice.

```bash
roborlbench train --algo crossq --env go2-flat --config configs/go2-flat-crossq.yaml
```

Run **one GPU process per GPU**: both Isaac Gym and Isaac Lab create a global
simulator instance.

---

## Diagnostics

`roborlbench doctor` is the first thing to run on a new machine:

```bash
roborlbench doctor                    # everything
roborlbench doctor --analytic-only    # CPU only, no simulator imports
roborlbench doctor --steps 10         # also step each env 10 times
```

It prints the Python / torch / numpy versions, the resolved path of each of the
four upstream repositories, whether each backend imports, and then instantiates
and probes every environment it can - validating the shape contract from
[above](#the-benchenv-contract) and reporting the dimensions it measured.

A healthy CPU-only installation ends with the two analytic environments
`ok` and the simulator backends marked as unavailable, without raising.

---

## Adding a new environment

```python
# roborlbench/envs/my_backend.py
import numpy as np
from .base import BenchEnv, Obs, Step

class MyEnv(BenchEnv):
    backend = "mujoco"          # free-form; used by diagnostics only
    obs_dim, act_dim, num_agents = 31, 12, 1

    def __init__(self, num_envs=1, seed=0, device="cpu"):
        super().__init__()
        self.num_envs = int(num_envs)
        self.state_dim = self.obs_dim          # for a single robot
        # ... create the simulator state ...

    def reset(self, mask=None) -> Obs:
        ...
        return Obs(obs, state)

    def step(self, actions) -> Step:
        actions = self.check_actions(actions)   # validates (+-1, shapes)
        ...
        return Step(obs=obs, state=state, reward=reward, done=done,
                    info={"timeout": timeout})
```

Register it once:

```python
from roborlbench.envs import register_env
register_env("my-env", "roborlbench.envs.my_backend:make_my_env",
             kind="single", backend="mujoco", max_episode_steps=1000)
```

Then:

* `roborlbench doctor` probes it automatically and will tell you exactly which
  shape is wrong if the contract is violated;
* `tests/test_envs.py` patterns apply unchanged - see `probe_env()` in
  `envs/__init__.py`, which is the same code the CLI uses.

Checklist for a new adapter:

1. accept `[-1, 1]` normalised actions and clip them to the actuator limits;
2. auto-reset inside `step`, and return the **post-reset** observation;
3. set `timeout` for time-limit ends and *not* for real terminals;
4. keep `obs` finite and of shape `(num_envs, num_agents, obs_dim)`;
5. import the simulator lazily, inside the factory, so that
   `import roborlbench` never requires it;
6. run `roborlbench doctor --steps 10` before committing.
