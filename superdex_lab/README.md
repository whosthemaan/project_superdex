# SuperDex Lab

[SuperDex Lab](https://projectsuperdex.com/lab/) provides Gymnasium-based
environment and benchmarking surfaces for Project SuperDex.

Provides the `superdex.lab` package: reinforcement-learning environments, reference
benchmark tasks, and the vectorized runners built on `superdex-robotics`.

```bash
pip install superdex-lab
```

The benchmark assets are not included with `pip install superdex-lab`. Clone the
[Project SuperDex repository](https://github.com/facebookresearch/project_superdex)
and set `SUPERDEX_ASSETS_PATH=<path-to-project_superdex>/assets` before running a
benchmark environment. See the
[setup guide](https://projectsuperdex.com/lab/docs/superdex_gym/setup/) for complete
instructions.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## FR3 + BrainCo Revo2 with tactile sensing

`superdex_gym/Fr3Revo2-v0` is a grasp-and-lift task: a Franka Research 3 on a desk,
carrying a BrainCo Revo2 hand, lifts a paper cup. Each fingertip has a Revo2 Touch-style
sensor (`superdex.lab.sensors.tactile`): normal force, tangential force and direction,
and proximity, at 0-25 N with 0.1 N resolution and 0-1 cm proximity range. Variants:
`fr3_revo2_left`, `fr3_revo2_box` (a box instead of the cup), `fr3_revo2_randomized`
(sensor noise and wider randomization). It runs on the published wheels plus this
source tree:

```bash
uv venv && uv pip install superdex
uv pip install --no-sources -e superdex_lab   # the env is newer than the superdex-lab wheel
export SUPERDEX_ASSETS_PATH=$PWD/assets

uv run --no-project superdex_lab/apps/envs/run_fr3_revo2_grasp.py --render  # scripted lift
uv run --no-project superdex_lab/apps/envs/run_sample.py fr3_revo2          # random actions
```

```python
import gymnasium as gym
from superdex.lab.gym.utils.env_discovery import register_all_envs

register_all_envs()
env = gym.make("superdex_gym/Fr3Revo2-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["tactile_normal_force"]  # {"thumb": ..., "index": ..., ...} [N]
info["tactile_proximity"]     # 0 (nothing within 1 cm) .. 1 (touching)
```

Actions are 7 arm joint-target increments and 6 absolute targets for the Revo2's actuated
joints. Observations include joint states, the object pose, and per fingertip
`[normal, tangential, cos(direction), sin(direction), proximity]`. See the module
docstring of `superdex/lab/gym/envs/robots/fr3_revo2_env.py` for the full layout, and
`assets/bots/hands/revo2/README.md` for the hand and sensor model.

The scripted demo closes each finger fast until proximity senses the cup, then regulates
each fingertip's normal force to 3 N, the kind of tactile grip the Revo2 Touch is built
for. It is a baseline, not a tuned policy (about 5/5 lifts per hand on its seeds).

In the default viewer, render models are drawn with their base material colors; heavy
grasping contact simulates at roughly 4-5x slower than real time on one core (free
motion runs faster than real time).

### Hold a paper cup while it fills

`superdex_gym/Fr3Revo2Fill-v0` is a tactile grip-force task. The Revo2 holds a paper cup
from the side (thumb up), the arm lifts it and sways gently, and water pours in at a
random rate to a random level (up to ~0.37 kg). The policy commands only the hand. It
has to squeeze just hard enough for the current, unobserved load: too light and the cup
slides out, too hard and it crushes (a fingertip above 6 N or a total squeeze above 25 N).
The fingertip shear the Revo2 Touch reports is what tells it the cup is getting heavier.

Variants: `Fr3Revo2FillLeft`, `Fr3Revo2FillStill` (no sway: easier, since the tapered cup
wedges in the hand), `Fr3Revo2FillRandomized` (sensor noise, stronger sway), and two
ablations: `Fr3Revo2FillNoTouch` (no tactile observation) and `Fr3Revo2FillOracleFill`
(told the fill level). Comparing a policy trained on the default against both ablations
shows what touch adds.

```bash
uv run --no-project superdex_lab/apps/envs/run_fr3_revo2_fill.py --render   # baselines
cd superdex_lab/apps/rllib && python train_samples.py -p fr3_revo2_fill     # PPO
```

Scripted baselines, 12 episodes each:

| Grip | Held | Mean squeeze |
|---|---|---|
| fixed light (0.4 N per fingertip) | 0 (4 dropped, 8 slid) | 3.3 N |
| fixed firm (3 N) | 0 (crushes the empty cup) | - |
| fixed medium (1 N) | 10 | 8.1 N |
| tactile (squeeze follows the load the fingertips sense) | 11 | 6.3 N |

Episodes run at about real time on one core (10 s simulated in ~8 s).

### Jenga: push the loose block out by touch

`superdex_gym/Fr3Revo2Jenga-v0` is a tactile probing task. A tower of 1.5x Jenga blocks
(9 levels) stands on the table; the Revo2 points its index finger at level 4, where one
of the three blocks is 1 mm thinner and carries no load. The policy moves the fingertip
(3D, Cartesian; inverse kinematics holds the hand's orientation) and has to push a
load-free block 35 mm in without moving the rest of the tower by more than 4 mm. Blocks
look alike and the loose one changes every episode: pressing a load-bearing block reads
3-10 N on the fingertip within its first millimeter, a load-free one 0.1-2 N, so the
policy can tell them apart before the tower moves. The block and tower positions it
observes carry 0.5 mm of camera-like noise.

Variants: `Fr3Revo2JengaRandomized` (sensor noise, noisier tracking, wider tower
placement), and two ablations: `Fr3Revo2JengaNoTouch` (no tactile observation: it can
only watch the tower move) and `Fr3Revo2JengaPerfectVision` (noise-free tracking).

```bash
uv run --no-project superdex_lab/apps/envs/run_fr3_revo2_jenga.py --render   # baselines
cd superdex_lab/apps/rllib && python train_samples.py -p fr3_revo2_jenga     # PPO
```

Scripted baselines, 12 episodes each:

| Strategy | Succeeded | Tower moved (mean / max) |
|---|---|---|
| blind: push the middle block | 6 | 2.1 / 4.7 mm |
| vision: probe blocks, back off when the camera sees the tower move | 11 | 1.5 / 4.1 mm |
| tactile: probe blocks, back off when the fingertip force rises | 12 | 0.4 / 1.2 mm |

Steps take ~33 ms on one core (the bottom 3 levels are fixed, and physics runs at 125 Hz).
