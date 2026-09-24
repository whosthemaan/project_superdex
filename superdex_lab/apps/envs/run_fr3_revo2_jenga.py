# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jenga: scripted baselines for pushing the loose block out of a tower level.

Three hand-written strategies drive ``Fr3Revo2JengaEnv``'s index fingertip:

- ``blind``: push the middle block straight through. Right only when it is the loose one.
- ``vision``: probe the blocks one by one (middle first) and back off when the top of the
  tower is seen to move, as a camera-only robot could.
- ``tactile``: probe the same way, but back off as soon as the fingertip's force rises,
  before the tower moves. A loose block slides with ~0.1 N; a load-bearing one resists
  with several newtons from the first millimeter.

These are baselines to beat, not policies: the env is meant for reinforcement learning.

Usage:
    python run_fr3_revo2_jenga.py [--strategy tactile|vision|blind|all] [--episodes 6]
                                  [--render]
"""

from __future__ import annotations

import argparse

import numpy as np
from superdex.lab.gym.envs.robots.fr3_revo2_jenga_env import (
    Fr3Revo2JengaEnv,
    Fr3Revo2JengaEnvCfg,
)

PROBE_ORDER = (1, 0, 2)  # middle block first
STANDOFF = 0.008  # [m] in front of the face while lining up
PROBE_SPEED = 0.25  # fraction of the env's tip_step while pushing
FORCE_LIMIT = 2.5  # [N] on the index fingertip that means "load-bearing"
SHIFT_FRAMES = 5  # camera frames averaged to see through the tracking noise
SHIFT_LIMIT = 0.0015  # [m] of seen tower motion that means "load-bearing"


class _Runner:
    """Steps the env and keeps the camera's recent views of the tower top."""

    def __init__(self, env: Fr3Revo2JengaEnv):
        self.env = env
        self.shifts: list[np.ndarray] = []
        self.info: dict = {}
        self.done = False

    def step(self, tip: np.ndarray) -> None:
        env = self.env
        flat, _, terminated, truncated, self.info = env.step(
            env.to_action({"tip": np.asarray(tip, dtype=np.float32)})
        )
        self.shifts.append(env.to_structured_observation(flat)["tower_shift"])
        self.done = terminated or truncated

    def seen_shift(self) -> np.ndarray:
        """The tower top's displacement, averaged over the last frames."""
        return np.mean(self.shifts[-SHIFT_FRAMES:], axis=0)

    def move(self, goal: np.ndarray, max_steps: int = 60) -> None:
        """Drive the fingertip to goal."""
        for _ in range(max_steps):
            delta = goal - self.env.tip_position()
            if np.linalg.norm(delta) < 0.001 or self.done:
                return
            self.step(np.clip(delta / self.env._cfg.tip_step, -1.0, 1.0))


def _felt_load(strategy: str, run: _Runner, shift0: np.ndarray) -> bool:
    """Whether the block being pushed carries the tower."""
    if strategy == "tactile":
        return run.info["index_force"] > FORCE_LIMIT
    if strategy == "vision":  # the tower top moving since this probe began
        return float(np.linalg.norm(run.seen_shift() - shift0)) > SHIFT_LIMIT
    return False


def run_episode(env: Fr3Revo2JengaEnv, strategy: str, seed: int) -> dict:
    env.reset(seed=seed)
    run = _Runner(env)
    order = (1,) if strategy == "blind" else PROBE_ORDER
    probes = 0
    for slot in order:
        run.move(env.face_center(slot) - [STANDOFF, 0.0, 0.0])
        if run.done:
            break
        probes += 1
        shift0 = run.seen_shift()
        backed_off = False
        while not run.done:
            run.step([PROBE_SPEED, 0.0, 0.0])
            if _felt_load(strategy, run, shift0):
                backed_off = True
                break
        if run.done or not backed_off:
            break
        # Back out of the level before trying the next block.
        tip = env.tip_position()
        run.move(np.array([env.face_center()[0] - STANDOFF, tip[1], tip[2]]))
    info = run.info
    return {
        "outcome": info.get("terminated_reason") or "gave up",
        "loose": info["loose_slot"],
        "pushed": info["pushed_slot"],
        "probes": probes,
        "time": env.get_step_count() * env.get_control_timestep(),
        "push": info["push"],
        "disturbance": info["disturbance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--strategy", choices=("tactile", "vision", "blind", "all"), default="all"
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--render", action="store_true", help="open the viewer")
    args = parser.parse_args()

    env = Fr3Revo2JengaEnv(
        Fr3Revo2JengaEnvCfg(render_mode="human" if args.render else None)
    )
    strategies = (
        ("blind", "vision", "tactile") if args.strategy == "all" else (args.strategy,)
    )
    print(
        f"{'strategy':8s} {'seed':>4s} {'loose':>5s} {'pushed':>6s} {'probes':>6s} "
        f"{'outcome':>16s} {'at [s]':>6s} {'push [mm]':>9s} {'tower moved [mm]':>16s}"
    )
    for strategy in strategies:
        wins = 0
        for seed in range(args.episodes):
            r = run_episode(env, strategy, seed)
            wins += r["outcome"] == "Success"
            print(
                f"{strategy:8s} {seed:4d} {r['loose']:5d} {r['pushed']:6d} "
                f"{r['probes']:6d} {r['outcome']:>16s} {r['time']:6.1f} "
                f"{1000 * r['push']:9.1f} {1000 * r['disturbance']:16.1f}"
            )
        print(f"{strategy:8s} succeeded {wins}/{args.episodes}\n")
    env.close()


if __name__ == "__main__":
    main()
