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

"""Behavior tests for the Jenga environment (``Fr3Revo2JengaEnv``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from superdex.lab.gym.envs.robots.fr3_revo2_env import FINGERS
from superdex.lab.gym.envs.robots.fr3_revo2_jenga_env import (
    Fr3Revo2JengaEnv,
    Fr3Revo2JengaEnvCfg,
)

_DEMO = Path(__file__).resolve().parents[2] / "apps" / "envs" / "run_fr3_revo2_jenga.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("run_fr3_revo2_jenga", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env():
    with Fr3Revo2JengaEnv(Fr3Revo2JengaEnvCfg()) as env:
        yield env


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


def _still(env):
    return env.to_action({"tip": np.zeros(3, np.float32)})


def _probe(env, demo, seed: int, slot: int, steps: int = 12) -> dict:
    """Line the fingertip up with a block and push it slowly; the last step's info."""
    env.reset(seed=seed)
    run = demo._Runner(env)
    run.move(env.face_center(slot) - [demo.STANDOFF, 0.0, 0.0])
    forces = []
    for _ in range(steps):
        run.step([demo.PROBE_SPEED, 0.0, 0.0])
        forces.append(run.info["index_force"])
        if run.done:
            break
    return {**run.info, "max_force": max(forces)}


def test_spaces(env):
    assert env.action_space.shape == (3,)
    structure = env.get_observation_space_structure()
    assert structure["tactile"].shape == (len(FINGERS) * 5,)
    assert structure["block_push"].shape == (3,)
    with Fr3Revo2JengaEnv(Fr3Revo2JengaEnvCfg(observe_tactile=False)) as ablation:
        assert "tactile" not in ablation.get_observation_space_structure()


def test_the_loose_block_changes_between_episodes(env):
    slots = {env.reset(seed=seed)[1]["loose_slot"] for seed in range(8)}
    assert slots == {0, 1, 2}


def test_tower_stands_still(env):
    """Left alone, the tower settles by a fraction of a millimeter and stays put."""
    env.reset(seed=0)
    for _ in range(25):
        *_, terminated, _, info = env.step(_still(env))
        assert not terminated
    assert info["disturbance"] < 0.0005
    assert info["push"] < 0.0005
    assert max(info["tactile_normal_force"].values()) == 0.0


def test_fingertip_follows_its_target(env):
    env.reset(seed=0)
    start = env.tip_position()
    for _ in range(10):
        env.step(env.to_action({"tip": np.array([0.0, 0.0, 1.0], np.float32)}))
    moved = env.tip_position() - start
    # 10 steps of 4 mm straight up, within a few millimeters.
    np.testing.assert_allclose(moved, [0.0, 0.0, 0.04], atol=0.004)


def test_touch_tells_a_loose_block_from_a_load_bearing_one(env, demo):
    """Seed 2's middle block is loose; seed 0's middle block carries the tower."""
    loose = _probe(env, demo, seed=2, slot=1)
    assert loose["loose_slot"] == 1
    loaded = _probe(env, demo, seed=0, slot=1)
    assert loaded["loose_slot"] != 1
    assert loose["max_force"] < 0.5
    assert loaded["max_force"] > demo.FORCE_LIMIT


def test_pushing_a_load_bearing_block_disturbs_the_tower(env, demo):
    result = demo.run_episode(env, "blind", seed=0)  # the middle block is loaded
    assert result["outcome"] == "Tower disturbed"
    assert env._last_reward["failure"] < 0.0


def test_tactile_probing_finds_the_loose_block(env, demo):
    """Seed 1's loose block is an outer one: the probe backs off the middle first."""
    result = demo.run_episode(env, "tactile", seed=1)
    assert result["outcome"] == "Success"
    assert result["probes"] == 2
    assert result["disturbance"] < env._cfg.disturb_limit
    assert env._last_reward["success"] > 0.0
