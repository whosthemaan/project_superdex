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

"""Jenga: push a loose block out of a tower by touch. FR3 + BrainCo Revo2 with
fingertip tactile sensing.

A tower of wooden blocks (3 per level, alternating directions) stands on the table. The
Revo2 points its index finger at one level (the other fingers and the thumb curled). One
of that level's three blocks is loose: 1 mm thinner than the others, so the level above
rests on the other two. As in real Jenga, a block that carries no load slides out with a
light push, while pushing a load-bearing one drags the levels above it along. When the
loose block is an outer one, the tower balances on the middle block and the other outer
block carries almost nothing either, so it comes out too. From outside the blocks look
the same, and which one is loose changes every episode. The policy has to feel for it:
press a block, and if the fingertip's force climbs before the block moves, back off and
try another one.

Scripted baselines (``apps/envs/run_fr3_revo2_jenga.py``, 12 episodes each, default
config): pushing the middle block blindly succeeds 6 times; probing block by block and
backing off when a (noisy) camera sees the tower move succeeds 11 times, moving the
tower by 1.3 mm on average; backing off when the fingertip force rises succeeds 12 times,
moving it by 0.4 mm on average (at most 1.2 mm). A learned policy should match the
tactile one.

The policy moves the index fingertip in Cartesian space. The hand orientation and
posture are held, and differential inverse kinematics turns the fingertip target into arm
joint targets. The fingertip target never leads the fingertip by more than ``tip_lead``,
which bounds how hard the finger can push (like a compliant arm).

Blocks are 1.5x standard Jenga (112.5 x 37.5 x 22.5 mm, 57 g), so the Revo2's fingertip
fits in a level's gap. The hand is yawed 30 degrees toward the palm side so that the
fingertip pad, where the Revo2 Touch sensor sits, meets the block. Pushing a loose block
reads about 0.1-0.3 N on the index fingertip, a load-bearing one 3-5 N within the first
5 mm.

Action (in [-1, 1]):

- ``tip`` (3): the fingertip target's motion along world x (into the tower), y and z, up
  to ``tip_step`` per step.

Observation (positions relative to the center of the target level's front face [m]):

- ``tip_pos``, ``tip_vel``, ``tip_target`` (3 each).
- ``tactile`` (5 x 5): per fingertip, normal force, tangential force, tangential direction
  as (cos, sin), and proximity, as in ``Fr3Revo2Env``.
- ``block_push`` (3): how far each of the target level's blocks (by slot, -y to +y) has
  been pushed into the tower [m], and ``tower_shift`` (3): displacement of the top
  level's middle block [m]; both as a camera would track them, with ``vision_noise_std``
  of noise.
- ``prev_action`` (3).

Episode outcome (``info``): **success** when a target-level block is pushed in by
``success_push`` while no other block has moved more than ``disturb_limit``; the tower is
**disturbed** (failure) when any other block moves more than that. Both end the episode.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import superdex.physics as physics
import trimesh
from scipy.spatial.transform import Rotation
from superdex.lab.gym.envs import (
    ActionSpace,
    Info,
    MochiEnv,
    ObservationSpace,
    RewardTerms,
    StructuredAction,
    StructuredObservation,
)
from superdex.lab.gym.envs.robots.fr3_revo2_env import (
    _SCENE_HANDLES,
    FINGERS,
    FLOOR_RGB,
    PROXIMITY_SDF_PADDING,
    TABLE_RGB,
    Fr3Revo2Env,
    Fr3Revo2EnvCfg,
)
from superdex.lab.gym.utils import mochi_helpers
from superdex.lab.gym.utils.render_materials import apply_render_model_colors
from superdex.lab.sensors import SdfProximityTarget
from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.decorators import override_from

# 1.5x standard Jenga blocks: length, width, height [m]; mass of wood at 0.6 g/cm^3.
BLOCK_SIZE = (0.1125, 0.0375, 0.0225)
BLOCK_MASS = 0.057  # [kg]
BLOCK_RGB = (0.86, 0.70, 0.47)
# Side-by-side clearance between a level's blocks [m]. Real blocks never sit perfectly
# flush; touching sides would clamp the middle block between its neighbors.
SLOT_GAP = 0.0005
# Stiff, short-range contact so stacked levels barely sink into each other.
BLOCK_CONTACT = dict(
    penalty_smoothing_half_distance=0.00025, penalty_threshold_default=0.00025
)
# Coarse meshes (corners, edge midpoints and face centers as contact samples) keep the
# tower cheap to simulate; it stands as still as with 60x the samples.
BLOCK_MESH_EDGE = 0.06  # [m]

# The tower's center on the table [m]; the target level's blocks run along x.
TOWER_XY = (0.68, 0.15)

# Pointing hand, in action units (thumb metacarpal, thumb flexion, index, middle, ring,
# pinky): index extended, the rest curled.
POINTING_HAND = (-1.0, 1.0, -1.0, 1.0, 1.0, 1.0)
# Fingertip point on the index pad's leading edge, in the hand base frame [m].
TIP_IN_HAND = (0.018, 0.033, 0.143)
# World-from-hand rotation held throughout: index on top, palm facing +y, fingers along +x
# yawed 30 degrees toward -y. The index's distal segment curls back ~17 degrees, so its
# pad (the sensing face) then meets a block's end at ~77 degrees incidence, within the
# sensor's 85 degrees; with less yaw the push lands on the fingertip's housing and goes
# unsensed, with more the finger rubs the neighboring blocks.
FINGER_YAW = np.radians(30.0)
HAND_ROTATION = np.array(
    [
        [np.sin(FINGER_YAW), 0.0, np.cos(FINGER_YAW)],
        [np.cos(FINGER_YAW), 0.0, -np.sin(FINGER_YAW)],
        [0.0, 1.0, 0.0],
    ]
)
# Arm joints [rad] placing the fingertip 30 mm in front of the target level's middle
# block (level 4 of the default tower) in HAND_ROTATION. Solved with inverse kinematics;
# >= 0.78 rad from every joint limit.
ARM_START = {"right": (0.9871, 0.773, -0.3463, -2.1975, -1.8414, 1.8573, 2.2317)}

IK_DAMPING = 1e-3
IK_ROTATION_WEIGHT = 0.3  # [m/rad], orientation error against position error


@functools.cache
def _block_model(height: float) -> physics.ModelData:
    """A block's mesh and SDF, centered at its origin (shared by every scene)."""
    box = trimesh.creation.box((BLOCK_SIZE[0], BLOCK_SIZE[1], height))
    while box.edges_unique_length.max() > BLOCK_MESH_EDGE:
        box = box.subdivide()
    model = physics.ModelData()
    model.mesh = physics.MeshData(
        nodes_per_element=3,
        coordinates=box.vertices.ravel(),
        connectivity=box.faces.ravel(),
    )
    physics.model.bake_sdf(
        model,
        physics.GridSdfParams(
            resolution_mode=physics.GridSdfResolutionMode.EXPLICIT,
            resolution_delta=[0.001] * 3,
            boundary_padding_dist=PROXIMITY_SDF_PADDING,
            min_grid_resolution=[6, 6, 6],
        ),
    )
    return model


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@configclass
class Fr3Revo2JengaEnvCfg(Fr3Revo2EnvCfg):
    """Options for the Jenga environment."""

    steps_per_episode: int = 250  # 10 s at 25 Hz
    simulation_frequency: int = 125
    """Half the base env's rate: the tower and the pointing hand stay as steady, at half
    the cost."""
    object_name: str = "paper_cup"  # unused: the tower replaces the object
    object_xy: tuple[float, float] = TOWER_XY

    levels: int = 9
    """Levels in the tower (3 blocks each)."""
    target_level: int = 4
    """Level (from 0 at the table) whose blocks the finger faces; the start pose is
    solved for level 4."""
    static_levels: int = 3
    """Bottom levels simulated as fixed (they carry the tower but a fingertip at the
    target level cannot move them); fewer moving blocks make each step cheaper. At most
    ``target_level - 1``, so the level below the target still moves."""
    loose_gap: float = 0.001
    """How much thinner the loose block is than the others [m]."""
    tower_xy_noise: float = 0.005
    """Half-width of the uniform randomization of the tower's position [m]."""
    block_friction: tuple[float, float] = (0.3, 0.5)
    """Wood-on-wood (and fingertip-on-wood) Coulomb friction, randomized per episode."""

    tip_step: float = 0.004
    """Largest fingertip target motion per control step [m] (10 cm/s)."""
    tip_lead: float = 0.004
    """Largest distance the fingertip target may lead the fingertip [m]; bounds the push
    force."""
    workspace: tuple[tuple[float, float], ...] = (
        (-0.06, 0.05),
        (-0.07, 0.07),
        (-0.025, 0.05),
    )
    """Bounds of the fingertip target along x, y, z, relative to the target level's
    front-face center [m]."""
    observe_tactile: bool = True
    """Observe the fingertip sensors (disable for the no-touch ablation)."""
    vision_noise_std: float = 0.0005
    """Gaussian noise on ``block_push`` and ``tower_shift`` [m], as tracking blocks with
    a camera from ~0.6 m would have; 0 gives the true values."""

    success_push: float = 0.035
    """Push-in of a target-level block that counts as success [m] (a third of the block
    then sticks out of the far side, ready to be taken)."""
    disturb_limit: float = 0.004
    """Displacement of any other block that counts as disturbing the tower [m]. Pushing
    a loose block out moves the rest by about 1 mm at most."""
    terminate_on_success: bool = True

    # Reward weights.
    reach_weight: float = 1.0
    push_weight: float = 100.0
    disturb_weight: float = 50.0
    force_weight: float = 0.02
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    action_rate_weight: float = 0.01


class Fr3Revo2JengaEnv(Fr3Revo2Env):
    """Find the loose block of a Jenga level by touch and push it out."""

    SCENE_PREFIX = "fr3_revo2_jenga"

    def __init__(self, cfg: Fr3Revo2JengaEnvCfg | dict[str, Any]):
        if not isinstance(cfg, Fr3Revo2JengaEnvCfg):
            cfg = Fr3Revo2JengaEnvCfg(**cfg)
        if cfg.hand_side not in ARM_START:
            raise ValueError("Fr3Revo2JengaEnv supports the right hand")
        if not (1 <= cfg.target_level <= cfg.levels - 2):
            raise ValueError("target_level needs a level below and above it")
        if not (0 <= cfg.static_levels <= cfg.target_level - 1):
            raise ValueError("static_levels must leave the level below the target free")
        super().__init__(cfg)
        self._cfg: Fr3Revo2JengaEnvCfg = cfg

        self._tip_in_hand = np.asarray(TIP_IN_HAND)
        self._tip = np.zeros(3)
        self._tip_vel = np.zeros(3)
        self._tip_target = np.zeros(3)
        self._friction = float(np.mean(cfg.block_friction))
        self._loose_slot = 1
        self._slots = np.zeros(3, dtype=int)  # target-level actor index by slot
        self._reference: np.ndarray | None = None
        self._prev_action = np.zeros(3)
        self._push = np.zeros(3)
        self._best_push = 0.0
        self._rewarded_push = 0.0
        self._disturbance = 0.0

        self._setup_action_space(tip=ActionSpace(-1.0, 1.0, (3,), dtype=np.float32))
        observation_space = {
            "tip_pos": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "tip_vel": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "tip_target": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "block_push": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "tower_shift": ObservationSpace(-np.inf, np.inf, (3,), dtype=np.float32),
            "prev_action": ObservationSpace(-1.0, 1.0, (3,), dtype=np.float32),
        }
        if cfg.observe_tactile:
            observation_space["tactile"] = ObservationSpace(
                -np.inf, np.inf, (len(FINGERS) * 5,), dtype=np.float32
            )
        self._setup_observation_space(**observation_space)

        if self._renderer:
            x, y = TOWER_XY
            self._renderer.set_camera_view(
                look_from=[x - 0.45, y - 0.55, 0.35], look_at=[x - 0.05, y, 0.1]
            )

    ####################################################################################
    # Scene
    ####################################################################################

    @classmethod
    def arm_home(cls, side: str) -> tuple[float, ...]:
        return ARM_START[side]

    @staticmethod
    def _create_object(scene: physics.Scene, cfg: Fr3Revo2JengaEnvCfg):
        """The tower, in its nominal pose. The target level has two load-bearing blocks
        and a thinner, loose one; which slot each takes is set per episode."""
        h = BLOCK_SIZE[2]
        tight = physics.create_model_shape(_block_model(h))
        loose = physics.create_model_shape(_block_model(h - cfg.loose_gap))
        contact = physics.ContactParams(
            coulomb_friction_coefficient=float(np.mean(cfg.block_friction)),
            friction_falloff_vel=1e-4,
            **BLOCK_CONTACT,
        )
        loose_block = None
        for level, slot, name in _tower_layout(cfg):
            is_loose = name.endswith("loose")
            block = scene.create_rigid_actor(
                name=name,
                shape=loose if is_loose else tight,
                is_static=level < cfg.static_levels,
                mass=BLOCK_MASS * ((h - cfg.loose_gap) / h if is_loose else 1.0),
                boundary_element_type=physics.ActorBoundaryElementType.DEFAULT,
                contact=contact,
                world_from_local=_block_pose(cfg, level, slot, np.zeros(2), is_loose),
            )
            if is_loose:
                loose_block = block
        # The base env treats the returned actor as its object (the fingertip sensors'
        # proximity target until the Jenga scene replaces it with the whole level).
        return loose_block, np.zeros(3), h / 2, _block_model(h).sdf

    @classmethod
    def _build_scene(cls, scene_key: str, cfg: Fr3Revo2JengaEnvCfg):
        scene, agent, cleanups = super()._build_scene(scene_key, cfg)
        # Proximity: the target level and the levels just above and below it.
        blocks = _find_blocks(scene)
        near = [
            (name, actor)
            for name, actor in blocks.items()
            if abs(_level_of(name) - cfg.target_level) <= 1
        ]
        targets = [
            SdfProximityTarget(
                actor,
                _block_model(
                    BLOCK_SIZE[2] - (cfg.loose_gap if name.endswith("loose") else 0.0)
                ).sdf,
            )
            for name, actor in near
        ]
        for sensor in _SCENE_HANDLES[scene_key].sensors.values():
            sensor.set_proximity_targets(targets)
        return scene, agent, cleanups

    def _init_scene(self, cfg: Fr3Revo2JengaEnvCfg):
        super()._init_scene(cfg)
        # Hold the pointing hand from the start.
        lo, hi = self._dof_min[self._hand_dofs], self._dof_max[self._hand_dofs]
        hand = lo + 0.5 * (np.asarray(POINTING_HAND) + 1.0) * (hi - lo)
        self._initial_pose[self._hand_dofs] = hand
        for follower, leader, multiplier in self._mimic:
            self._initial_pose[follower] = multiplier * self._initial_pose[leader]

        blocks = _find_blocks(self._scene)
        self._layout = _tower_layout(cfg)
        self._blocks = [blocks[name] for _, _, name in self._layout]
        self._target_blocks = [
            i
            for i, (level, _, _) in enumerate(self._layout)
            if level == cfg.target_level
        ]  # tight0, tight1, loose
        self._other_blocks = [
            i for i in range(len(self._layout)) if i not in self._target_blocks
        ]
        top = cfg.levels - 1
        self._top_block = next(
            i
            for i, (level, slot, _) in enumerate(self._layout)
            if level == top and slot == 1
        )

    @override_from(Fr3Revo2Env)
    def _reset_object(self):
        cfg, rng = self._cfg, self.np_random
        self._tower_xy = rng.uniform(-cfg.tower_xy_noise, cfg.tower_xy_noise, 2)
        self._loose_slot = int(rng.integers(3))
        tight_slots = [s for s in range(3) if s != self._loose_slot]
        # Target-level actors are (tight0, tight1, loose).
        slot_of = {0: tight_slots[0], 1: tight_slots[1], 2: self._loose_slot}
        self._slots = np.zeros(3, dtype=int)
        for k, i in enumerate(self._target_blocks):
            self._slots[slot_of[k]] = i
        for i, (level, slot, name) in enumerate(self._layout):
            if level == cfg.target_level:
                slot = slot_of[self._target_blocks.index(i)]
            pose = _block_pose(cfg, level, slot, self._tower_xy, name.endswith("loose"))
            self._blocks[i].set_root_transform(pose)
            if level >= cfg.static_levels:
                self._blocks[i].set_velocity([0, 0, 0], [0, 0, 0])
        self._friction = float(rng.uniform(*cfg.block_friction))
        self._apply_friction()

        self._face = np.array(
            [
                TOWER_XY[0] + self._tower_xy[0] - BLOCK_SIZE[0] / 2,
                TOWER_XY[1] + self._tower_xy[1],
                (cfg.target_level + 0.5) * BLOCK_SIZE[2],
            ]
        )
        self._reference = None
        self._prev_action = np.zeros(3)
        self._push = np.zeros(3)
        self._best_push = 0.0
        self._rewarded_push = 0.0
        self._disturbance = 0.0
        self._tip = self._tip_position()
        self._tip_vel = np.zeros(3)
        self._tip_target = self._tip.copy()

    def _apply_friction(self):
        """This env's friction onto the (possibly shared) blocks."""
        first = self._blocks[0].get_contact_params()
        if first.coulomb_friction_coefficient == np.float32(self._friction):
            return
        for block in self._blocks:
            contact = block.get_contact_params()
            contact.coulomb_friction_coefficient = self._friction
            block.set_contact_params(contact)

    ####################################################################################
    # Stepping
    ####################################################################################

    def _tip_position(self) -> np.ndarray:
        T = self._wrist.get_root_transform()
        R = Rotation.from_quat(list(T.rotation)).as_matrix()
        return R @ self._tip_in_hand + np.asarray(T.translation)

    def tip_position(self) -> np.ndarray:
        """The index fingertip, world frame [m]."""
        return self._tip.copy()

    def face_center(self, slot: int = 1) -> np.ndarray:
        """Center of the front face of the target level's block in ``slot`` (0-2, from -y
        to +y; the default, the middle one, is the level's center), world frame [m]."""
        return self._face + [0.0, (slot - 1) * (BLOCK_SIZE[1] + SLOT_GAP), 0.0]

    @override_from(MochiEnv)
    def _simulate(self, action: StructuredAction):
        cfg = self._cfg
        tip = np.clip(np.asarray(action["tip"], dtype=np.float64), -1.0, 1.0)
        self._action_rate = float(np.sum((tip - self._prev_action) ** 2))
        self._prev_action = tip.copy()
        target = self._tip_target + tip * cfg.tip_step
        lo = self._face + np.array([b[0] for b in cfg.workspace])
        hi = self._face + np.array([b[1] for b in cfg.workspace])
        target = np.clip(target, lo, hi)
        # Never lead the fingertip by more than tip_lead.
        lead = target - self._tip
        dist = np.linalg.norm(lead)
        if dist > cfg.tip_lead:
            target = self._tip + lead * (cfg.tip_lead / dist)
        self._tip_target = target
        self._substep = 0
        MochiEnv._simulate(self, action)

    @override_from(MochiEnv)
    def _apply_action(self, action: StructuredAction):
        if self._substep == 0:
            # After the scene state is restored: this env's friction and arm goal.
            self._apply_friction()
            self._goal[self._arm_dofs] = self._solve_ik()
            np.clip(self._goal, self._dof_min, self._dof_max, out=self._goal)
        self._substep += 1
        super()._apply_action(action)

    def _solve_ik(self) -> np.ndarray:
        """Arm joints moving the fingertip to its target with the hand's orientation
        held: one damped least-squares step from the measured pose."""
        n = self._agent.get_num_dofs()
        J = np.asarray(self._wrist.get_articulated_jacobian(), dtype=np.float64)
        J = J.reshape(6, n)[:, self._arm_dofs]
        T = self._wrist.get_root_transform()
        R = Rotation.from_quat(list(T.rotation)).as_matrix()
        r = R @ self._tip_in_hand
        tip = r + np.asarray(T.translation)
        J_tip = np.vstack([J[:3] - _skew(r) @ J[3:], IK_ROTATION_WEIGHT * J[3:]])
        error = np.concatenate(
            [
                self._tip_target - tip,
                IK_ROTATION_WEIGHT
                * Rotation.from_matrix(HAND_ROTATION @ R.T).as_rotvec(),
            ]
        )
        dq = J_tip.T @ np.linalg.solve(
            J_tip @ J_tip.T + IK_DAMPING**2 * np.eye(6), error
        )
        q = mochi_helpers.get_articulated_pose(self._agent)[self._arm_dofs]
        return np.asarray(q, dtype=np.float64) + dq

    ####################################################################################
    # Observation, reward, termination
    ####################################################################################

    def _block_positions(self) -> np.ndarray:
        return np.array(
            [list(b.get_root_transform().translation) for b in self._blocks]
        )

    @override_from(MochiEnv)
    def _make_observation(self) -> tuple[StructuredObservation, Info]:
        cfg = self._cfg
        tip = self._tip_position()
        self._tip_vel = (tip - self._tip) / self._control_dt
        self._tip = tip

        readings = {
            f: self._sensors[f].compute_signal(self._control_dt) for f in FINGERS
        }
        self._readings = readings
        tactile, normals = self._tactile_features(readings)

        positions = self._block_positions()
        if self._reference is None or self._step_count <= 1:
            # The tower settles by a fraction of a millimeter in the first step.
            self._reference = positions
        displacement = positions - self._reference
        push = displacement[self._slots, 0]
        others = np.linalg.norm(displacement[self._other_blocks], axis=1)
        self._push = push
        self._disturbance = float(others.max())
        best = int(np.argmax(push))
        self._best_push = float(push[best])

        seen = np.concatenate([push, displacement[self._top_block]])
        if cfg.vision_noise_std > 0:
            seen = seen + self.np_random.normal(0.0, cfg.vision_noise_std, 6)
        obs = {
            "tip_pos": tip - self._face,
            "tip_vel": self._tip_vel,
            "tip_target": self._tip_target - self._face,
            "block_push": seen[:3],
            "tower_shift": seen[3:],
            "prev_action": self._prev_action,
        }
        if cfg.observe_tactile:
            obs["tactile"] = tactile
        obs = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}

        disturbed = self._disturbance > cfg.disturb_limit
        success = bool(self._best_push >= cfg.success_push and not disturbed)
        index = readings["index"]
        info = {
            "tactile_normal_force": normals,
            "tactile_tangential_force": {
                f: float(v)
                for f, v in zip(FINGERS, tactile.reshape(len(FINGERS), 5)[:, 1])
            },
            "tactile_proximity": {f: readings[f].proximity for f in FINGERS},
            "index_force": float(np.hypot(index.normal_force, index.tangential_force)),
            "loose_slot": self._loose_slot,
            "pushed_slot": best,
            "push": self._best_push,
            "loose_push": float(push[self._loose_slot]),
            "disturbance": self._disturbance,
            "friction": self._friction,
            "tip_to_face": float(self._reach_distance(tip)),
            "disturbed": bool(disturbed),
            "is_success": success,
        }
        return obs, info

    def _reach_distance(self, tip: np.ndarray) -> float:
        """Distance from the fingertip to the target level's front face [m]."""
        w, h = 1.5 * BLOCK_SIZE[1], 0.5 * BLOCK_SIZE[2]
        d = tip - self._face
        outside = np.array(
            [max(-d[0], 0.0), max(abs(d[1]) - w, 0.0), max(abs(d[2]) - h, 0.0)]
        )
        return float(np.linalg.norm(outside))

    @override_from(MochiEnv)
    def _compute_reward_terms(
        self, action: StructuredAction, observation: StructuredObservation, info: Info
    ) -> RewardTerms:
        cfg = self._cfg
        # Pay for new push-in only, so pushing a block back and forth earns nothing.
        progress = float(np.clip(info["push"], 0.0, cfg.success_push))
        gain = max(progress - self._rewarded_push, 0.0)
        self._rewarded_push = max(self._rewarded_push, progress)
        return {
            "reach": -cfg.reach_weight * info["tip_to_face"],
            "push": cfg.push_weight * gain,
            "disturb": -cfg.disturb_weight * info["disturbance"],
            "force": -cfg.force_weight * info["index_force"],
            "success": cfg.success_bonus * info["is_success"],
            "failure": -cfg.failure_penalty * info["disturbed"],
            "action_rate": -cfg.action_rate_weight * self._action_rate,
        }

    @override_from(MochiEnv)
    def _check_stop_criteria(
        self,
        observation: StructuredObservation,
        action: StructuredAction,
        reward: RewardTerms,
        info: Info,
    ):
        MochiEnv._check_stop_criteria(
            self, observation=observation, action=action, reward=reward, info=info
        )
        if info["disturbed"]:
            info["terminated_reason"] = "Tower disturbed"
            self._terminated = True
        elif info["is_success"] and self._cfg.terminate_on_success:
            info["terminated_reason"] = "Success"
            self._terminated = True

    @override_from(MochiEnv)
    def _reset_renderer(self):
        # Wood tones varying a little from block to block, so the blocks stand apart.
        # The target level's are drawn alike: which one is loose is not visible.
        tones = np.random.default_rng(0).uniform(0.9, 1.08, len(self._layout))
        colors = {
            name: tuple(np.clip(np.asarray(BLOCK_RGB) * tone, 0.0, 1.0))
            for (level, _, name), tone in zip(self._layout, tones)
            if level != self._cfg.target_level
        }
        colors.update(
            {
                name: BLOCK_RGB
                for level, _, name in self._layout
                if level == self._cfg.target_level
            }
        )
        colors.update({"table": TABLE_RGB, "StaticPlane": FLOOR_RGB})
        apply_render_model_colors(self._renderer, self._scene, colors)


def _tower_layout(cfg: Fr3Revo2JengaEnvCfg) -> list[tuple[int, int, str]]:
    """(level, slot, name) of every block; the target level's are named tight0, tight1
    and loose, and their slots are placeholders."""
    layout = []
    for level in range(cfg.levels):
        if level == cfg.target_level:
            names = ("tight0", "tight1", "loose")
        else:
            names = ("0", "1", "2")
        for slot, suffix in enumerate(names):
            layout.append((level, slot, f"jenga_{level}_{suffix}"))
    return layout


def _level_of(name: str) -> int:
    return int(name.split("_")[1])


def _block_pose(
    cfg: Fr3Revo2JengaEnvCfg, level: int, slot: int, offset: np.ndarray, loose: bool
) -> physics.TransformRT:
    """A block resting in its level; levels alternate between running along x (the
    target level's direction) and along y. Slots go from -y (or -x) to +y (or +x)."""
    _, width, height = BLOCK_SIZE
    h = height - cfg.loose_gap if loose else height
    x, y = TOWER_XY[0] + offset[0], TOWER_XY[1] + offset[1]
    z = level * height + h / 2
    across = (slot - 1) * (width + SLOT_GAP)
    if (level - cfg.target_level) % 2 == 0:
        return physics.TransformRT(physics.Quaternion.identity(), [x, y + across, z])
    return physics.TransformRT(
        physics.Quaternion.rotation_z(np.pi / 2), [x + across, y, z]
    )


def _find_blocks(scene: physics.Scene) -> dict[str, physics.Actor]:
    blocks = {}

    def visit(actor):
        name = actor.get_name()
        if name.startswith("jenga_"):
            blocks[name] = actor

    scene.for_each_actor(visit)
    return blocks
