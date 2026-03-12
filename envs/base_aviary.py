import numpy as np
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from .candy_function import RewardConfig, compute_evader_reward, compute_chaser_reward


class base_aviary(BaseRLAviary):
    """
    Two-drone pursuit-evasion game.

    Drone indexing:
    - drone 0 = evader
    - drone 1 = chaser

    Single-agent RL interface:
    - controlled_agent = "evader": learner controls evader
    - controlled_agent = "chaser": learner controls chaser

    The other drone is controlled by an injected opponent policy.
    """

    AGENT_EVADER = "evader"
    AGENT_CHASER = "chaser"

    def __init__(
        self,
        controlled_agent: str,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,

        # Environment parameters
        episode_len_sec: float = 30.0,
        capture_radius: float = 0.25,
        goal_radius: float = 0.35,
        arena_xy: float = 3.0,
        arena_z_min: float = 0.1,
        arena_z_max: float = 3.0,
        opponent_speed: float = 1.0,
        draw_goal: bool = False,
        reward_config: RewardConfig | None = None,
    ):
        self.controlled_agent = controlled_agent

        self.capture_radius = capture_radius
        self.goal_radius = goal_radius
        self.arena_xy = arena_xy
        self.arena_z_min = arena_z_min
        self.arena_z_max = arena_z_max
        self.opponent_speed = opponent_speed

        self.EPISODE_LEN_SEC = episode_len_sec

        self.prev_goal_dist = None
        self.prev_capture_dist = None

        self.draw_goal = draw_goal
        self.goal_vis_id = None

        self.opponent_policy = None
        self.opponent_pool = None
        self.p_old = None

        self.reward_config = reward_config if reward_config is not None else RewardConfig()

        self.termination_stats = {
            "goal": 0,
            "captured": 0,
            "evader_out": 0,
            "chaser_out": 0,
            "timeout": 0,
        }

        super().__init__(
            drone_model=drone_model,
            num_drones=2,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        if act != ActionType.VEL:
            raise ValueError("This environment currently assumes ActionType.VEL")

        self.SPEED_LIMIT = 1.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        if self.controlled_agent == self.AGENT_EVADER:
            # evader_pos(3), evader_vel(3), rel(3), goal(3)
            obs_dim = 12
        elif self.controlled_agent == self.AGENT_CHASER:
            # chaser_pos(3), chaser_vel(3), evader_pos(3), evader_vel(3), rel(3), goal(3)
            obs_dim = 18
        else:
            raise ValueError(f"Unknown controlled_agent={self.controlled_agent}")

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # Opponent injection
    # ---------------------------------------------------------------------

    def set_opponent_policy(self, policy):
        self.opponent_policy = policy
        self.opponent_pool = None
        self.p_old = None

    def set_opponent_pool(self, pool, p_old: float):
        self.opponent_pool = pool
        self.p_old = p_old
        self.opponent_policy = None

    def _sample_opponent(self):
        if self.opponent_pool is None:
            return

        if len(self.opponent_pool) == 0:
            raise ValueError("Opponent pool is empty")

        if len(self.opponent_pool) == 1:
            self.opponent_policy = self.opponent_pool[-1]
            return

        if self.rng.random() < self.p_old:
            idx = self.rng.integers(0, len(self.opponent_pool) - 1)
            self.opponent_policy = self.opponent_pool[idx]
        else:
            self.opponent_policy = self.opponent_pool[-1]

    # ---------------------------------------------------------------------
    # Reset / step
    # ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, "rng"):
            self.rng = np.random.default_rng()

        if self.opponent_pool is not None:
            self._sample_opponent()

        if self.opponent_policy is None:
            raise ValueError("No opponent policy or opponent pool has been set")

        self.goal_pos = self._sample_goal()
        self.INIT_XYZS = self._sample_initial_positions()

        obs, info = super().reset(seed=seed, options=options)

        self._draw_goal_marker()

        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        self.prev_goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))
        self.prev_capture_dist = float(np.linalg.norm(chaser_pos - evader_pos))

        return obs, info

    def step(self, action):
        joint_action = self._build_single_agent_action(action)
        obs, reward, terminated, truncated, info = super().step(joint_action)

        if terminated or truncated:
            if info["evader_reached_goal"]:
                self.termination_stats["goal"] += 1
            elif info["captured"]:
                self.termination_stats["captured"] += 1
            elif info["evader_out"]:
                self.termination_stats["evader_out"] += 1
            elif info["chaser_out"]:
                self.termination_stats["chaser_out"] += 1
            elif info["timeout"]:
                self.termination_stats["timeout"] += 1

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _pos(self, drone_id: int) -> np.ndarray:
        return self._getDroneStateVector(drone_id)[0:3].copy()

    def _vel(self, drone_id: int) -> np.ndarray:
        return self._getDroneStateVector(drone_id)[10:13].copy()

    def _is_out(self, pos: np.ndarray) -> bool:
        return (
            abs(pos[0]) > self.arena_xy
            or abs(pos[1]) > self.arena_xy
            or pos[2] < self.arena_z_min
            or pos[2] > self.arena_z_max
        )

    def _get_agent_obs(self, agent: str) -> np.ndarray:
        evader_pos = self._pos(0)*2/self.arena_xy
        evader_vel = self._vel(0)
        chaser_pos = self._pos(1)*2/self.arena_xy
        chaser_vel = self._vel(1)
        rel = chaser_pos - evader_pos

        if agent == self.AGENT_EVADER:
            return np.concatenate([
                evader_pos,
                evader_vel/self.SPEED_LIMIT,
                rel,
                self.goal_pos*2/self.arena_xy,
            ]).astype(np.float32)

        if agent == self.AGENT_CHASER:
            return np.concatenate([
                chaser_pos,
                chaser_vel/self.SPEED_LIMIT,
                evader_pos,
                evader_vel/self.opponent_speed,
                rel,
                self.goal_pos*2/self.arena_xy,
            ]).astype(np.float32)

        raise ValueError(f"Unknown agent={agent}")

    def _build_single_agent_action(self, agent_action: np.ndarray) -> np.ndarray:
        agent_action = np.asarray(agent_action, dtype=np.float32).reshape(4,)

        if self.opponent_policy is None:
            raise ValueError("Opponent policy not set")

        if self.controlled_agent == self.AGENT_EVADER:
            evader_action = agent_action
            opponent_obs = self._get_agent_obs(self.AGENT_CHASER)
            chaser_action, _ = self.opponent_policy.predict(opponent_obs, deterministic=True)

        elif self.controlled_agent == self.AGENT_CHASER:
            chaser_action = agent_action
            opponent_obs = self._get_agent_obs(self.AGENT_EVADER)
            evader_action, _ = self.opponent_policy.predict(opponent_obs, deterministic=True)

        else:
            raise ValueError(f"Unknown controlled_agent={self.controlled_agent}")

        evader_action = np.asarray(evader_action, dtype=np.float32).reshape(4,)
        chaser_action = np.asarray(chaser_action, dtype=np.float32).reshape(4,)

        return np.vstack([evader_action, chaser_action]).astype(np.float32)

    # ---------------------------------------------------------------------
    # RL API hooks
    # ---------------------------------------------------------------------

    def _computeObs(self):
        return self._get_agent_obs(self.controlled_agent)

    def _computeInfo(self):
        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        capture_dist = float(np.linalg.norm(chaser_pos - evader_pos))
        goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))

        captured = capture_dist <= self.capture_radius
        evader_reached_goal = goal_dist <= self.goal_radius
        evader_out = self._is_out(evader_pos)
        chaser_out = self._is_out(chaser_pos)
        timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC

        return {
            "captured": captured,
            "evader_reached_goal": evader_reached_goal,
            "evader_out": evader_out,
            "chaser_out": chaser_out,
            "timeout": timeout,
            "distance": capture_dist,
            "goal_distance": goal_dist,
        }

    def _computeTerminated(self):
        info = self._computeInfo()
        return (
            info["captured"]
            or info["evader_reached_goal"]
            or info["evader_out"]
            or info["chaser_out"]
        )  ## Maybe change the termination depending on which agent we are training

    def _computeTruncated(self):
        info = self._computeInfo()
        return info["timeout"]

    def _computeReward(self):
        info = self._computeInfo()

        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))
        capture_dist = info["distance"]

        reward_evader = compute_evader_reward(
            goal_dist=goal_dist,
            prev_goal_dist=self.prev_goal_dist,
            Evader_pos=evader_pos,
            Chaser_pos=chaser_pos,
            info=info,
            cfg=self.reward_config,
        )

        reward_chaser = compute_chaser_reward(
            capture_dist=capture_dist,
            prev_capture_dist=self.prev_capture_dist,
            info=info,
            cfg=self.reward_config,
        )

        self.prev_goal_dist = goal_dist
        self.prev_capture_dist = capture_dist

        if self.controlled_agent == self.AGENT_EVADER:
            return reward_evader
        elif self.controlled_agent == self.AGENT_CHASER:
            return reward_chaser

        raise ValueError(f"Unknown controlled_agent={self.controlled_agent}")

    # ---------------------------------------------------------------------
    # Sampling / visualization
    # ---------------------------------------------------------------------

    def _sample_initial_positions(self) -> np.ndarray:
        rng = self.rng

        evader_pos = np.array([
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(-self.arena_xy + 2 * self.goal_radius, -self.arena_xy + 4 * self.goal_radius),
            rng.uniform(self.goal_radius * 2, self.arena_z_max - self.goal_radius * 2),
        ], dtype=np.float32)

        chaser_pos = np.array([
            rng.choice([
                rng.uniform(-self.arena_xy, -self.arena_xy + self.goal_radius),
                rng.uniform(self.arena_xy - self.goal_radius, self.arena_xy),
            ]),
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(self.arena_z_min, self.arena_z_max),
        ], dtype=np.float32)

        return np.vstack([evader_pos, chaser_pos])

    def _sample_goal(self) -> np.ndarray:
        rng = self.rng

        return np.array([
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(self.arena_xy - 3 * self.goal_radius, self.arena_xy - self.goal_radius),
            rng.uniform(self.arena_z_min, self.arena_z_max),
        ], dtype=np.float32)

    def _draw_goal_marker(self):
        if not self.draw_goal:
            return

        if self.goal_vis_id is not None:
            try:
                p.removeBody(self.goal_vis_id, physicsClientId=self.CLIENT)
            except Exception:
                pass
            self.goal_vis_id = None

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.goal_radius,
            rgbaColor=[1, 0, 0, 0.35],
            physicsClientId=self.CLIENT,
        )

        self.goal_vis_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.goal_pos.tolist(),
            physicsClientId=self.CLIENT,
        )