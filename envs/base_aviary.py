import numpy as np
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType

from .candy_function import compute_evader_reward, compute_chaser_reward
from config.configs import EnvConfig
from utils.Raycast import RaySensor

## Debug
import psutil


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
        config: EnvConfig | None = None,
        gui: bool = False,
    ):
        self.controlled_agent = controlled_agent
        self.config = config or EnvConfig()

        self.capture_radius = self.config.capture_radius
        self.goal_radius = self.config.goal_radius
        self.arena_xy = self.config.arena_xy
        self.arena_z_min = self.config.arena_z_min
        self.arena_z_max = self.config.arena_z_max
        self.opponent_speed = self.config.opponent_speed
        self.SPEED_LIMIT = self.config.speed_limit

        self.EPISODE_LEN_SEC = self.config.episode_len_sec

        self.prev_goal_dist = None
        self.prev_capture_dist = None

        self.goal_vis_id = None

        self.opponent_policy = None
        self.opponent_pool = None
        self.p_old = None

        self.reward_config = self.config.reward_config

        self.termination_stats = {
            "goal": 0,
            "captured": 0,
            "evader_out": 0,
            "chaser_out": 0,
            "timeout": 0,
        }

        self.use_ray_sensor = self.config.use_ray_sensor
        self.ray_include_hits = self.config.ray_include_hits

        self.ray_sensor = None
        self.ray_obs_dim = 0

        if self.use_ray_sensor:
            self.ray_sensor = RaySensor(
                num_rays=self.config.ray_num_rays,
                max_range=self.config.ray_max_range,
                use_3d=self.config.ray_use_3d,
                z_levels=self.config.ray_z_levels,
            )
            base_ray_dim = len(self.ray_sensor.local_dirs)
            self.ray_obs_dim = base_ray_dim * 2 if self.ray_include_hits else base_ray_dim

        self.ray_visualize = False
        self.draw_goal = False
        if gui:
            self.draw_goal = True
            if self.ray_sensor is not None:
                self.ray_visualize = True
            
        super().__init__(
            drone_model=self.config.drone_model,
            num_drones=2,
            physics=self.config.physics,
            pyb_freq=self.config.pyb_freq,
            ctrl_freq=self.config.ctrl_freq,
            gui=gui,
            record=self.config.record,
            obs=self.config.obs,
            act=self.config.act,
        )

        if self.config.act != ActionType.VEL:
            raise ValueError("This environment currently assumes ActionType.VEL")

        self.action_space = spaces.Box(
            low=np.full((4,), -1.0, dtype=np.float32),
            high=np.full((4,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        if self.controlled_agent == self.AGENT_EVADER:
            obs_dim = 12 + self.ray_obs_dim
        elif self.controlled_agent == self.AGENT_CHASER:
            obs_dim = 18 + self.ray_obs_dim
        else:
            raise ValueError(f"Unknown controlled_agent={self.controlled_agent}")

        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -1.0, dtype=np.float32),
            high=np.full((obs_dim,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # rest of your class stays the same

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

        self._create_arena_walls()
        self._draw_goal_marker()

        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        self.prev_goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))
        self.prev_capture_dist = float(np.linalg.norm(chaser_pos - evader_pos))

        return obs, info

    def step(self, action):
        joint_action = self._build_single_agent_action(action)
        obs, reward, terminated, truncated, info = super().step(joint_action)
        #Debug
        print(f"Available memory: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%")
            
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

    def _create_arena_walls(self):
        # Remove previous walls if they exist
        if self.ray_visualize:
            for wid in getattr(self, "wall_ids", []):
                try:
                    p.removeBody(wid, physicsClientId=self.CLIENT)
                except Exception:
                    pass
        self.wall_ids = []

        wall_thickness = 0.02
        z_center = 0.5 * (self.arena_z_min + self.arena_z_max)
        wall_height = 0.5 * (self.arena_z_max - self.arena_z_min)

        rgba = [0.7, 0.7, 0.7, 0.15]

        # +X wall
        self.wall_ids.append(self._create_box(
            half_extents=[wall_thickness, self.arena_xy, wall_height],
            position=[ self.arena_xy + wall_thickness, 0.0, z_center],
            rgba=rgba
        ))

        # -X wall
        self.wall_ids.append(self._create_box(
            half_extents=[wall_thickness, self.arena_xy, wall_height],
            position=[-self.arena_xy - wall_thickness, 0.0, z_center],
            rgba=rgba
        ))

        # +Y wall
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, wall_thickness, wall_height],
            position=[0.0,  self.arena_xy + wall_thickness, z_center],
            rgba=rgba
        ))

        # -Y wall
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, wall_thickness, wall_height],
            position=[0.0, -self.arena_xy - wall_thickness, z_center],
            rgba=rgba
        ))

        # Floor
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, self.arena_xy, wall_thickness],
            position=[0.0, 0.0, self.arena_z_min - wall_thickness],
            rgba=[0.5, 0.5, 0.5, 0.1]
        ))

        # Ceiling (optional)
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, self.arena_xy, wall_thickness],
            position=[0.0, 0.0, self.arena_z_max + wall_thickness],
            rgba=[0.5, 0.5, 0.5, 0.1]
        ))

    def _create_box(self, half_extents, position, rgba):
        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.CLIENT
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.CLIENT
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self.CLIENT
        )
        return body_id

    def _get_ray_obs(self, drone_index: int) -> np.ndarray:
        if not self.use_ray_sensor or self.ray_sensor is None:
            return np.array([], dtype=np.float32)

        body_id = self.DRONE_IDS[drone_index]

        ray_obs = self.ray_sensor.get_observation(
            drone_id=body_id,
            client_id=self.CLIENT,
            visualize=self.ray_visualize,
            ignore_body_ids={body_id},
            return_hits=self.ray_include_hits,
        ).astype(np.float32)

        return ray_obs

        # Only keep normalized distances
        n = len(self.ray_sensor.local_dirs)
        return ray_obs[:n]

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
            ray_obs = self._get_ray_obs(0)
            return np.concatenate([
                evader_pos,
                evader_vel/self.SPEED_LIMIT,
                rel,
                self.goal_pos*2/self.arena_xy,
                ray_obs,
            ]).astype(np.float32)

        if agent == self.AGENT_CHASER:
            ray_obs = self._get_ray_obs(1)
            return np.concatenate([
                chaser_pos,
                chaser_vel/self.SPEED_LIMIT,
                evader_pos,
                evader_vel/self.opponent_speed,
                rel,
                self.goal_pos*2/self.arena_xy,
                ray_obs,
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

        self.prev_goal_dist = goal_dist
        self.prev_capture_dist = capture_dist

        if self.controlled_agent == self.AGENT_EVADER:
            reward_evader = compute_evader_reward(
                goal_dist=goal_dist,
                prev_goal_dist=self.prev_goal_dist,
                Evader_pos=evader_pos,
                Chaser_pos=chaser_pos,
                info=info,
                cfg=self.reward_config,
            )
            return reward_evader
        elif self.controlled_agent == self.AGENT_CHASER:
            reward_chaser = compute_chaser_reward(
                E_2_C_distance=capture_dist,
                info=info,
                cfg=self.reward_config,
            )
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