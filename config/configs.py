from dataclasses import dataclass, field

from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)

from envs.candy_function import RewardConfig


@dataclass
class EnvConfig:
    # Base simulator settings
    drone_model: DroneModel = DroneModel.CF2X
    physics: Physics = Physics.PYB
    pyb_freq: int = 240
    ctrl_freq: int = 30
    gui: bool = False
    record: bool = False
    obs: ObservationType = ObservationType.KIN
    act: ActionType = ActionType.VEL

    # Environment parameters
    episode_len_sec: float = 15.0
    capture_radius: float = 0.25
    goal_radius: float = 0.35
    arena_xy: float = 3.0
    arena_z_min: float = 0.1
    arena_z_max: float = 3.0
    opponent_speed: float = 1.0
    speed_limit: float = 1.0
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Ray sensor parameters
    use_ray_sensor: bool = True
    ray_num_rays: int = 4
    ray_max_range: float = 3.0
    ray_use_3d: bool = False
    ray_z_levels: list[float] = field(default_factory=lambda: [0.0])
    ray_include_hits: bool = True