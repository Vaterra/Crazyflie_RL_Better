from dataclasses import dataclass
from envs.candy_function import RewardConfig


@dataclass
class EnvConfig:
    # Environment geometry
    arena_xy: float = 3.0
    arena_z_min: float = 0.1
    arena_z_max: float = 3.0

    # Game rules
    capture_radius: float = 0.25
    goal_radius: float = 0.35
    episode_len_sec: float = 30.0

    # Drone behaviour
    opponent_speed: float = 1.0
    speed_limit: float = 1.0

    # Visualization
    gui: bool = False
    draw_goal: bool = False

    # Reward configuration
    reward_config: RewardConfig = RewardConfig()