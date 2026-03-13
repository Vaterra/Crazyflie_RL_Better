from dataclasses import dataclass
from datetime import datetime

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

@dataclass
class TrainConfig:
    n_envs: int = 8
    total_timesteps: int = 100_000
    seed: int = 42
    device: str = "cuda"
    verbose: int = 1

    # PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 256
    gamma: float = 0.99

    # AMSPB parameters
    N: int = 1
    p_old: float = 0.5
    # Output directories
    Version: str = "version_4"
    save_dir: str = "./models/"+Version
    tb_root: str = "./tb_logs"