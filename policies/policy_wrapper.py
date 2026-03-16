from dataclasses import dataclass
from stable_baselines3 import PPO
## Debug
import psutil

@dataclass
class PolicyEntry:
    policy: object
    name: str
    kind: str


class LazyPPOPolicy:

    def __init__(self, path, device="cuda"):

        self.path = path
        self.device = device
        self.model = None

    def predict(self, obs, deterministic=True):


        if self.model is None:
            print(f"Available memory: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%")
            self.model = PPO.load(self.path, device=self.device)

        return self.model.predict(obs, deterministic=deterministic)