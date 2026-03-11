from dataclasses import dataclass
from stable_baselines3 import PPO


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
            self.model = PPO.load(self.path, device=self.device)

        return self.model.predict(obs, deterministic=deterministic)