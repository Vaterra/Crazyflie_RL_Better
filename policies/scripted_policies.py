import numpy as np


def unit_vec(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def chase_target(chaser_pos, target_pos, speed=1.0):
    direction = unit_vec(target_pos - chaser_pos)
    return np.array([direction[0], direction[1], direction[2], speed], dtype=np.float32)


def flee_from_target(evader_pos, chaser_pos, goal_pos,
                     w_goal=0.7, w_away=0.3, speed=1.0):

    to_goal = unit_vec(goal_pos - evader_pos)
    away = unit_vec(evader_pos - chaser_pos)
    direction = unit_vec(w_goal * to_goal + w_away * away)

    return np.array([direction[0], direction[1], direction[2], speed], dtype=np.float32)


class ScriptedChaserPolicy:

    def __init__(self, speed=1.0):
        self.speed = speed

    def predict(self, obs, deterministic=True):

        chaser_pos = obs[0:3]
        evader_pos = obs[6:9]

        action = chase_target(chaser_pos, evader_pos, self.speed)

        return action, None


class ScriptedEvaderPolicy:
    def __init__(self, speed=1.0, w_goal=0.7, w_away=0.3):
        self.speed = speed
        self.w_goal = w_goal
        self.w_away = w_away

    def predict(self, obs, deterministic=True):
        evader_pos = obs[0:3]
        rel = obs[6:9]
        goal = obs[9:12]
        chaser_pos = evader_pos + rel

        action = flee_from_target(
            evader_pos,
            chaser_pos,
            goal,
            w_goal=self.w_goal,
            w_away=self.w_away,
            speed=self.speed,
        )
        return action, None