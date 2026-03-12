from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.base_aviary import base_aviary


def make_env(rank, controlled_agent, seed, opponent_pool, p_old):

    def _init():

        env = base_aviary(controlled_agent=controlled_agent)

        env.set_opponent_pool(opponent_pool, p_old)

        env.reset(seed=seed + rank)

        return env

    return _init


def build_vec_env(controlled_agent, n_envs, seed, opponent_pool, p_old):

    env_fns = [

        make_env(i, controlled_agent, seed, opponent_pool, p_old)

        for i in range(n_envs)
    ]

    return VecMonitor(SubprocVecEnv(env_fns))