import time
import numpy as np
from stable_baselines3 import PPO

from envs.base_aviary import base_aviary
from envs.candy_function import (
    RewardConfig,
    compute_evader_reward,
    compute_chaser_reward,
)
from AMSPB_training import ScriptedChaserPolicy, ScriptedEvaderPolicy

## Debug
import psutil

# ============================================================================
# Default model paths
# Change these to your actual final saved models
# ============================================================================

FINAL_EVADER_MODEL = "./models/version_4/evader_seed_44_2026-03-13_13-17"
FINAL_CHASER_MODEL = "./models/version_4/chaser_seed_44_2026-03-13_13-19"

N_EPISODES = 10
MAX_STEPS = 2000


# ============================================================================
# Policy wrappers
# ============================================================================

class LoadedPPOPolicy:
    """
    Small wrapper so PPO models look like opponent policies
    with a .predict(obs, deterministic=True) interface.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


# ============================================================================
# Helpers
# ============================================================================

def make_scripted_opponent(role: str):
    """
    role = which side the OPPONENT controls
    """
    if role == "chaser":
        return ScriptedChaserPolicy(speed=1.0)
    elif role == "evader":
        return ScriptedEvaderPolicy(speed=1.0, w_goal=0.7, w_away=0.3)
    else:
        raise ValueError(f"Unknown scripted opponent role: {role}")


def classify_outcome(info: dict) -> str:
    if info.get("evader_reached_goal", False):
        return "goal"
    elif info.get("captured", False):
        return "captured"
    elif info.get("evader_out", False):
        return "evader_out"
    elif info.get("chaser_out", False):
        return "chaser_out"
    elif info.get("timeout", False):
        return "timeout"
    return "other"


def print_summary(name: str, episode_rewards, episode_lengths, outcome_counts):
    print(f"\n=== Summary: {name} ===")
    print(f"Mean reward: {np.mean(episode_rewards):.3f}")
    print(f"Std reward:  {np.std(episode_rewards):.3f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    print("Outcomes:")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v}")


# ============================================================================
# Main evaluation
# ============================================================================

def run_evaluation(
    role: str,                               # "evader" or "chaser"
    learner_model_path: str,
    opponent_policy,
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    gui: bool = True,
    record: bool = False,
    seed_offset: int = 0,
    reward_config: RewardConfig | None = None,
    device: str = "cuda",
):
    """
    Evaluate one learned policy in base_aviary against a supplied opponent policy.

    role="evader":
        learner controls evader, opponent controls chaser

    role="chaser":
        learner controls chaser, opponent controls evader
    """
    if role not in ["evader", "chaser"]:
        raise ValueError(f"Invalid role: {role}")

    controlled_agent = (
        base_aviary.AGENT_EVADER if role == "evader"
        else base_aviary.AGENT_CHASER
    )
    

    env = base_aviary(
        controlled_agent=controlled_agent,
        gui = True,
    )
    env.set_opponent_policy(opponent_policy)

    learner = PPO.load(learner_model_path, device=device)

    print(f"Evaluating {role} model: {learner_model_path}")

    episode_rewards = []
    episode_lengths = []

    # Optional: track both evader and chaser returns, even when only one is learned
    evader_episode_rewards = []
    chaser_episode_rewards = []

    outcome_counts = {
        "goal": 0,
        "captured": 0,
        "evader_out": 0,
        "chaser_out": 0,
        "timeout": 0,
        "other": 0,
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        evader_return = 0.0
        chaser_return = 0.0

        while not done and ep_len < max_steps:
            #Debug
            #print(f"Available memory: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%")
            # Save reward inputs exactly like env._computeReward() needs them
            prev_goal_distance = env.prev_goal_dist

            action, _ = learner.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_len += 1

            # Post-step state, matching env._computeReward()
            evader_pos = env._pos(0)
            chaser_pos = env._pos(1)

            goal_dist = float(info["goal_distance"])
            capture_dist = float(info["distance"])

            evader_step_reward = compute_evader_reward(
                goal_dist=goal_dist,
                prev_goal_dist=prev_goal_distance,
                Evader_pos=evader_pos,
                Chaser_pos=chaser_pos,
                info=info,
                cfg=env.reward_config,
            )

            chaser_step_reward = compute_chaser_reward(
                E_2_C_distance=capture_dist,
                info=info,
                cfg=env.reward_config,
            )

            evader_return += evader_step_reward
            chaser_return += chaser_step_reward

            if gui:
                time.sleep(env.CTRL_TIMESTEP)

            done = terminated or truncated

        outcome = classify_outcome(info)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        evader_episode_rewards.append(evader_return)
        chaser_episode_rewards.append(chaser_return)
        outcome_counts[outcome] += 1

        print(
            f"Episode {ep+1}/{n_episodes} | "
            f"learner_reward={ep_reward:.3f} | "
            f"evader_reward={evader_return:.3f} | "
            f"chaser_reward={chaser_return:.3f} | "
            f"length={ep_len} | "
            f"outcome={outcome}"
        )

    env.close()

    print_summary(
        name=f"{role} evaluation",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        outcome_counts=outcome_counts,
    )

    print(f"Mean evader return: {np.mean(evader_episode_rewards):.3f}")
    print(f"Mean chaser return: {np.mean(chaser_episode_rewards):.3f}")

    return {
        "learner_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "evader_rewards": evader_episode_rewards,
        "chaser_rewards": chaser_episode_rewards,
        "outcomes": outcome_counts,
    }


# ============================================================================
# Convenience wrappers
# ============================================================================

def evaluate_evader_vs_scripted_chaser(
    evader_model_path: str,
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    gui: bool = True,
    record: bool = False,
    reward_config: RewardConfig | None = None,
    device: str = "cpu",
):
    opponent = make_scripted_opponent("chaser")
    return run_evaluation(
        role="evader",
        learner_model_path=evader_model_path,
        opponent_policy=opponent,
        n_episodes=n_episodes,
        max_steps=max_steps,
        gui=gui,
        record=record,
        reward_config=reward_config,
        device=device,
    )


def evaluate_chaser_vs_scripted_evader(
    chaser_model_path: str,
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    gui: bool = True,
    record: bool = False,
    reward_config: RewardConfig | None = None,
    device: str = "cpu",
):
    opponent = make_scripted_opponent("evader")
    return run_evaluation(
        role="chaser",
        learner_model_path=chaser_model_path,
        opponent_policy=opponent,
        n_episodes=n_episodes,
        max_steps=max_steps,
        gui=gui,
        record=record,
        reward_config=reward_config,
        device=device,
    )


def evaluate_final_models_head_to_head(
    evader_model_path: str,
    chaser_model_path: str,
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    gui: bool = True,
    record: bool = False,
    reward_config: RewardConfig | None = None,
    device: str = "cpu",
):
    """
    Evaluate final evader vs final chaser.

    We run the environment in evader-controlled mode:
    - learner = evader PPO
    - opponent = chaser PPO

    This is enough because the env simulates both drones.
    """
    chaser_opponent = LoadedPPOPolicy(chaser_model_path, device=device)

    return run_evaluation(
        role="evader",
        learner_model_path=evader_model_path,
        opponent_policy=chaser_opponent,
        n_episodes=n_episodes,
        max_steps=max_steps,
        gui=gui,
        record=record,
        reward_config=reward_config,
        device=device,
    )


    def _pos(self, drone_id: int) -> np.ndarray:
        return self._getDroneStateVector(drone_id)[0:3].copy()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    reward_cfg = RewardConfig()

    # ------------------------------------------------------------------
    # Pick ONE of these
    # ------------------------------------------------------------------

    # 1) Evaluate final evader against scripted chaser
    # evaluate_evader_vs_scripted_chaser(
    #     evader_model_path=FINAL_EVADER_MODEL,
    #     n_episodes=10,
    #     max_steps=2000,
    #     gui=True,
    #     record=False,
    #     reward_config=reward_cfg,
    #     device="cpu",
    # )

    # 2) Evaluate final chaser against scripted evader
    # evaluate_chaser_vs_scripted_evader(
    #     chaser_model_path=FINAL_CHASER_MODEL,
    #     n_episodes=10,
    #     max_steps=2000,
    #     gui=True,
    #     record=False,
    #     reward_config=reward_cfg,
    #     device="cpu",
    # )

    # 3) Evaluate final evader vs final chaser
    evaluate_final_models_head_to_head(
        evader_model_path=FINAL_EVADER_MODEL,
        chaser_model_path=FINAL_CHASER_MODEL,
        n_episodes=10,
        max_steps=2000,
        gui=True,
        record=False,
        reward_config=reward_cfg,
        device="cuda",
    )