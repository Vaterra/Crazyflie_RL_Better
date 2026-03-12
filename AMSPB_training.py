import os
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.base_aviary import base_aviary
from envs.candy_function import RewardConfig

from policy_wrapper import PolicyEntry, LazyPPOPolicy
from Scripted_policies import ScriptedChaserPolicy, ScriptedEvaderPolicy
from Vec_envs:builder import build_vec_env


# =============================================================================
# Training helpers
# =============================================================================

def save_model(model: PPO, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    model.save(path)
    if not path.endswith(".zip"):
        path += ".zip"
    return path


def train_from(
    agent_role: str,                  # "evader" or "chaser"
    init_policy_path: str | None,
    opponent_pool: list,
    p_old: float,
    n_envs: int,
    total_timesteps: int,
    seed: int,
    reward_config: RewardConfig | None = None,
    save_dir: str = "./models",
    device: str = "cpu",
):
    if agent_role not in ["evader", "chaser"]:
        raise ValueError(f"Unknown agent_role={agent_role}")

    controlled_agent = (
        base_aviary.AGENT_EVADER if agent_role == "evader"
        else base_aviary.AGENT_CHASER
    )

    env = build_vec_env(
        controlled_agent=controlled_agent,
        n_envs=n_envs,
        seed=seed,
        opponent_pool=opponent_pool,
        p_old=p_old,
        reward_config=reward_config,
        gui=False,
        draw_goal=False,
    )

    tb_log = "./tb_logs_evader_pybullet" if agent_role == "evader" else "./tb_logs_chaser_pybullet"
    tb_name = f"{agent_role}_seed_{seed}_V_{Version}"

    if init_policy_path is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=256,
            gamma=0.99,
            device=device,
            tensorboard_log=tb_log,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(
            init_policy_path,
            env=env,
            device=device,
        )
        model.tensorboard_log = tb_log
        reset_num_timesteps = False

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_name,
        reset_num_timesteps=reset_num_timesteps,
    )

    save_name = f"{agent_role}_{tb_name}"
    save_path = save_model(model, save_dir, save_name)

    env.close()
    return save_path


# =============================================================================
# AMSPB
# =============================================================================

def AMSPB(
    N: int = 4,
    p_old: float = 0.5,
    n_envs: int = 32,
    timesteps_per_stage: int = 1_000_000,
    seed: int = 42,
    reward_config: RewardConfig | None = None,
    save_dir: str = "./models",
    device: str = "cpu",
    Version: str = "version_1",
):
    save_dir += Version
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # Initial scripted pools
    # Pi_P = pursuer/chaser pool
    # Pi_E = evader pool
    # -------------------------------------------------------------------------
    Pi_P = [
        PolicyEntry(
            policy=ScriptedChaserPolicy(speed=1.0),
            name="scripted_chaser",
            kind="scripted",
        )
    ]

    Pi_E = [
        PolicyEntry(
            policy=ScriptedEvaderPolicy(speed=1.0, w_goal=0.7, w_away=0.3),
            name="scripted_evader",
            kind="scripted",
        )
    ]

    # -------------------------------------------------------------------------
    # Stage 0: train first chaser against scripted evader pool
    # -------------------------------------------------------------------------
    print("\n=== Stage 0: Train chaser against scripted evader pool ===")
    pi_P_0_path = train_from(
        agent_role="chaser",
        init_policy_path=None,
        opponent_pool=[entry.policy for entry in Pi_E],
        p_old=p_old,
        n_envs=n_envs,
        total_timesteps=timesteps_per_stage,
        seed=seed,
        reward_config=reward_config,
        save_dir=save_dir,
        device=device,
    )

    Pi_P.append(
        PolicyEntry(
            policy=LazyPPOPolicy(pi_P_0_path, device=device),
            name="pi_P_0",
            kind="learned",
        )
    )

    # -------------------------------------------------------------------------
    # Stage 0: train first evader against current chaser pool
    # -------------------------------------------------------------------------
    print("\n=== Stage 0: Train evader against current chaser pool ===")
    pi_E_0_path = train_from(
        agent_role="evader",
        init_policy_path=None,
        opponent_pool=[entry.policy for entry in Pi_P],
        p_old=p_old,
        n_envs=n_envs,
        total_timesteps=timesteps_per_stage,
        seed=seed + 1,
        reward_config=reward_config,
        save_dir=save_dir,
        device=device,
    )

    Pi_E.append(
        PolicyEntry(
            policy=LazyPPOPolicy(pi_E_0_path, device=device),
            name="pi_E_0",
            kind="learned",
        )
    )

    prev_evader_path = pi_E_0_path
    prev_chaser_path = pi_P_0_path

    # -------------------------------------------------------------------------
    # Adversarial cross-training
    # -------------------------------------------------------------------------
    for k in range(1, N + 1):
        print(f"\n========== AMSPB Stage {k}/{N} ==========\n")

        print(f"Training evader pi_E_{k} from {prev_evader_path}")
        pi_E_k_path = train_from(
            agent_role="evader",
            init_policy_path=prev_evader_path,
            opponent_pool=[entry.policy for entry in Pi_P],
            p_old=p_old,
            n_envs=n_envs,
            total_timesteps=timesteps_per_stage,
            seed=seed + 10 * k,
            reward_config=reward_config,
            save_dir=save_dir,
            device=device,
        )

        Pi_E.append(
            PolicyEntry(
                policy=LazyPPOPolicy(pi_E_k_path, device=device),
                name=f"pi_E_{k}",
                kind="learned",
            )
        )

        print(f"Training chaser pi_P_{k} from {prev_chaser_path}")
        pi_P_k_path = train_from(
            agent_role="chaser",
            init_policy_path=prev_chaser_path,
            opponent_pool=[entry.policy for entry in Pi_E],
            p_old=p_old,
            n_envs=n_envs,
            total_timesteps=timesteps_per_stage,
            seed=seed + 10 * k + 1,
            reward_config=reward_config,
            save_dir=save_dir,
            device=device,
        )

        Pi_P.append(
            PolicyEntry(
                policy=LazyPPOPolicy(pi_P_k_path, device=device),
                name=f"pi_P_{k}",
                kind="learned",
            )
        )

        prev_evader_path = pi_E_k_path
        prev_chaser_path = pi_P_k_path

    print("\nAMSPB training complete.")
    return Pi_E, Pi_P


# =============================================================================
# Optional quick environment check
# =============================================================================

def quick_env_check():
    scripted_chaser = ScriptedChaserPolicy(speed=1.0)
    env = base_aviary(
        controlled_agent=base_aviary.AGENT_EVADER,
        gui=False,
        draw_goal=False,
        reward_config=RewardConfig(),
    )
    env.set_opponent_policy(scripted_chaser)
    check_env(env, warn=True)
    env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    reward_cfg = RewardConfig(
        evader_goal_progress_weight=1.0,
        evader_capture_escape_weight=0.5,
        chaser_capture_progress_weight=0.5,
        evader_goal_bonus=100.0,
        evader_captured_penalty=-100.0,
        chaser_capture_bonus=100.0,
        chaser_goal_fail_penalty=-100.0,
        evader_out_penalty=-50.0,
        chaser_out_bonus_against_evader=20.0,
        chaser_out_penalty=-50.0,
        evader_bonus_against_chaser_out=20.0,
    )

    AMSPB(
        N=4,
        p_old=0.5,
        n_envs=16,
        timesteps_per_stage=100_000,
        seed=42,
        reward_config=reward_cfg,
        save_dir="./models",
        device="cuda",   # switch to "cuda" after everything works reliably
        Version= "/version_1",
    )