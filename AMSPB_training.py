import os
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.base_aviary import base_aviary
from envs.candy_function import RewardConfig

from policies.policy_wrapper import PolicyEntry, LazyPPOPolicy
from policies.scripted_policies import ScriptedChaserPolicy, ScriptedEvaderPolicy
from utils.vec_env_builder import build_vec_env
from config.configs import EnvConfig
from config.train_config import TrainConfig, timestamp


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
    training: TrainConfig,
    reward_config: RewardConfig,
    seed_input: int,
):
    
    if agent_role not in ["evader", "chaser"]:
        raise ValueError(f"Unknown agent_role={agent_role}")

    controlled_agent = (
        base_aviary.AGENT_EVADER if agent_role == "evader"
        else base_aviary.AGENT_CHASER
    )

    env = build_vec_env(
        controlled_agent=controlled_agent,
        n_envs=training.n_envs,
        seed= seed_input,
        opponent_pool=opponent_pool,
        p_old=training.p_old,
    )

    #Tensorboard logging setup
    tb_log = os.path.join(training.tb_root, agent_role)
    tb_name = f"{agent_role}_seed_{seed_input}"

    if init_policy_path is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=training.verbose,
            learning_rate=training.learning_rate,
            n_steps=training.n_steps,
            batch_size=training.batch_size,
            gamma=training.gamma,
            device=training.device,
            tensorboard_log=tb_log,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(
            init_policy_path,
            env=env,
            device=training.device,
        )
        model.tensorboard_log = tb_log
        reset_num_timesteps = False

    model.learn(
        total_timesteps=training.total_timesteps,
        tb_log_name=tb_name,
        reset_num_timesteps=reset_num_timesteps,
    )


    #Saving
    run_stamp = timestamp()
    save_name = f"{agent_role}_seed_{seed_input}_{run_stamp}"
    save_path = save_model(model, training.save_dir, save_name)

    env.close()
    return save_path

# =============================================================================
# AMSPB
# =============================================================================

def AMSPB(
    train_cfg: TrainConfig,
    env_config: EnvConfig,
    reward_config: RewardConfig,
):

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    evader_seed = train_cfg.seed
    chaser_seed = train_cfg.seed  + 1

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
        training=train_cfg,
        reward_config=reward_config,
        seed=chaser_seed,

    )

    Pi_P.append(
        PolicyEntry(
            policy=LazyPPOPolicy(pi_P_0_path, device=train_cfg.device),
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
        training=train_cfg,
        reward_config=reward_config,
        seed=evader_seed,
    )

    Pi_E.append(
        PolicyEntry(
            policy=LazyPPOPolicy(pi_E_0_path, device=train_cfg.device),
            name="pi_E_0",
            kind="learned",
        )
    )

    prev_evader_path = pi_E_0_path
    prev_chaser_path = pi_P_0_path

    # -------------------------------------------------------------------------
    # Adversarial cross-training
    # -------------------------------------------------------------------------
    for k in range(1, train_cfg.N + 1):
        print(f"\n========== AMSPB Stage {k}/{train_cfg.N} ==========\n")

        print(f"Training evader pi_E_{k} from {prev_evader_path}")
        pi_E_k_path = train_from(
            agent_role="evader",
            init_policy_path=prev_evader_path,
            opponent_pool=[entry.policy for entry in Pi_P],
            training=train_cfg,
            reward_config=reward_config,
            seed=evader_seed + 2*k,
        )

        Pi_E.append(
            PolicyEntry(
                policy=LazyPPOPolicy(pi_E_k_path, device=train_cfg.device),
                name=f"pi_E_{k}",
                kind="learned",
            )
        )

        print(f"Training chaser pi_P_{k} from {prev_chaser_path}")
        pi_P_k_path = train_from(
            agent_role="chaser",
            init_policy_path=prev_chaser_path,
            opponent_pool=[entry.policy for entry in Pi_E],
            training=train_cfg,
            reward_config=reward_config,
            seed=chaser_seed + 2*k-1,
        )

        Pi_P.append(
            PolicyEntry(
                policy=LazyPPOPolicy(pi_P_k_path, device=train_cfg.device),
                name=f"pi_P_{k}",
                kind="learned",
            )
        )

        prev_evader_path = pi_E_k_path
        prev_chaser_path = pi_P_k_path

    print("\nAMSPB training complete.")
    return Pi_E, Pi_P
# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    reward_cfg = RewardConfig()
    Settings = TrainConfig()
    Env = EnvConfig
    AMSPB(train_cfg=Settings, env_config=Env, reward_config=reward_cfg)