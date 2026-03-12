import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    evader_goal_progress_weight: float = 1.0
    evader_escape_weight: float = 0.5
    chaser_capture_progress_weight: float = 0.5

    evader_goal_bonus: float = 100.0
    evader_captured_penalty: float = -100.0
    chaser_capture_bonus: float = 100.0
    chaser_goal_fail_penalty: float = -100.0

    evader_out_penalty: float = -50.0
    chaser_out_bonus_against_evader: float = 20.0

    chaser_out_penalty: float = -50.0
    evader_bonus_against_chaser_out: float = 20.0

    safe_rad: float = 1.0


def compute_evader_reward(
    goal_dist: float,
    prev_goal_dist: float | None,
    Evader_pos: float,
    Chaser_pos: float,
    info: dict,
    cfg: RewardConfig,
) -> float:
    reward = 0.0
    E_2_C_pos = np.linalg.norm(Evader_pos - Chaser_pos)

    if prev_goal_dist is not None:
        reward += cfg.evader_goal_progress_weight * (prev_goal_dist - goal_dist)

    if E_2_C_pos < cfg.safe_rad:
        reward += cfg.evader_escape_weight * (E_2_C_pos)

    if info["evader_reached_goal"]:
        reward += cfg.evader_goal_bonus

    if info["captured"]:
        reward += cfg.evader_captured_penalty

    if info["evader_out"]:
        reward += cfg.evader_out_penalty

    if info["chaser_out"]:
        reward += cfg.evader_bonus_against_chaser_out

    return float(reward)


def compute_chaser_reward(
    E_2_C_distance: float,
    info: dict,
    cfg: RewardConfig,
) -> float:
    reward = 0.0

    if E_2_C_distance < cfg.safe_rad:
        reward += cfg.chaser_capture_progress_weight * (cfg.safe_rad - E_2_C_distance)
        
    if info["captured"]:
        reward += cfg.chaser_capture_bonus

    if info["evader_reached_goal"]:
        reward += cfg.chaser_goal_fail_penalty

    if info["evader_out"]:
        reward += cfg.chaser_out_bonus_against_evader

    if info["chaser_out"]:
        reward += cfg.chaser_out_penalty

    return float(reward)