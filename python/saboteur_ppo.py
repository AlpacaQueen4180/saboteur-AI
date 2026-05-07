from typing import Any, Dict, List, Tuple

import os
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

from miner_ppo import (
    MAX_ACTIONS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_EPS,
    VALUE_COEF,
    MOVE_TYPES,
    CARD_TYPES,
    DEVICE,
    SaboteurHttpEnv,
    ActionConditionedActorCritic,
    RolloutBuffer,
    compute_gae,
    state_to_tensors,
    role_of,
    safe_float,
    safe_int,
    get_nested,
    clamp,
)


LR = 1e-4
ENTROPY_COEF = 0.025
PPO_EPOCHS = 6
MINIBATCH_SIZE = 128


def best_sabotage_opportunity(actions: List[Dict[str, Any]]) -> float:
    best = 0.0

    for a in actions:
        t = a.get("type")

        if t == "PLAY_ROCKFALL":
            best = max(best, safe_float(a.get("remove_delta"), 0.0))

        elif t == "PLAY_PATH":
            delta = safe_float(a.get("delta_target_distance"), 0.0)
            best = max(best, -delta)

        elif t == "PLAY_PLAYER" and a.get("card_type") == "BLOCK":
            best = max(best, 1.0)

    return best


def compute_saboteur_reward(
    prev_state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    role = role_of(prev_state)
    assert role == "SABOTEUR", f"Saboteur PPO received non-saboteur role: {role}"

    done = bool(next_state.get("done", False))
    winner = next_state.get("winner", None)

    parts: Dict[str, float] = {
        "step": 0.0,
        "terminal": 0.0,
        "path": 0.0,
        "player_action": 0.0,
        "rockfall": 0.0,
        "map": 0.0,
        "fold": 0.0,
    }

    reward = 0.0

    if done and winner is not None:
        parts["terminal"] = 1.0 if winner == "SABOTEUR" else -1.0
        reward += parts["terminal"]

    move_type = action.get("type")
    card_type = action.get("card_type", action.get("cardType"))
    target_player = safe_int(action.get("target_player", action.get("targetPlayer", -1)), -1)

    delta = safe_float(action.get("delta_target_distance"), 0.0)
    remove_delta = safe_float(action.get("remove_delta"), 0.0)

    if move_type == "PLAY_PATH":
        before = safe_float(action.get("before_target_distance"), 0.0)

        if before >= 6.0:
            stage_weight = 0.60
        elif before >= 3.0:
            stage_weight = 0.90
        else:
            stage_weight = 1.20

        parts["path"] = -0.05 * stage_weight * delta
        reward += parts["path"]

    if move_type == "PLAY_PLAYER":
        own_idx = safe_int(get_nested(prev_state, ["observation", "private", "playerIndex"], 3), 3)

        if card_type == "BLOCK" and target_player != own_idx:
            parts["player_action"] = 0.06
        elif card_type == "REPAIR" and target_player == own_idx:
            parts["player_action"] = 0.04
        elif card_type == "REPAIR" and target_player != own_idx:
            parts["player_action"] = -0.05

        reward += parts["player_action"]

    if move_type == "PLAY_ROCKFALL":
        parts["rockfall"] = 0.05 * remove_delta
        reward += parts["rockfall"]

    if move_type == "PLAY_MAP":
        # Saboteur can use map, but it is not central. Small neutral-positive if unknown.
        known_goals = get_nested(
            prev_state,
            ["observation", "board", "path_features", "known_goals"],
            ["UNKNOWN", "UNKNOWN", "UNKNOWN"],
        )
        goal_index = safe_int(action.get("goal_index", action.get("goalIndex", -1)), -1)

        if 0 <= goal_index < 3 and known_goals[goal_index] == "UNKNOWN":
            parts["map"] = 0.005
        else:
            parts["map"] = -0.005

        reward += parts["map"]

    if move_type == "DISCARD":
        sabotage = best_sabotage_opportunity(prev_state.get("legalActions", []))

        if sabotage > 0.5:
            parts["fold"] = -min(0.08, 0.05 * sabotage)
        else:
            parts["fold"] = 0.0

        reward += parts["fold"]

    reward = clamp(reward, -1.05, 1.05)
    parts["total"] = reward
    return reward, parts


def print_debug(state: Dict[str, Any], action: Dict[str, Any], reward_parts: Dict[str, float]) -> None:
    print("\n[SABOTEUR DEBUG ACTION]")
    print(
        f"type={action.get('type')} card={action.get('card_name')} "
        f"hand={action.get('handIndex')} x={action.get('x')} y={action.get('y')} "
        f"rotated={action.get('rotated')} targetPlayer={action.get('target_player')} "
        f"goalIndex={action.get('goal_index')}"
    )
    print(
        f"targetDist={action.get('before_target_distance')} -> "
        f"{action.get('after_target_distance')} "
        f"delta={action.get('delta_target_distance')}"
    )
    print(
        f"removeDelta={action.get('remove_delta')} "
        f"idealFillDelta={action.get('ideal_fill_delta')}"
    )
    print("reward_parts:", {k: round(v, 4) for k, v in reward_parts.items()})


def train_saboteur(
    base_url: str,
    initial_state: Dict[str, Any],
    total_updates: int,
    rollout_steps: int,
    save_every: int,
    debug_every: int,
) -> None:
    env = SaboteurHttpEnv(base_url)

    state = initial_state
    if role_of(state) != "SABOTEUR":
        state = env.reset_until_role("SABOTEUR")

    obs, action_feats, mask = state_to_tensors(state)

    obs_dim = obs.shape[0]
    action_dim = action_feats.shape[1]

    print("Saboteur PPO training started.")
    print("device:", DEVICE)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("initial legal_actions:", int(mask.sum()))

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer([], [], [], [], [], [], [], [])

    episode_count = 0
    win_count = 0

    for update in range(1, total_updates + 1):
        buffer.clear()
        rollout_reward = 0.0
        steps_collected = 0

        while steps_collected < rollout_steps:
            if role_of(state) != "SABOTEUR":
                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)

            legal_actions = state.get("legalActions", [])
            if len(legal_actions) == 0:
                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
                continue

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action(obs_t, act_t, mask_t)

            action_id = int(action_t.item())
            if action_id >= len(legal_actions):
                action_id = 0

            selected_action = legal_actions[action_id]
            prev_state = state

            try:
                next_state = env.step(selected_action)
            except requests.HTTPError as e:
                print("[saboteur] HTTP step error:", e)
                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
                continue

            reward, reward_parts = compute_saboteur_reward(prev_state, selected_action, next_state)
            done = bool(next_state.get("done", False))

            buffer.obs.append(obs)
            buffer.action_feats.append(action_feats)
            buffer.masks.append(mask)
            buffer.actions.append(action_id)
            buffer.logprobs.append(float(logprob_t.item()))
            buffer.rewards.append(float(reward))
            buffer.dones.append(done)
            buffer.values.append(float(value_t.item()))

            rollout_reward += reward
            steps_collected += 1

            if update % debug_every == 0 and steps_collected == 1:
                print_debug(prev_state, selected_action, reward_parts)

            if done:
                episode_count += 1
                if next_state.get("winner") == "SABOTEUR":
                    win_count += 1

                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
            else:
                state = next_state
                obs, action_feats, mask = state_to_tensors(state)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, next_value_t = model.forward(obs_t, act_t)
            next_value = float(next_value_t.item())

        advantages, returns = compute_gae(buffer, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=DEVICE)
        act_batch = torch.tensor(np.asarray(buffer.action_feats), dtype=torch.float32, device=DEVICE)
        mask_batch = torch.tensor(np.asarray(buffer.masks), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)
        old_logprob_batch = torch.tensor(buffer.logprobs, dtype=torch.float32, device=DEVICE)
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret_batch = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        n = len(buffer.actions)
        indices = np.arange(n)

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx = indices[start:start + MINIBATCH_SIZE]

                new_logprob, entropy, value = model.evaluate_actions(
                    obs_batch[mb_idx],
                    act_batch[mb_idx],
                    mask_batch[mb_idx],
                    action_batch[mb_idx],
                )

                ratio = torch.exp(new_logprob - old_logprob_batch[mb_idx])
                surr1 = ratio * adv_batch[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_batch[mb_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - ret_batch[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))

        win_rate = win_count / max(1, episode_count)
        avg_step_reward = rollout_reward / max(1, steps_collected)

        print(
            f"[saboteur update {update:04d}] "
            f"episodes={episode_count} "
            f"rollout_reward={rollout_reward:.3f} "
            f"avg_step_reward={avg_step_reward:.5f} "
            f"policy_loss={np.mean(policy_losses):.5f} "
            f"value_loss={np.mean(value_losses):.5f} "
            f"entropy={np.mean(entropies):.5f} "
            f"saboteur_win={win_rate:.3f}"
        )

        if update % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/saboteur_ppo_http_update_{update}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "max_actions": MAX_ACTIONS,
                    "role": "SABOTEUR",
                    "move_types": MOVE_TYPES,
                    "card_types": CARD_TYPES,
                },
                ckpt_path,
            )
            print("saved checkpoint:", ckpt_path)