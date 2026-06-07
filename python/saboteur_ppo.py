from typing import Any, Dict, Tuple, Optional

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
    clamp,
)


# ============================================================
# Saboteur PPO Config
# ============================================================

# Conservative PPO fine-tuning after supervised behavior cloning.
# The goal is to avoid destroying the pretrained BC policy.
LR = 1e-5
CLIP_EPS = 0.10
ENTROPY_COEF = 0.001
PPO_EPOCHS = 2
MINIBATCH_SIZE = 128

# KL regularization toward frozen BC policy.
# Larger = stay closer to BC.
# Smaller = allow more RL drift.
KL_COEF = 0.03


# ============================================================
# Saboteur Reward Config
# ============================================================

# Sparse terminal reward.
# The BC model already learned local behavior.
# RL is used only for conservative final-outcome fine-tuning.
STEP_REWARD = 0.0

SABOTEUR_WIN_REWARD = 1.0
SABOTEUR_LOSE_PENALTY = -1.0

# If the saboteur loses too early, give an additional small penalty.
EARLY_LOSE_STEP_THRESHOLD = 8
EARLY_LOSE_EXTRA_PENALTY = -0.3

REWARD_MIN = -1.5
REWARD_MAX = 1.2


# ============================================================
# Pretrained BC Policy Loader
# ============================================================

def load_checkpoint(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("Checkpoint path is empty.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=DEVICE)


def load_pretrained_policy(
    model: ActionConditionedActorCritic,
    pretrained_path: str,
    obs_dim: int,
    action_dim: int,
) -> Dict[str, Any]:
    """
    Load a supervised behavior cloning checkpoint into a trainable model.

    Expected checkpoint format:
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            ...
        }
    """
    ckpt = load_checkpoint(pretrained_path)

    ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
    ckpt_action_dim = int(ckpt.get("action_dim", action_dim))

    if ckpt_obs_dim != obs_dim or ckpt_action_dim != action_dim:
        raise ValueError(
            "Pretrained checkpoint dimension mismatch: "
            f"ckpt obs_dim={ckpt_obs_dim}, current obs_dim={obs_dim}; "
            f"ckpt action_dim={ckpt_action_dim}, current action_dim={action_dim}"
        )

    model.load_state_dict(ckpt["model"])

    print(f"[saboteur] loaded pretrained policy: {pretrained_path}")
    print(f"[saboteur] checkpoint source: {ckpt.get('source', 'unknown')}")
    print(f"[saboteur] best_val_acc: {ckpt.get('best_val_acc', 'unknown')}")

    return ckpt


def build_frozen_bc_policy(
    pretrained_path: str,
    obs_dim: int,
    action_dim: int,
) -> ActionConditionedActorCritic:
    """
    Build a frozen BC policy used as KL regularization target.
    This model is not updated during PPO.
    """
    frozen_model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)

    ckpt = load_checkpoint(pretrained_path)

    ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
    ckpt_action_dim = int(ckpt.get("action_dim", action_dim))

    if ckpt_obs_dim != obs_dim or ckpt_action_dim != action_dim:
        raise ValueError(
            "Frozen BC checkpoint dimension mismatch: "
            f"ckpt obs_dim={ckpt_obs_dim}, current obs_dim={obs_dim}; "
            f"ckpt action_dim={ckpt_action_dim}, current action_dim={action_dim}"
        )

    frozen_model.load_state_dict(ckpt["model"])
    frozen_model.eval()

    for p in frozen_model.parameters():
        p.requires_grad = False

    print(f"[saboteur] frozen BC policy loaded for KL regularization: {pretrained_path}")

    return frozen_model


# ============================================================
# KL Regularization
# ============================================================

def compute_kl_to_bc(
    current_model: ActionConditionedActorCritic,
    frozen_bc_model: ActionConditionedActorCritic,
    obs_batch: torch.Tensor,
    act_batch: torch.Tensor,
    mask_batch: torch.Tensor,
) -> torch.Tensor:
    """
    KL(BC || current)

    This penalizes current policy when it deviates too much from the frozen BC policy.
    Invalid actions are masked out.

    KL = sum_a pi_bc(a|s) * [log pi_bc(a|s) - log pi_current(a|s)]
    """
    with torch.no_grad():
        bc_scores, _ = frozen_bc_model(obs_batch, act_batch)
        bc_logits = bc_scores.masked_fill(mask_batch <= 0, -1e9)
        bc_log_probs = torch.log_softmax(bc_logits, dim=-1)
        bc_probs = torch.softmax(bc_logits, dim=-1)

    cur_scores, _ = current_model(obs_batch, act_batch)
    cur_logits = cur_scores.masked_fill(mask_batch <= 0, -1e9)
    cur_log_probs = torch.log_softmax(cur_logits, dim=-1)

    kl = torch.sum(
        bc_probs * (bc_log_probs - cur_log_probs),
        dim=-1,
    )

    return kl.mean()


# ============================================================
# Saboteur Reward
# ============================================================

def compute_saboteur_reward(
    prev_state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Dict[str, Any],
    episode_step: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Sparse terminal reward for BC + KL-regularized PPO fine-tuning.

    Design:
        - Non-terminal reward = 0.
        - If SABOTEUR wins, give +1.
        - If SABOTEUR loses, give -1.
        - If SABOTEUR loses very early, give additional small penalty.

    Reason:
        The BC model already learned useful local action preferences.
        PPO should optimize win/loss while KL prevents policy drift.
    """
    role = role_of(prev_state)
    assert role == "SABOTEUR", f"Saboteur PPO received non-saboteur role: {role}"

    done = bool(next_state.get("done", False))
    winner = next_state.get("winner", None)

    parts: Dict[str, float] = {
        "step": STEP_REWARD,
        "terminal": 0.0,
        "early_lose": 0.0,
        "total": 0.0,
    }

    reward = STEP_REWARD

    if done and winner is not None:
        if winner == "SABOTEUR":
            parts["terminal"] = SABOTEUR_WIN_REWARD
        else:
            parts["terminal"] = SABOTEUR_LOSE_PENALTY

            if episode_step <= EARLY_LOSE_STEP_THRESHOLD:
                parts["early_lose"] = EARLY_LOSE_EXTRA_PENALTY

        reward += parts["terminal"] + parts["early_lose"]

    reward = clamp(reward, REWARD_MIN, REWARD_MAX)
    parts["total"] = reward

    return reward, parts


# ============================================================
# Debug
# ============================================================

def print_debug(
    state: Dict[str, Any],
    action: Dict[str, Any],
    reward_parts: Dict[str, float],
    episode_step: int,
) -> None:
    print("\n[SABOTEUR DEBUG ACTION]")
    print(f"episode_step={episode_step}")
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
    print("reward_parts:", {k: round(v, 5) for k, v in reward_parts.items()})


# ============================================================
# Train Saboteur PPO
# ============================================================

def train_saboteur(
    base_url: str,
    initial_state: Dict[str, Any],
    total_updates: int,
    rollout_steps: int,
    save_every: int,
    debug_every: int,
    pretrained_path: str = "",
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
    print("pretrained_path:", pretrained_path if pretrained_path else "None")
    print("reward_mode: sparse terminal win/loss reward")
    print("regularization: KL to frozen BC policy" if pretrained_path else "regularization: disabled")
    print("LR:", LR)
    print("CLIP_EPS:", CLIP_EPS)
    print("ENTROPY_COEF:", ENTROPY_COEF)
    print("PPO_EPOCHS:", PPO_EPOCHS)
    print("KL_COEF:", KL_COEF)

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)

    frozen_bc_model: Optional[ActionConditionedActorCritic] = None

    if pretrained_path:
        load_pretrained_policy(
            model=model,
            pretrained_path=pretrained_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        frozen_bc_model = build_frozen_bc_policy(
            pretrained_path=pretrained_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer([], [], [], [], [], [], [], [])

    episode_count = 0
    win_count = 0

    current_episode_step = 0

    for update in range(1, total_updates + 1):
        buffer.clear()

        rollout_reward = 0.0
        rollout_step_reward = 0.0
        rollout_terminal_reward = 0.0
        rollout_early_lose_reward = 0.0

        steps_collected = 0

        while steps_collected < rollout_steps:
            if role_of(state) != "SABOTEUR":
                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
                current_episode_step = 0

            legal_actions = state.get("legalActions", [])
            if len(legal_actions) == 0:
                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
                current_episode_step = 0
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
                current_episode_step = 0
                continue

            current_episode_step += 1

            reward, reward_parts = compute_saboteur_reward(
                prev_state=prev_state,
                action=selected_action,
                next_state=next_state,
                episode_step=current_episode_step,
            )

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
            rollout_step_reward += reward_parts.get("step", 0.0)
            rollout_terminal_reward += reward_parts.get("terminal", 0.0)
            rollout_early_lose_reward += reward_parts.get("early_lose", 0.0)

            steps_collected += 1

            if update % debug_every == 0 and steps_collected == 1:
                print_debug(
                    state=prev_state,
                    action=selected_action,
                    reward_parts=reward_parts,
                    episode_step=current_episode_step,
                )

            if done:
                episode_count += 1

                if next_state.get("winner") == "SABOTEUR":
                    win_count += 1

                state = env.reset_until_role("SABOTEUR")
                obs, action_feats, mask = state_to_tensors(state)
                current_episode_step = 0
            else:
                state = next_state
                obs, action_feats, mask = state_to_tensors(state)

        # Bootstrap next value.
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
        kl_losses = []
        total_losses = []

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

                if frozen_bc_model is not None:
                    kl_loss = compute_kl_to_bc(
                        current_model=model,
                        frozen_bc_model=frozen_bc_model,
                        obs_batch=obs_batch[mb_idx],
                        act_batch=act_batch[mb_idx],
                        mask_batch=mask_batch[mb_idx],
                    )
                else:
                    kl_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

                loss = (
                    policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy_loss
                    + KL_COEF * kl_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                kl_losses.append(float(kl_loss.item()))
                total_losses.append(float(loss.item()))

        win_rate = win_count / max(1, episode_count)
        avg_step_reward = rollout_reward / max(1, steps_collected)
        avg_terminal_reward = rollout_terminal_reward / max(1, steps_collected)
        avg_early_lose_reward = rollout_early_lose_reward / max(1, steps_collected)

        print(
            f"[saboteur update {update:04d}] "
            f"episodes={episode_count} "
            f"rollout_reward={rollout_reward:.3f} "
            f"avg_step_reward={avg_step_reward:.5f} "
            f"avg_terminal_reward={avg_terminal_reward:.5f} "
            f"avg_early_lose_reward={avg_early_lose_reward:.5f} "
            f"policy_loss={np.mean(policy_losses):.5f} "
            f"value_loss={np.mean(value_losses):.5f} "
            f"entropy={np.mean(entropies):.5f} "
            f"kl_to_bc={np.mean(kl_losses):.5f} "
            f"total_loss={np.mean(total_losses):.5f} "
            f"saboteur_win={win_rate:.3f}"
        )

        if update % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)

            prefix = "saboteur_bc_ppo_kl" if pretrained_path else "saboteur_ppo_no_kl"
            ckpt_path = f"checkpoints/{prefix}_http_update_{update}.pt"

            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "max_actions": MAX_ACTIONS,
                    "role": "SABOTEUR",
                    "move_types": MOVE_TYPES,
                    "card_types": CARD_TYPES,
                    "source": "bc_ppo_kl_regularized_finetune" if pretrained_path else "ppo_from_scratch_no_kl",
                    "pretrained_path": pretrained_path,
                    "reward_mode": "sparse_terminal_win_loss_reward",
                    "regularization": "kl_to_frozen_bc_policy" if pretrained_path else "none",
                    "reward_config": {
                        "STEP_REWARD": STEP_REWARD,
                        "SABOTEUR_WIN_REWARD": SABOTEUR_WIN_REWARD,
                        "SABOTEUR_LOSE_PENALTY": SABOTEUR_LOSE_PENALTY,
                        "EARLY_LOSE_STEP_THRESHOLD": EARLY_LOSE_STEP_THRESHOLD,
                        "EARLY_LOSE_EXTRA_PENALTY": EARLY_LOSE_EXTRA_PENALTY,
                    },
                    "ppo_config": {
                        "LR": LR,
                        "CLIP_EPS": CLIP_EPS,
                        "ENTROPY_COEF": ENTROPY_COEF,
                        "PPO_EPOCHS": PPO_EPOCHS,
                        "MINIBATCH_SIZE": MINIBATCH_SIZE,
                        "GAMMA": GAMMA,
                        "GAE_LAMBDA": GAE_LAMBDA,
                        "VALUE_COEF": VALUE_COEF,
                        "KL_COEF": KL_COEF,
                    },
                    "update": update,
                    "episode_count": episode_count,
                    "win_count": win_count,
                    "win_rate": win_rate,
                },
                ckpt_path,
            )

            print("saved checkpoint:", ckpt_path)