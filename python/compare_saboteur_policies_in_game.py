import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from miner_ppo import (
    DEVICE,
    SaboteurHttpEnv,
    ActionConditionedActorCritic,
    state_to_tensors,
    role_of,
    get_nested,
    safe_float,
    safe_int,
)


# ============================================================
# Load Model
# ============================================================

def load_policy(checkpoint_path: str) -> Tuple[ActionConditionedActorCritic, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, ckpt


# ============================================================
# Action Scoring
# ============================================================

def get_policy_ranking(
    model: ActionConditionedActorCritic,
    state: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    legal_actions = state.get("legalActions", [])

    if len(legal_actions) == 0:
        return []

    obs, action_feats, mask = state_to_tensors(state)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        scores, _ = model.forward(obs_t, act_t)
        masked_scores = scores.masked_fill(mask_t <= 0, -1e9)

        probs = torch.softmax(masked_scores, dim=-1)[0].detach().cpu().numpy()
        logits = masked_scores[0].detach().cpu().numpy()

    valid_n = len(legal_actions)
    valid_indices = np.arange(valid_n)
    ranked = sorted(valid_indices, key=lambda i: logits[i], reverse=True)

    result = []

    for rank, idx in enumerate(ranked[:top_k], start=1):
        action = legal_actions[int(idx)]

        result.append({
            "rank": rank,
            "action_index": int(idx),
            "prob": float(probs[idx]),
            "logit": float(logits[idx]),
            "action": action,
            "summary": summarize_action(action),
        })

    return result


def select_top1_action(
    model: ActionConditionedActorCritic,
    state: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    ranking = get_policy_ranking(model, state, top_k=1)

    if not ranking:
        raise RuntimeError("No legal actions available.")

    idx = ranking[0]["action_index"]
    action = state["legalActions"][idx]
    return idx, action


# ============================================================
# Formatting
# ============================================================

def summarize_action(action: Dict[str, Any]) -> str:
    move_type = action.get("type", action.get("move_type", "UNKNOWN"))
    card_name = action.get("card_name", action.get("cardName", ""))
    card_type = action.get("card_type", action.get("cardType", ""))

    hand_index = action.get("handIndex", action.get("hand_index", None))

    x = action.get("x", None)
    y = action.get("y", None)
    rotated = action.get("rotated", None)

    target_player = action.get("target_player", action.get("targetPlayer", None))
    goal_index = action.get("goal_index", action.get("goalIndex", None))

    delta = safe_float(
        action.get("delta_target_distance", action.get("deltaTargetDistance", 0.0)),
        0.0,
    )
    remove_delta = safe_float(
        action.get("remove_delta", action.get("removeDelta", 0.0)),
        0.0,
    )
    ideal_delta = safe_float(
        action.get("ideal_fill_delta", action.get("idealFillDelta", 0.0)),
        0.0,
    )

    parts = [f"type={move_type}"]

    if card_name:
        parts.append(f"card={card_name}")
    elif card_type:
        parts.append(f"cardType={card_type}")

    if hand_index is not None:
        parts.append(f"hand={hand_index}")

    if move_type == "PLAY_PATH":
        parts.append(f"pos=({x},{y})")
        parts.append(f"rotated={rotated}")
        parts.append(f"delta={delta:.2f}")

    elif move_type == "PLAY_PLAYER":
        parts.append(f"targetPlayer={target_player}")

    elif move_type == "PLAY_MAP":
        parts.append(f"goalIndex={goal_index}")

    elif move_type == "PLAY_ROCKFALL":
        parts.append(f"pos=({x},{y})")
        parts.append(f"removeDelta={remove_delta:.2f}")
        parts.append(f"idealDelta={ideal_delta:.2f}")

    elif move_type == "DISCARD":
        pass

    return " | ".join(parts)


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    pf = get_nested(state, ["observation", "board", "path_features"], {})
    players = get_nested(state, ["observation", "playerStatus"], [])

    role = role_of(state)
    hand = get_nested(state, ["observation", "private", "hand"], [])

    return {
        "role": role,
        "hand_size": len(hand),
        "legal_actions": len(state.get("legalActions", [])),
        "gold_known": bool(pf.get("gold_known", False)),
        "known_goals": pf.get("known_goals", []),
        "target_distance": pf.get("target_distance", None),
        "distance_to_known_gold": pf.get("distance_to_known_gold", None),
        "average_distance_to_all_goals": pf.get("average_distance_to_all_goals", None),
        "frontier_count": pf.get("frontier_count", None),
        "reachable_count": pf.get("reachable_count", None),
        "destroyable_count": pf.get("destroyable_count", None),
        "players": players,
    }


def print_ranking(name: str, ranking: List[Dict[str, Any]]) -> None:
    print(f"\n{name} top actions:")

    for item in ranking:
        print(
            f"  #{item['rank']} "
            f"idx={item['action_index']:03d} "
            f"prob={item['prob']:.4f} "
            f"logit={item['logit']:.4f} "
            f"{item['summary']}"
        )


# ============================================================
# Episode Comparison
# ============================================================

def compare_one_episode(
    env: SaboteurHttpEnv,
    bc_model: ActionConditionedActorCritic,
    rl_model: ActionConditionedActorCritic,
    driver: str,
    top_k: int,
    max_steps: int,
    episode_id: int,
    out_dir: str,
    pause: bool,
) -> Dict[str, Any]:
    state = env.reset_until_role("SABOTEUR")

    transcript: Dict[str, Any] = {
        "episode_id": episode_id,
        "driver": driver,
        "steps": [],
        "winner": None,
        "done": False,
        "disagreements": 0,
    }

    done = False
    step = 0

    print("\n" + "=" * 80)
    print(f"Episode {episode_id} started. driver={driver}")
    print("=" * 80)

    while not done and step < max_steps:
        if role_of(state) != "SABOTEUR":
            print("[WARN] role changed unexpectedly:", role_of(state))
            transcript["winner"] = "ROLE_CHANGED"
            break

        legal_actions = state.get("legalActions", [])
        if len(legal_actions) == 0:
            print("[WARN] no legal actions")
            transcript["winner"] = "NO_LEGAL_ACTIONS"
            break

        step += 1

        bc_ranking = get_policy_ranking(bc_model, state, top_k=top_k)
        rl_ranking = get_policy_ranking(rl_model, state, top_k=top_k)

        bc_top = bc_ranking[0]
        rl_top = rl_ranking[0]

        same_top1 = bc_top["action_index"] == rl_top["action_index"]

        if not same_top1:
            transcript["disagreements"] += 1

        print("\n" + "-" * 80)
        print(f"Episode {episode_id} | Saboteur decision step {step}")
        print("-" * 80)

        state_summary = summarize_state(state)

        print("State summary:")
        print(
            f"  legal_actions={state_summary['legal_actions']} "
            f"gold_known={state_summary['gold_known']} "
            f"known_goals={state_summary['known_goals']}"
        )
        print(
            f"  target_distance={state_summary['target_distance']} "
            f"distance_to_known_gold={state_summary['distance_to_known_gold']} "
            f"avg_goal_dist={state_summary['average_distance_to_all_goals']}"
        )
        print(
            f"  frontier={state_summary['frontier_count']} "
            f"reachable={state_summary['reachable_count']} "
            f"destroyable={state_summary['destroyable_count']}"
        )

        print_ranking("BC/SFT", bc_ranking)
        print_ranking("RL", rl_ranking)

        print("\nDecision comparison:")
        print(f"  same_top1={same_top1}")
        print(f"  BC top1: idx={bc_top['action_index']} | {bc_top['summary']}")
        print(f"  RL top1: idx={rl_top['action_index']} | {rl_top['summary']}")

        if driver == "bc":
            chosen_idx = bc_top["action_index"]
            chosen_action = legal_actions[chosen_idx]
            chosen_by = "BC/SFT"
        elif driver == "rl":
            chosen_idx = rl_top["action_index"]
            chosen_action = legal_actions[chosen_idx]
            chosen_by = "RL"
        else:
            raise ValueError(f"Unknown driver: {driver}")

        print(f"\nExecuted by {chosen_by}:")
        print(f"  idx={chosen_idx} | {summarize_action(chosen_action)}")

        step_record = {
            "step": step,
            "state_summary": state_summary,
            "bc_top": bc_top,
            "rl_top": rl_top,
            "bc_top_k": bc_ranking,
            "rl_top_k": rl_ranking,
            "same_top1": same_top1,
            "executed_by": chosen_by,
            "executed_action_index": chosen_idx,
            "executed_action_summary": summarize_action(chosen_action),
            "executed_action": chosen_action,
        }

        transcript["steps"].append(step_record)

        if pause:
            input("\nPress Enter to execute this action and continue...")

        next_state = env.step(chosen_action)
        state = next_state
        done = bool(state.get("done", False))

    transcript["done"] = done
    transcript["winner"] = state.get("winner", transcript.get("winner", None))
    transcript["total_steps"] = step
    transcript["won"] = bool(done and transcript["winner"] == "SABOTEUR")

    print("\n" + "=" * 80)
    print(f"Episode {episode_id} finished.")
    print(f"winner={transcript['winner']} won={transcript['won']}")
    print(f"steps={transcript['total_steps']}")
    print(f"disagreements={transcript['disagreements']}")
    print("=" * 80)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compare_episode_{episode_id:03d}_{driver}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    print("saved transcript:", out_path)

    return transcript


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", type=str, default="http://localhost:8000")

    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default="checkpoints/saboteur_bc.pt",
        help="SFT / behavior cloning checkpoint.",
    )

    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default="checkpoints/saboteur_bc_ppo_sparse_http_update_300.pt",
        help="RL fine-tuned checkpoint.",
    )

    parser.add_argument(
        "--driver",
        type=str,
        choices=["bc", "rl"],
        default="bc",
        help="Which model actually plays the game trajectory.",
    )

    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="policy_compare_logs")
    parser.add_argument("--pause", action="store_true")

    args = parser.parse_args()

    print("Loading models...")
    bc_model, bc_ckpt = load_policy(args.bc_checkpoint)
    rl_model, rl_ckpt = load_policy(args.rl_checkpoint)

    print("\nBC/SFT checkpoint:")
    print("  path:", args.bc_checkpoint)
    print("  source:", bc_ckpt.get("source", "unknown"))
    print("  best_val_acc:", bc_ckpt.get("best_val_acc", "N/A"))

    print("\nRL checkpoint:")
    print("  path:", args.rl_checkpoint)
    print("  source:", rl_ckpt.get("source", "unknown"))
    print("  win_rate:", rl_ckpt.get("win_rate", "N/A"))
    print("  reward_mode:", rl_ckpt.get("reward_mode", "N/A"))

    env = SaboteurHttpEnv(args.base_url)

    all_transcripts = []

    for ep in range(1, args.episodes + 1):
        transcript = compare_one_episode(
            env=env,
            bc_model=bc_model,
            rl_model=rl_model,
            driver=args.driver,
            top_k=args.top_k,
            max_steps=args.max_steps,
            episode_id=ep,
            out_dir=args.out_dir,
            pause=args.pause,
        )

        all_transcripts.append(transcript)

    total_steps = sum(t.get("total_steps", 0) for t in all_transcripts)
    total_disagreements = sum(t.get("disagreements", 0) for t in all_transcripts)
    wins = sum(1 for t in all_transcripts if t.get("won", False))

    print("\n" + "=" * 80)
    print("Summary over inspected episodes")
    print("=" * 80)
    print("driver:", args.driver)
    print("episodes:", args.episodes)
    print("wins:", wins)
    print("total_steps:", total_steps)
    print("total_disagreements:", total_disagreements)
    print("disagreement_rate:", total_disagreements / max(1, total_steps))
    print("logs saved to:", args.out_dir)


if __name__ == "__main__":
    main()