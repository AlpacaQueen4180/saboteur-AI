import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from miner_ppo import (
    SaboteurHttpEnv,
    state_to_tensors,
    role_of,
    safe_float,
    safe_int,
    get_nested,
    clamp,
)


# ============================================================
# Basic board / player utility
# ============================================================

def player_threat_score(state: Dict[str, Any], player_index: int) -> float:
    """
    Estimate how dangerous a miner is from the saboteur perspective.

    Higher score means:
        - more likely worth blocking
        - recently contributed to path progress
        - has more cards
        - not already sabotaged
    """
    players = get_nested(state, ["observation", "playerStatus"], [])
    events = get_nested(state, ["observation", "events"], [])

    hand_size = 0.0
    already_sabotaged = 0.0

    for p in players:
        idx = safe_int(p.get("index"), -1)
        if idx == player_index:
            hand_size = safe_float(p.get("handSize"), 0.0) / 6.0
            already_sabotaged = 1.0 if p.get("sabotaged", False) else 0.0
            break

    recent_path_count = 0.0
    recent_repair_count = 0.0

    for e in events:
        idx = safe_int(e.get("playerIndex"), -1)
        event_type = e.get("type", "")

        if idx == player_index:
            if event_type == "PLAY_PATH":
                recent_path_count += 1.0
            elif event_type == "PLAY_PLAYER":
                card_type = e.get("card_type", e.get("cardType", ""))
                if card_type == "REPAIR":
                    recent_repair_count += 1.0

    recent_path_score = clamp(recent_path_count / 3.0, 0.0, 1.0)
    recent_repair_score = clamp(recent_repair_count / 2.0, 0.0, 1.0)

    threat = (
        0.50 * recent_path_score
        + 0.25 * hand_size
        + 0.15 * recent_repair_score
        - 0.40 * already_sabotaged
    )

    return clamp(threat, 0.0, 1.0)


def miner_progress_score(state: Dict[str, Any]) -> float:
    """
    Larger score means miners are closer to winning.
    Saboteur wants this score to go down or stay low.
    """
    pf = get_nested(state, ["observation", "board", "path_features"], {})

    target_distance = safe_float(pf.get("target_distance"), 12.0)
    distance_to_known_gold = safe_float(pf.get("distance_to_known_gold"), 12.0)
    average_distance = safe_float(pf.get("average_distance_to_all_goals"), 12.0)

    reachable_count = safe_float(pf.get("reachable_count"), 0.0)
    frontier_count = safe_float(pf.get("frontier_count"), 0.0)
    destroyable_count = safe_float(pf.get("destroyable_count"), 0.0)

    gold_known = 1.0 if pf.get("gold_known", False) else 0.0

    # Normalize distance: smaller distance should mean higher miner progress.
    target_progress = 1.0 - clamp(target_distance / 12.0, 0.0, 1.0)
    known_gold_progress = 1.0 - clamp(distance_to_known_gold / 12.0, 0.0, 1.0)
    avg_goal_progress = 1.0 - clamp(average_distance / 12.0, 0.0, 1.0)

    reachable_score = clamp(reachable_count / 45.0, 0.0, 1.0)
    frontier_score = clamp(frontier_count / 20.0, 0.0, 1.0)
    destroyable_score = clamp(destroyable_count / 45.0, 0.0, 1.0)

    return (
        0.35 * target_progress
        + 0.25 * known_gold_progress
        + 0.15 * avg_goal_progress
        + 0.10 * reachable_score
        + 0.05 * frontier_score
        + 0.05 * destroyable_score
        + 0.05 * gold_known
    )


# ============================================================
# Saboteur expert policy
# ============================================================

def score_saboteur_action(state: Dict[str, Any], action: Dict[str, Any]) -> float:
    """
    Rule-based saboteur expert.

    This does not need to be perfect.
    Its purpose is to generate a reasonable supervised warm-start dataset.
    """
    move_type = action.get("type", action.get("move_type", ""))
    card_type = action.get("card_type", action.get("cardType", ""))

    score = 0.0

    delta = safe_float(action.get("delta_target_distance", action.get("deltaTargetDistance", 0.0)), 0.0)
    remove_delta = safe_float(action.get("remove_delta", action.get("removeDelta", 0.0)), 0.0)
    ideal_delta = safe_float(action.get("ideal_fill_delta", action.get("idealFillDelta", 0.0)), 0.0)

    before_distance = safe_float(
        action.get("before_target_distance", action.get("beforeTargetDistance", 12.0)),
        12.0,
    )

    # ------------------------------------------------------------
    # 1. BLOCK dangerous miners
    # ------------------------------------------------------------
    if move_type == "PLAY_PLAYER" and card_type == "BLOCK":
        own_idx = safe_int(get_nested(state, ["observation", "private", "playerIndex"], 3), 3)
        target_player = safe_int(action.get("target_player", action.get("targetPlayer", -1)), -1)

        if target_player == own_idx or target_player < 0:
            return -10.0

        threat = player_threat_score(state, target_player)

        # Blocking a real threat is highly valuable.
        score += 2.0 + 4.0 * threat

        # Late game: blocking is even more important.
        if before_distance <= 3.0:
            score += 1.5
        elif before_distance <= 6.0:
            score += 0.7

        return score

    # ------------------------------------------------------------
    # 2. REPAIR self only if saboteur is blocked
    # ------------------------------------------------------------
    if move_type == "PLAY_PLAYER" and card_type == "REPAIR":
        own_idx = safe_int(get_nested(state, ["observation", "private", "playerIndex"], 3), 3)
        target_player = safe_int(action.get("target_player", action.get("targetPlayer", -1)), -1)

        if target_player == own_idx:
            score += 1.0
        else:
            score -= 4.0

        return score

    # ------------------------------------------------------------
    # 3. ROCKFALL useful miner path
    # ------------------------------------------------------------
    if move_type == "PLAY_ROCKFALL":
        # remove_delta > 0 means removing this tile makes target farther.
        score += 1.0 + 3.0 * remove_delta

        # If ideal_delta is high, removing this card may allow miners to improve later.
        # Penalize possible miner-helping rockfall.
        if ideal_delta > 0:
            score -= 1.5 * ideal_delta

        # Late game rockfall is more valuable.
        if before_distance <= 3.0:
            score += 1.5
        elif before_distance <= 6.0:
            score += 0.8

        return score

    # ------------------------------------------------------------
    # 4. PLAY_PATH as sabotage
    # ------------------------------------------------------------
    if move_type == "PLAY_PATH":
        # For saboteur:
        #   delta < 0 means the path makes miners farther from target.
        #   delta > 0 means the path helps miners.
        if delta < 0:
            score += 1.0 + 3.0 * (-delta)

            if card_type == "DEADEND":
                score += 1.0

            if before_distance <= 4.0:
                score += 0.8
        else:
            score -= 2.0 + 2.0 * delta

        return score

    # ------------------------------------------------------------
    # 5. MAP is mildly useful early, less useful later
    # ------------------------------------------------------------
    if move_type == "PLAY_MAP":
        pf = get_nested(state, ["observation", "board", "path_features"], {})
        known_goals = pf.get("known_goals", ["UNKNOWN", "UNKNOWN", "UNKNOWN"])
        gold_known = bool(pf.get("gold_known", False))

        goal_index = safe_int(action.get("goal_index", action.get("goalIndex", -1)), -1)

        if gold_known:
            score -= 0.5
        elif 0 <= goal_index < 3 and known_goals[goal_index] == "UNKNOWN":
            score += 0.4
        else:
            score -= 0.2

        return score

    # ------------------------------------------------------------
    # 6. DISCARD
    # ------------------------------------------------------------
    if move_type == "DISCARD":
        # Discard is acceptable when no strong sabotage exists.
        # But it should not dominate.
        score += 0.0

        # If this action contains a card_type field, avoid discarding strong sabotage cards.
        if card_type in ["BLOCK", "ROCKFALL", "DEADEND"]:
            score -= 1.0

        return score

    return score


def choose_expert_action(
    state: Dict[str, Any],
    epsilon: float = 0.10,
    top_k: int = 3,
) -> Tuple[int, Dict[str, Any], List[float]]:
    """
    Choose an action index from legalActions.

    epsilon:
        with small probability, sample among top-k actions.
        This prevents collecting overly deterministic / narrow data.
    """
    legal_actions = state.get("legalActions", [])

    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    scores = [score_saboteur_action(state, a) for a in legal_actions]

    # Occasionally sample from top-k for diversity.
    if random.random() < epsilon:
        ranked = np.argsort(scores)[::-1]
        k = min(top_k, len(ranked))
        chosen_idx = int(random.choice(ranked[:k]))
    else:
        chosen_idx = int(np.argmax(scores))

    return chosen_idx, legal_actions[chosen_idx], scores


# ============================================================
# Dataset collection
# ============================================================

def collect_one_episode(
    env: SaboteurHttpEnv,
    max_steps: int,
    epsilon: float,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
        won, episode_data
    """
    state = env.reset_until_role("SABOTEUR")

    ep_obs = []
    ep_action_feats = []
    ep_masks = []
    ep_labels = []
    ep_action_json = []
    ep_expert_scores = []

    steps = 0
    done = False
    winner = None

    while not done and steps < max_steps:
        if role_of(state) != "SABOTEUR":
            # For this training branch, /step should return the next player-3 decision state.
            # If somehow the role changes, restart this episode as failed.
            if verbose:
                print("[warn] role changed unexpectedly:", role_of(state))
            return False, {}

        legal_actions = state.get("legalActions", [])
        if len(legal_actions) == 0:
            if verbose:
                print("[warn] no legal actions")
            return False, {}

        obs, action_feats, mask = state_to_tensors(state)

        action_idx, action, scores = choose_expert_action(
            state=state,
            epsilon=epsilon,
            top_k=3,
        )

        ep_obs.append(obs)
        ep_action_feats.append(action_feats)
        ep_masks.append(mask)
        ep_labels.append(action_idx)
        ep_action_json.append(action)
        ep_expert_scores.append(scores)

        try:
            next_state = env.step(action)
        except requests.HTTPError as e:
            if verbose:
                print("[warn] HTTPError during step:", e)
            return False, {}

        state = next_state
        done = bool(state.get("done", False))
        winner = state.get("winner", None)

        steps += 1

    won = bool(done and winner == "SABOTEUR")

    episode_data = {
        "obs": ep_obs,
        "action_feats": ep_action_feats,
        "masks": ep_masks,
        "labels": ep_labels,
        "actions_json": ep_action_json,
        "expert_scores": ep_expert_scores,
        "steps": steps,
        "winner": winner,
    }

    return won, episode_data


def save_dataset(
    output_path: str,
    obs: List[np.ndarray],
    action_feats: List[np.ndarray],
    masks: List[np.ndarray],
    labels: List[int],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    np.savez_compressed(
        output_path,
        obs=np.asarray(obs, dtype=np.float32),
        action_feats=np.asarray(action_feats, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--out", type=str, default="data/saboteur_win_bc_dataset.npz")

    parser.add_argument("--target-win-episodes", type=int, default=200)
    parser.add_argument("--max-total-episodes", type=int, default=5000)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)

    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-wins", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SaboteurHttpEnv(args.base_url)

    all_obs: List[np.ndarray] = []
    all_action_feats: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_labels: List[int] = []

    win_episodes = 0
    total_episodes = 0
    total_win_steps = 0

    episode_lengths: List[int] = []

    print("Collecting saboteur winning episodes...")
    print("base_url:", args.base_url)
    print("output:", args.out)
    print("target_win_episodes:", args.target_win_episodes)
    print("max_total_episodes:", args.max_total_episodes)
    print("epsilon:", args.epsilon)

    while (
        win_episodes < args.target_win_episodes
        and total_episodes < args.max_total_episodes
    ):
        total_episodes += 1

        won, ep = collect_one_episode(
            env=env,
            max_steps=args.max_steps_per_episode,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )

        if won:
            win_episodes += 1
            steps = int(ep["steps"])
            total_win_steps += steps
            episode_lengths.append(steps)

            all_obs.extend(ep["obs"])
            all_action_feats.extend(ep["action_feats"])
            all_masks.extend(ep["masks"])
            all_labels.extend(ep["labels"])

            print(
                f"[WIN {win_episodes:04d}] "
                f"episode={total_episodes} "
                f"steps={steps} "
                f"samples={len(all_labels)}"
            )

            if win_episodes % args.save_every_wins == 0:
                meta = {
                    "role": "SABOTEUR",
                    "total_episodes": total_episodes,
                    "win_episodes": win_episodes,
                    "samples": len(all_labels),
                    "epsilon": args.epsilon,
                    "seed": args.seed,
                    "avg_win_steps": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    "note": "Only samples from episodes where winner == SABOTEUR are stored.",
                }

                save_dataset(
                    output_path=args.out,
                    obs=all_obs,
                    action_feats=all_action_feats,
                    masks=all_masks,
                    labels=all_labels,
                    meta=meta,
                )

                print(f"[checkpoint saved] {args.out}")

        if total_episodes % 50 == 0:
            current_rate = win_episodes / max(1, total_episodes)
            print(
                f"[progress] "
                f"episodes={total_episodes} "
                f"wins={win_episodes} "
                f"win_rate={current_rate:.4f} "
                f"samples={len(all_labels)}"
            )

    meta = {
        "role": "SABOTEUR",
        "total_episodes": total_episodes,
        "win_episodes": win_episodes,
        "samples": len(all_labels),
        "epsilon": args.epsilon,
        "seed": args.seed,
        "avg_win_steps": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "note": "Only samples from episodes where winner == SABOTEUR are stored.",
    }

    save_dataset(
        output_path=args.out,
        obs=all_obs,
        action_feats=all_action_feats,
        masks=all_masks,
        labels=all_labels,
        meta=meta,
    )

    print("\nDone.")
    print("saved:", args.out)
    print("total_episodes:", total_episodes)
    print("win_episodes:", win_episodes)
    print("samples:", len(all_labels))
    print("avg_win_steps:", meta["avg_win_steps"])


if __name__ == "__main__":
    main()