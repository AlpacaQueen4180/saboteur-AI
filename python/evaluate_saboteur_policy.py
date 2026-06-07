import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import requests
import torch

from miner_ppo import (
    DEVICE,
    SaboteurHttpEnv,
    ActionConditionedActorCritic,
    state_to_tensors,
    role_of,
)


def load_policy(checkpoint_path: str) -> Tuple[ActionConditionedActorCritic, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[eval] loaded checkpoint:", checkpoint_path)
    print("[eval] source:", ckpt.get("source", "unknown"))
    print("[eval] role:", ckpt.get("role", "unknown"))
    print("[eval] obs_dim:", obs_dim)
    print("[eval] action_dim:", action_dim)
    print("[eval] checkpoint win_rate:", ckpt.get("win_rate", "N/A"))
    print("[eval] reward_mode:", ckpt.get("reward_mode", "N/A"))

    return model, ckpt


def select_action(
    model: ActionConditionedActorCritic,
    state: Dict[str, Any],
    deterministic: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    legal_actions = state.get("legalActions", [])

    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    obs, action_feats, mask = state_to_tensors(state)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        scores, _ = model.forward(obs_t, act_t)
        masked_scores = scores.masked_fill(mask_t <= 0, -1e9)

        if deterministic:
            action_t = torch.argmax(masked_scores, dim=1)
        else:
            dist = torch.distributions.Categorical(logits=masked_scores)
            action_t = dist.sample()

    action_id = int(action_t.item())

    if action_id >= len(legal_actions):
        # Safety fallback. Ideally this should never happen because of mask.
        action_id = 0

    return action_id, legal_actions[action_id]


def evaluate_one_episode(
    env: SaboteurHttpEnv,
    model: ActionConditionedActorCritic,
    max_steps_per_episode: int,
    deterministic: bool,
    verbose: bool = False,
) -> Dict[str, Any]:
    state = env.reset_until_role("SABOTEUR")

    steps = 0
    done = False
    winner = None

    action_type_counts = {
        "DISCARD": 0,
        "PLAY_PATH": 0,
        "PLAY_PLAYER": 0,
        "PLAY_MAP": 0,
        "PLAY_ROCKFALL": 0,
        "OTHER": 0,
    }

    while not done and steps < max_steps_per_episode:
        if role_of(state) != "SABOTEUR":
            # In your current server design, /step should return the next decision
            # state for player 3. If this happens, treat this episode as abnormal.
            return {
                "valid": False,
                "won": False,
                "winner": "ROLE_CHANGED",
                "steps": steps,
                "action_type_counts": action_type_counts,
            }

        legal_actions = state.get("legalActions", [])
        if len(legal_actions) == 0:
            return {
                "valid": False,
                "won": False,
                "winner": "NO_LEGAL_ACTIONS",
                "steps": steps,
                "action_type_counts": action_type_counts,
            }

        try:
            _, action = select_action(
                model=model,
                state=state,
                deterministic=deterministic,
            )

            move_type = action.get("type", action.get("move_type", "OTHER"))
            if move_type in action_type_counts:
                action_type_counts[move_type] += 1
            else:
                action_type_counts["OTHER"] += 1

            next_state = env.step(action)

        except requests.HTTPError as e:
            if verbose:
                print("[eval] HTTP step error:", e)

            return {
                "valid": False,
                "won": False,
                "winner": "HTTP_ERROR",
                "steps": steps,
                "action_type_counts": action_type_counts,
            }

        except Exception as e:
            if verbose:
                print("[eval] unexpected error:", repr(e))

            return {
                "valid": False,
                "won": False,
                "winner": "ERROR",
                "steps": steps,
                "action_type_counts": action_type_counts,
            }

        state = next_state
        done = bool(state.get("done", False))
        winner = state.get("winner", None)

        steps += 1

    if not done:
        winner = "TIMEOUT"

    won = bool(done and winner == "SABOTEUR")

    return {
        "valid": True,
        "won": won,
        "winner": winner,
        "steps": steps,
        "action_type_counts": action_type_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/saboteur_bc_ppo_http_update_300.pt",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use sampling instead of argmax action selection.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)

    args = parser.parse_args()

    deterministic = not args.stochastic

    env = SaboteurHttpEnv(args.base_url)
    model, _ = load_policy(args.checkpoint)

    wins = 0
    valid_episodes = 0
    invalid_episodes = 0

    step_lengths = []
    winner_counts = {}

    total_action_type_counts = {
        "DISCARD": 0,
        "PLAY_PATH": 0,
        "PLAY_PLAYER": 0,
        "PLAY_MAP": 0,
        "PLAY_ROCKFALL": 0,
        "OTHER": 0,
    }

    print("\nEvaluation started.")
    print("base_url:", args.base_url)
    print("checkpoint:", args.checkpoint)
    print("episodes:", args.episodes)
    print("deterministic:", deterministic)
    print("max_steps_per_episode:", args.max_steps_per_episode)

    for ep in range(1, args.episodes + 1):
        result = evaluate_one_episode(
            env=env,
            model=model,
            max_steps_per_episode=args.max_steps_per_episode,
            deterministic=deterministic,
            verbose=args.verbose,
        )

        if result["valid"]:
            valid_episodes += 1
        else:
            invalid_episodes += 1

        if result["won"]:
            wins += 1

        winner = result["winner"]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

        step_lengths.append(result["steps"])

        for k, v in result["action_type_counts"].items():
            total_action_type_counts[k] += v

        if ep % args.print_every == 0:
            win_rate = wins / max(1, valid_episodes)
            avg_steps = float(np.mean(step_lengths)) if step_lengths else 0.0

            print(
                f"[eval {ep:04d}/{args.episodes}] "
                f"valid={valid_episodes} "
                f"invalid={invalid_episodes} "
                f"wins={wins} "
                f"win_rate={win_rate:.4f} "
                f"avg_steps={avg_steps:.2f}"
            )

    final_win_rate = wins / max(1, valid_episodes)
    avg_steps = float(np.mean(step_lengths)) if step_lengths else 0.0
    std_steps = float(np.std(step_lengths)) if step_lengths else 0.0

    total_actions = sum(total_action_type_counts.values())
    action_type_rates = {
        k: v / max(1, total_actions)
        for k, v in total_action_type_counts.items()
    }

    print("\n========== Evaluation Result ==========")
    print("checkpoint:", args.checkpoint)
    print("episodes requested:", args.episodes)
    print("valid episodes:", valid_episodes)
    print("invalid episodes:", invalid_episodes)
    print("wins:", wins)
    print("win_rate:", round(final_win_rate, 4))
    print("avg_steps:", round(avg_steps, 3))
    print("std_steps:", round(std_steps, 3))

    print("\nWinner counts:")
    for k, v in sorted(winner_counts.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("\nAction type counts:")
    for k, v in total_action_type_counts.items():
        print(f"  {k}: {v}")

    print("\nAction type rates:")
    for k, v in action_type_rates.items():
        print(f"  {k}: {v:.4f}")

    print("=======================================")


if __name__ == "__main__":
    main()