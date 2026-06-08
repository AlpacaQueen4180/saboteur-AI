import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import requests
import torch

from miner_ppo import (
    DEVICE,
    TRUST_MODEL_PATH,
    ActionConditionedActorCritic,
    MinerTrustInference,
    SaboteurHttpEnv,
    role_of,
    state_to_tensors,
)


def load_policy(checkpoint_path: str) -> Tuple[ActionConditionedActorCritic, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    role = ckpt.get("role", "")
    if role != "GOLD_MINER":
        raise ValueError(f"Expected a GOLD_MINER checkpoint, got role={role!r}.")

    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[eval miner] loaded checkpoint:", checkpoint_path)
    print("[eval miner] source:", ckpt.get("source", "unknown"))
    print("[eval miner] role:", role)
    print("[eval miner] obs_dim:", obs_dim)
    print("[eval miner] action_dim:", action_dim)
    print("[eval miner] checkpoint win_rate:", ckpt.get("win_rate", "N/A"))
    print("[eval miner] trust_model_path:", ckpt.get("trust_model_path", TRUST_MODEL_PATH))

    return model, ckpt


def validate_policy_dimensions(
    state: Dict[str, Any],
    ckpt: Dict[str, Any],
    trust_inference: MinerTrustInference,
) -> None:
    obs, action_feats, _ = state_to_tensors(state, trust_inference)
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])

    if obs.shape[0] != obs_dim or action_feats.shape[1] != action_dim:
        raise ValueError(
            "Checkpoint dimension mismatch for current miner environment: "
            f"checkpoint obs/action=({obs_dim}, {action_dim}), "
            f"current obs/action=({obs.shape[0]}, {action_feats.shape[1]})"
        )


def select_greedy_action(
    model: ActionConditionedActorCritic,
    state: Dict[str, Any],
    trust_inference: MinerTrustInference,
) -> Tuple[int, Dict[str, Any]]:
    legal_actions = state.get("legalActions", [])
    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    obs, action_feats, mask = state_to_tensors(state, trust_inference)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        scores, _ = model.forward(obs_t, act_t)
        masked_scores = scores.masked_fill(mask_t <= 0, -1e9)
        action_t = torch.argmax(masked_scores, dim=1)

    action_id = int(action_t.item())
    if action_id >= len(legal_actions):
        raise RuntimeError(
            f"Model selected padded action index {action_id}, "
            f"but only {len(legal_actions)} legal actions exist."
        )

    return action_id, legal_actions[action_id]


def evaluate_one_episode(
    env: SaboteurHttpEnv,
    model: ActionConditionedActorCritic,
    trust_inference: MinerTrustInference,
    max_steps_per_episode: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    state = env.reset_until_role("GOLD_MINER")
    trust_inference.reset(state, max_steps_per_game=max_steps_per_episode)

    steps = 0
    done = False
    winner = None

    while not done and steps < max_steps_per_episode:
        if role_of(state) != "GOLD_MINER":
            return {
                "valid": False,
                "won": False,
                "winner": "ROLE_CHANGED",
                "steps": steps,
            }

        legal_actions = state.get("legalActions", [])
        if len(legal_actions) == 0:
            return {
                "valid": False,
                "won": False,
                "winner": "NO_LEGAL_ACTIONS",
                "steps": steps,
            }

        try:
            _, action = select_greedy_action(model, state, trust_inference)
            next_state = env.step(action)
        except requests.HTTPError as e:
            if verbose:
                print("[eval miner] HTTP step error:", e)
            return {
                "valid": False,
                "won": False,
                "winner": "HTTP_ERROR",
                "steps": steps,
            }
        except Exception as e:
            if verbose:
                print("[eval miner] unexpected error:", repr(e))
            return {
                "valid": False,
                "won": False,
                "winner": "ERROR",
                "steps": steps,
            }

        state = next_state
        done = bool(state.get("done", False))
        winner = state.get("winner", None)
        steps += 1

        if not done:
            trust_inference.update_from_state(state, increment_step=True)

    if not done:
        winner = "TIMEOUT"

    won = bool(done and winner == "GOLD_MINER")

    return {
        "valid": True,
        "won": won,
        "winner": winner,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/miner_ppo_http_trust_update_300.pt",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    env = SaboteurHttpEnv(args.base_url)
    model, ckpt = load_policy(args.checkpoint)
    trust_inference = MinerTrustInference(TRUST_MODEL_PATH, DEVICE)

    initial_state = env.reset_until_role("GOLD_MINER")
    trust_inference.reset(initial_state, max_steps_per_game=args.max_steps_per_episode)
    validate_policy_dimensions(initial_state, ckpt, trust_inference)

    wins = 0
    valid_episodes = 0
    invalid_episodes = 0
    step_lengths = []
    valid_step_lengths = []
    winner_counts: Dict[Any, int] = {}

    print("\nMiner evaluation started.")
    print("base_url:", args.base_url)
    print("checkpoint:", args.checkpoint)
    print("episodes:", args.episodes)
    print("selection: greedy argmax")
    print("max_steps_per_episode:", args.max_steps_per_episode)
    print("trust_features_enabled:", trust_inference.available)

    for ep in range(1, args.episodes + 1):
        result = evaluate_one_episode(
            env=env,
            model=model,
            trust_inference=trust_inference,
            max_steps_per_episode=args.max_steps_per_episode,
            verbose=args.verbose,
        )

        if result["valid"]:
            valid_episodes += 1
            valid_step_lengths.append(result["steps"])
        else:
            invalid_episodes += 1

        if result["won"]:
            wins += 1

        winner = result["winner"]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        step_lengths.append(result["steps"])

        if ep % args.print_every == 0:
            win_rate = wins / max(1, valid_episodes)
            avg_steps = float(np.mean(valid_step_lengths)) if valid_step_lengths else 0.0
            print(
                f"[eval miner {ep:04d}/{args.episodes}] "
                f"valid={valid_episodes} "
                f"invalid={invalid_episodes} "
                f"wins={wins} "
                f"win_rate={win_rate:.4f} "
                f"avg_steps={avg_steps:.2f}"
            )

    final_win_rate = wins / max(1, valid_episodes)
    avg_steps = float(np.mean(valid_step_lengths)) if valid_step_lengths else 0.0
    std_steps = float(np.std(valid_step_lengths)) if valid_step_lengths else 0.0

    print("\n========== Miner Evaluation Result ==========")
    print("checkpoint:", args.checkpoint)
    print("episodes requested:", args.episodes)
    print("valid_episodes:", valid_episodes)
    print("invalid_episodes:", invalid_episodes)
    print("wins:", wins)
    print("win_rate:", round(final_win_rate, 4))
    print("avg_steps:", round(avg_steps, 3))
    print("std_steps:", round(std_steps, 3))

    print("\nWinner counts:")
    for k, v in sorted(winner_counts.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("============================================")


if __name__ == "__main__":
    main()
