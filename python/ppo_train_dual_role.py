import argparse
from typing import Any, Dict

import requests

from miner_ppo import train_miner
from saboteur_ppo import train_saboteur
from miner_trust import train_miner_trust


BASE_URL = "http://localhost:8000"


def get_role(state: Dict[str, Any]) -> str:
    return state["observation"]["private"]["role"]


def wait_for_role(base_url: str, target_role: str, max_tries: int = 1000) -> Dict[str, Any]:
    for i in range(1, max_tries + 1):
        r = requests.post(base_url.rstrip("/") + "/reset", json={}, timeout=20)
        r.raise_for_status()
        state = r.json()
        role = get_role(state)

        if role == target_role:
            print(f"[role matched] target={target_role}, tries={i}")
            return state

        if i % 20 == 0:
            print(f"[waiting role] target={target_role}, tries={i}, last_role={role}")

    raise RuntimeError(f"Could not sample target role {target_role} after {max_tries} resets.")


def check_server(base_url: str) -> None:
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=5)
        r.raise_for_status()
        print("[server health]", r.json())
    except Exception as e:
        raise RuntimeError(
            "Cannot connect to Java HTTP server. "
            "Please run `mvn exec:java` in another terminal first."
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--miner", action="store_true", help="Train only the Miner PPO.")
    mode.add_argument("--saboteur", action="store_true", help="Train only the Saboteur PPO.")
    mode.add_argument("--miner_trust", action="store_true", help="Train Miner-side learned trust / saboteur classifier.")

    parser.add_argument("--base-url", type=str, default=BASE_URL)

    # PPO args
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--debug-every", type=int, default=10)

    # Trust model args
    parser.add_argument("--games", type=int, default=500, help="Number of miner-perspective games for trust training.")
    parser.add_argument("--max-steps-per-game", type=int, default=80)
    parser.add_argument("--trust-epochs", type=int, default=30)
    parser.add_argument("--trust-batch-size", type=int, default=128)
    parser.add_argument("--trust-lr", type=float, default=1e-3)
    parser.add_argument("--trust-save-path", type=str, default="checkpoints/miner_trust_model.pt")

    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="",
        help="Optional pretrained BC checkpoint path for PPO fine-tuning.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Optional miner PPO checkpoint path for pure-PPO resume training.",
    )

    args = parser.parse_args()

    check_server(args.base_url)

    if args.miner:
        print("Mode: train Miner PPO only")
        initial_state = wait_for_role(args.base_url, "GOLD_MINER")
        train_miner(
            base_url=args.base_url,
            initial_state=initial_state,
            total_updates=args.updates,
            rollout_steps=args.rollout_steps,
            save_every=args.save_every,
            debug_every=args.debug_every,
            resume_path=args.resume_path,
        )

    elif args.saboteur:
        print("Mode: train Saboteur PPO only")
        initial_state = wait_for_role(args.base_url, "SABOTEUR")
        train_saboteur(
            base_url=args.base_url,
            initial_state=initial_state,
            total_updates=args.updates,
            rollout_steps=args.rollout_steps,
            save_every=args.save_every,
            debug_every=args.debug_every,
            pretrained_path=args.pretrained_path,
        )

    elif args.miner_trust:
        print("Mode: train Miner Trust model only")
        train_miner_trust(
            base_url=args.base_url,
            num_games=args.games,
            max_steps_per_game=args.max_steps_per_game,
            epochs=args.trust_epochs,
            batch_size=args.trust_batch_size,
            lr=args.trust_lr,
            save_path=args.trust_save_path,
        )


if __name__ == "__main__":
    main()
