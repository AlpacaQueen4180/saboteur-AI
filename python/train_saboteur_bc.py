import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from miner_ppo import (
    ActionConditionedActorCritic,
    MAX_ACTIONS,
    MOVE_TYPES,
    CARD_TYPES,
    DEVICE,
)


def masked_cross_entropy(scores, masks, labels):
    masked_scores = scores.masked_fill(masks <= 0, -1e9)
    return nn.functional.cross_entropy(masked_scores, labels)


def evaluate(model, obs, action_feats, masks, labels, batch_size):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        n = obs.shape[0]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            obs_b = torch.tensor(obs[start:end], dtype=torch.float32, device=DEVICE)
            act_b = torch.tensor(action_feats[start:end], dtype=torch.float32, device=DEVICE)
            mask_b = torch.tensor(masks[start:end], dtype=torch.float32, device=DEVICE)
            label_b = torch.tensor(labels[start:end], dtype=torch.long, device=DEVICE)

            scores, _ = model(obs_b, act_b)
            loss = masked_cross_entropy(scores, mask_b, label_b)

            masked_scores = scores.masked_fill(mask_b <= 0, -1e9)
            pred = torch.argmax(masked_scores, dim=1)

            total_loss += float(loss.item()) * (end - start)
            total_correct += int((pred == label_b).sum().item())
            total_samples += end - start

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="data/saboteur_win_bc_dataset.npz")
    parser.add_argument("--out", type=str, default="checkpoints/saboteur_bc.pt")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)

    obs = data["obs"].astype(np.float32)
    action_feats = data["action_feats"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    meta = {}
    if "meta" in data.files:
        meta = json.loads(str(data["meta"]))

    n = obs.shape[0]
    obs_dim = obs.shape[1]
    action_dim = action_feats.shape[2]

    print("Loaded dataset:", args.data)
    print("samples:", n)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("meta:", meta)

    indices = np.arange(n)
    np.random.shuffle(indices)

    val_size = int(n * args.val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    obs_train = obs[train_idx]
    act_train = action_feats[train_idx]
    mask_train = masks[train_idx]
    label_train = labels[train_idx]

    obs_val = obs[val_idx]
    act_val = action_feats[val_idx]
    mask_val = masks[val_idx]
    label_val = labels[val_idx]

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_order = np.arange(obs_train.shape[0])
        np.random.shuffle(train_order)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(train_order), args.batch_size):
            batch_idx = train_order[start:start + args.batch_size]

            obs_b = torch.tensor(obs_train[batch_idx], dtype=torch.float32, device=DEVICE)
            act_b = torch.tensor(act_train[batch_idx], dtype=torch.float32, device=DEVICE)
            mask_b = torch.tensor(mask_train[batch_idx], dtype=torch.float32, device=DEVICE)
            label_b = torch.tensor(label_train[batch_idx], dtype=torch.long, device=DEVICE)

            scores, _ = model(obs_b, act_b)
            loss = masked_cross_entropy(scores, mask_b, label_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                masked_scores = scores.masked_fill(mask_b <= 0, -1e9)
                pred = torch.argmax(masked_scores, dim=1)

                total_loss += float(loss.item()) * len(batch_idx)
                total_correct += int((pred == label_b).sum().item())
                total_samples += len(batch_idx)

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)

        val_loss, val_acc = evaluate(
            model=model,
            obs=obs_val,
            action_feats=act_val,
            masks=mask_val,
            labels=label_val,
            batch_size=args.batch_size,
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.5f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "max_actions": MAX_ACTIONS,
                    "role": "SABOTEUR",
                    "move_types": MOVE_TYPES,
                    "card_types": CARD_TYPES,
                    "source": "behavior_cloning",
                    "dataset": args.data,
                    "dataset_meta": meta,
                    "best_val_acc": best_val_acc,
                },
                args.out,
            )

            print("saved best checkpoint:", args.out)

    print("Done.")
    print("best_val_acc:", best_val_acc)
    print("saved:", args.out)


if __name__ == "__main__":
    main()