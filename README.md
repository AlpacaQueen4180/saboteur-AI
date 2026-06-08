# Saboteur AI — Saboteur Training Branch

This branch focuses on training a **Saboteur-side AI agent** for the board game *Saboteur*.  
Unlike the original game repository, this branch is mainly an experimental RL/AI training branch. The core goal is not to reproduce the full GUI game flow, but to provide a stable headless environment where a Python-controlled Saboteur policy can be trained and evaluated against rule-based agents.

The final training pipeline used in this branch is:

```text
Rule-based expert trajectory collection
        ↓
Supervised learning / Behavior Cloning
        ↓
Pretrained Saboteur policy
        ↓
KL-regularized PPO fine-tuning
        ↓
Final Saboteur model
```

The main focus is therefore:

> Train a stronger Saboteur agent by first using supervised learning to obtain a stable policy, then applying reinforcement learning to further improve performance.

---

## Project Overview

The project uses a Java game backend and Python learning scripts.

```text
Java game environment
    ├── Saboteur game rules
    ├── HeuristicsAI rule-based players
    ├── Headless HTTP server
    └── Rule-based baseline evaluation

Python training side
    ├── State/action encoding
    ├── Action-conditioned actor-critic model
    ├── Saboteur behavior cloning
    ├── PPO fine-tuning
    ├── KL regularization toward the pretrained policy
    └── Evaluation and policy comparison tools
```

The current training setting is a fixed 4-player game:

```text
Player 0: Rule-based HeuristicsAI
Player 1: Rule-based HeuristicsAI
Player 2: Rule-based HeuristicsAI
Player 3: Python-controlled Saboteur model
```

During Saboteur evaluation, player 3 is reset until assigned the `SABOTEUR` role. The other three players are controlled by Java rule-based agents.

---

## Motivation

Initial experiments with pure PPO were unstable and failed to train a strong Saboteur policy. The main reasons were:

- Saboteur rewards are sparse and delayed.
- The Saboteur role is harder than the miner role because it must prevent progress rather than directly build toward the goal.
- Pure PPO frequently received many losing trajectories, making credit assignment difficult.
- Unregularized PPO fine-tuning could drift away from a good supervised policy.

To address this, the branch uses a two-stage approach:

1. **Supervised learning / Behavior Cloning**
   - Collect winning Saboteur trajectories from a rule-based expert.
   - Train the policy to imitate expert choices.

2. **KL-regularized PPO**
   - Initialize PPO from the behavior-cloned model.
   - Fine-tune using terminal win/loss reward.
   - Add KL regularization toward the frozen behavior-cloned policy to prevent policy drift.

---

## Model Design

### Action-Conditioned Actor-Critic

The policy model is action-conditioned. Instead of outputting a fixed action ID, the model scores each currently legal action returned by the Java environment.

This is useful because Saboteur has a variable legal action set every turn.

```text
observation vector
        ↓
observation encoder
        ↓
observation embedding

legal action features
        ↓
action encoder
        ↓
action embeddings

observation embedding + action embedding
        ↓
action scorer
        ↓
score for each legal action
```

The model selects among legal actions using a mask.

---

## Training Pipeline

### 1. Collect Saboteur Winning Data

Script:

```text
python/collect_saboteur_win_dataset.py
```

This script runs a rule-based Saboteur expert against three rule-based players. It only stores trajectories where the Saboteur side wins.

Each sample contains:

```text
obs            encoded observation
action_feats   encoded legal actions
mask           legal action mask
label          expert action index
```

Example command:

```powershell
python python/collect_saboteur_win_dataset.py --target-win-episodes 200 --max-total-episodes 5000 --out data/saboteur_win_bc_dataset.npz
```

---

### 2. Train Behavior Cloning / SFT Model

Script:

```text
python/train_saboteur_bc.py
```

This script trains the Saboteur policy with supervised learning using the collected winning trajectories.

Example command:

```powershell
python python/train_saboteur_bc.py --data data/saboteur_win_bc_dataset.npz --out checkpoints/saboteur_bc.pt --epochs 50
```

The output checkpoint is:

```text
checkpoints/saboteur_bc.pt
```

In the current experiments, the behavior-cloned model achieved about:

```text
BC / SFT Saboteur win rate: ~72%
```

---

### 3. KL-Regularized PPO Fine-Tuning

Script:

```text
python/saboteur_ppo.py
```

Entry point:

```text
python/ppo_train_dual_role.py
```

The PPO fine-tuning stage loads the pretrained behavior-cloned model and continues training with reinforcement learning.

The current objective is:

```text
PPO clipped policy loss
+ value loss
- entropy bonus
+ KL regularization toward frozen BC policy
```

The reward is sparse and terminal-oriented:

```text
Non-terminal step: 0
Saboteur win:      +1
Saboteur lose:     -1
Early lose:        additional small penalty
```

Example command:

```powershell
python python/ppo_train_dual_role.py --saboteur --pretrained-path checkpoints/saboteur_bc.pt --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10
```

The best current checkpoint from evaluation was:

```text
checkpoints/saboteur_bc_ppo_kl_http_update_100.pt
```

with:

```text
BC + KL-PPO Saboteur win rate: 75.6%
```

over 500 deterministic evaluation games.

---

## Evaluation

### Evaluate a Saboteur Checkpoint

Script:

```text
python/evaluate_saboteur_policy.py
```

Example:

```powershell
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500
```

To save the result into a text file:

```powershell
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500 | Tee-Object -FilePath eval_logs/eval_kl_update_100.txt
```

The evaluation reports:

```text
valid episodes
wins
win_rate
average episode steps
winner counts
action type counts
action type rates
```

---

## Qualitative Policy Comparison

Script:

```text
python/compare_saboteur_policies_in_game.py
```

This tool compares two Saboteur models on the same game states.

For example, compare the behavior-cloned model and the RL-fine-tuned model:

```powershell
python python/compare_saboteur_policies_in_game.py --bc-checkpoint checkpoints/saboteur_bc.pt --rl-checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --driver bc --episodes 3 --top-k 5
```

This prints the top-k actions from each model at each Saboteur decision state.

The tool is useful for analyzing whether RL fine-tuning changed the model's strategy. In observed cases, most decisions remained similar to the BC policy, while differences mainly appeared in:

- which player to block,
- which card to discard,
- small local action preferences.

---

## Rule-Based Baseline

A Java baseline evaluator is included to measure the performance of four rule-based `HeuristicsAI` players.

Script:

```text
src/main/java/main/RuleBasedEvalMain.java
```

Compile first:

```powershell
mvn compile
```

Build classpath:

```powershell
mvn dependency:build-classpath "-Dmdep.outputFile=cp.txt"
```

Run baseline evaluation:

```powershell
java -cp "target/classes;$(Get-Content cp.txt)" main.RuleBasedEvalMain 500
```

Observed baseline result:

```text
4 rule-based HeuristicsAI players
games: 500
miner win rate:    65.2%
saboteur win rate: 34.8%
```

This baseline is not exactly the same setting as the Python model evaluation, because the model evaluation fixes player 3 as the Saboteur. However, it provides a useful reference for the strength of the pure rule-based setup.

---

## Important Results

| Setting | Description | Saboteur Win Rate |
|---|---|---:|
| Rule-based baseline | Four Java `HeuristicsAI` players | 34.8% |
| BC / SFT model | Behavior cloning from winning Saboteur trajectories | ~72% |
| BC + unregularized PPO | PPO fine-tuning without KL constraint | ~70–71% |
| BC + KL-regularized PPO | PPO fine-tuning constrained toward frozen BC policy | 75.6% |

The key finding is:

> Pure PPO was not sufficient for training a strong Saboteur agent from scratch. Behavior cloning gave a strong initial policy, and KL-regularized PPO further improved it while reducing policy drift.

---

## Main Python Files

### `python/ppo_train_dual_role.py`

Main CLI entry point for training. It selects whether to train miner, Saboteur, or miner trust model.

### `python/miner_ppo.py`

Shared RL infrastructure:

- HTTP environment wrapper
- observation encoding
- action encoding
- action-conditioned actor-critic model
- rollout buffer
- GAE calculation
- miner PPO components

### `python/saboteur_ppo.py`

Saboteur PPO fine-tuning logic.  
This file implements:

- loading pretrained BC checkpoint,
- frozen BC model for KL regularization,
- Saboteur sparse terminal reward,
- PPO update loop,
- checkpoint saving.

### `python/collect_saboteur_win_dataset.py`

Collects winning Saboteur expert trajectories for supervised training.

### `python/train_saboteur_bc.py`

Trains the behavior cloning / SFT Saboteur policy.

### `python/evaluate_saboteur_policy.py`

Evaluates a trained Saboteur checkpoint over many games.

### `python/compare_saboteur_policies_in_game.py`

Compares two models on the same game states and prints their top-k decisions.

### `python/miner_trust.py`

Auxiliary trust model for miner-side reasoning.  
This is not the main focus of this branch.

---

## Running the Headless Server

The project includes a headless HTTP server for Python/RL control. It runs a fixed 4-player game where Python controls player `3`, and the other players use `HeuristicsAI`.

Start the Java server:

```powershell
mvn exec:java
```

Test server health:

```powershell
curl http://localhost:8000/health
```

See:

```text
SERVER_API.md
```

for endpoint details.

---

## Suggested Workflow

A typical workflow for this branch is:

```powershell
# 1. Start Java server
mvn exec:java

# 2. Collect expert winning data
python python/collect_saboteur_win_dataset.py --target-win-episodes 200 --max-total-episodes 5000 --out data/saboteur_win_bc_dataset.npz

# 3. Train behavior cloning model
python python/train_saboteur_bc.py --data data/saboteur_win_bc_dataset.npz --out checkpoints/saboteur_bc.pt --epochs 50

# 4. Fine-tune with KL-regularized PPO
python python/ppo_train_dual_role.py --saboteur --pretrained-path checkpoints/saboteur_bc.pt --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10

# 5. Evaluate best checkpoint
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500
```

---

## Notes

- This branch is primarily designed for Saboteur-side training.
- The current best model is not pure PPO. It is behavior cloning followed by KL-regularized PPO fine-tuning.
- Evaluation is performed against three Java rule-based agents.
- Checkpoints and datasets may be large and are usually not committed to Git.
- Later checkpoints are not always better. In current experiments, `update_100` outperformed later checkpoints, suggesting that excessive PPO fine-tuning can still cause policy drift or overfitting.
