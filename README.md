# Saboteur AI — Role-Specific Policy Learning Branch

This branch focuses on training role-specific AI agents for the board game *Saboteur* / 矮人礦坑.

Unlike the original GUI-oriented game repository, this branch is mainly an experimental reinforcement learning branch. The goal is to provide a stable headless Java environment and Python learning interface for training and evaluating learned policies under hidden-role, partially observable, turn-based gameplay.

The project trains two role-specific policies:

```text
Gold Miner policy
Pure reinforcement learning
Action-conditioned PPO
Reward shaping + terminal team reward
Optional MinerTrustNet trust features

Saboteur policy
Rule-based expert winning trajectories
↓
Supervised fine-tuning / behavior cloning
↓
PPO fine-tuning
↓
Final Saboteur policy
```

The central idea is that the two roles require different learning recipes.

- The **Gold Miner** role has a more direct cooperative objective: build useful paths, repair teammates, use information cards, and help the Miner team reach the gold. Therefore, it is trained with pure action-conditioned PPO.
- The **Saboteur** role has sparse and delayed credit assignment: useful sabotage often only matters several turns later. Therefore, it first learns from heuristic winning trajectories through SFT / behavior cloning, then improves through PPO fine-tuning.

---

## Project Overview

The project separates the Java game engine from the Python learning code.

```text
Java game environment
├── Saboteur game rules
├── Hidden roles and private hands
├── Board/path updates
├── Legal action generation
├── HeuristicsAI rule-based players
├── Headless HTTP server
└── Rule-based baseline evaluation

Python training side
├── HTTP environment wrapper
├── Observation encoding
├── Legal-action encoding
├── Action-conditioned actor-critic model
├── Miner PPO training
├── Optional MinerTrustNet trust model
├── Saboteur behavior cloning / SFT
├── Saboteur PPO fine-tuning
└── Evaluation and policy comparison tools
```

The current controlled experiment uses a fixed 4-player setting:

```text
Player 0: Java HeuristicsAI
Player 1: Java HeuristicsAI
Player 2: Java HeuristicsAI
Player 3: Python-controlled learned policy
```

Only one learned player is tested at a time.

For role-conditioned evaluation:

```text
Miner evaluation:
Player 3 is reset as a Gold Miner.

Saboteur evaluation:
Player 3 is reset as the Saboteur.
```

The other three players are controlled by Java rule-based agents.

---

## Environment and Learning Interface

The Java program owns the game rules. Python does not directly modify the board or generate arbitrary moves. Instead, Java exposes a headless HTTP interface and returns the currently legal actions for the focal player.

Typical interaction:

```text
Python calls reset
↓
Java creates a new game
↓
Python receives the focal player's observation
↓
Java enumerates legal actions
↓
Python scores legal actions and selects one
↓
Java applies the action and advances heuristic players
↓
The next focal-player decision state is returned
```

This design makes the learning side similar to a Gym-style environment while keeping all rule validation in Java.

---

## Model Design

### Action-Conditioned Actor-Critic

The learned policy is action-conditioned. Instead of predicting an action from a fixed global action ID space, the model scores only the legal actions returned by the Java engine.

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

A softmax over legal-action scores gives the policy distribution:

```text
π(a_i | s) ∝ exp(fθ(ObsEnc(s), ActEnc(a_i))),  a_i ∈ A_legal(s)
```

Illegal actions are never selected because they are filtered out by the Java game engine before the policy is called.

### Observation Features

The focal player receives only the information visible to that player in a real game.

Observation features include:

```text
role / focal-player information
hand-card features
public board and path features
player tool status
recent public action history
known goal information
optional trust features for Miner
```

### Action Features

Each legal action is encoded as a feature vector containing information such as:

```text
move type
card type
board coordinate
target player
target goal
tool effect
path-distance improvement
discard / play / map / rockfall information
```

This design is especially useful for Saboteur because the set of valid actions changes every turn.

---

## Training Pipeline

The final project uses two parallel role-specific training pipelines.

```text
                 ┌──────────────────────────────┐
                 │ Java Saboteur environment     │
                 │ legal actions + game dynamics │
                 └───────────────┬──────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
   Gold Miner training pipeline          Saboteur training pipeline
   Pure RL / PPO                         SFT / BC followed by RL
              │                                     │
              ▼                                     ▼
   Miner PPO checkpoint                  Saboteur BC checkpoint
              │                                     │
              ▼                                     ▼
   Optional trust-aware PPO              PPO fine-tuning from BC
              │                                     │
              ▼                                     ▼
   Miner evaluation                      Saboteur evaluation
```

The reason for using two different pipelines is that the roles have different learning difficulty.

- The **Miner** role receives denser and more interpretable shaped rewards. PPO can learn cooperative behavior directly.
- The **Saboteur** role receives sparse terminal feedback, and successful sabotage may require delayed strategic effects. Behavior cloning gives the policy a stable initial strategy before PPO.

---

## Miner Training: Pure PPO

The Miner policy is trained directly with action-conditioned PPO.

Unlike the Saboteur policy, the Miner policy does not use expert demonstrations or behavior cloning. This makes the Miner experiment a direct test of whether the action-conditioned PPO design can learn useful cooperative path-building behavior from interaction.

### Miner PPO Objective

The Miner agent is optimized with PPO using:

```text
clipped policy loss
+ value loss
- entropy bonus
```

The policy samples from legal actions during training and uses greedy action selection during evaluation.

### Miner Reward Design

The Miner reward combines terminal game outcome with intermediate shaping.

Terminal reward:

```text
Miner team wins: positive reward
Saboteur wins: negative reward
```

Intermediate shaping encourages useful Miner behavior:

```text
building paths toward the goal
using map cards productively
using rockfall cards productively
repairing blocked teammates
avoiding unnecessary discards
avoiding unproductive actions
```

This reward design makes learning easier than sparse terminal-only PPO while still aligning the policy with the final Miner-team objective.

### MinerTrustNet

The project also includes an optional trust-aware Miner variant.

`MinerTrustNet` reads public action-history features and predicts per-player suspiciousness / harmfulness scores. These scores are appended to the Miner observation as auxiliary trust features.

Two Miner variants are evaluated:

```text
Miner PPO without trust features
    PPO receives neutral trust features.

Miner PPO with trust features
    PPO receives MinerTrustNet-derived suspiciousness / harmfulness features.
```

This comparison isolates the effect of learned trust information while keeping the main PPO architecture unchanged.

---

## Saboteur Training: SFT Followed by PPO

Initial experiments with pure PPO were unstable for the Saboteur role.

The main reasons were:

- Saboteur rewards are sparse and delayed.
- The Saboteur must prevent progress rather than directly build toward the goal.
- Many useful sabotage actions only affect the game outcome several turns later.
- Pure PPO frequently receives losing trajectories, making credit assignment difficult.
- Unregularized PPO fine-tuning can drift away from a good supervised policy.

To address this, the Saboteur policy uses a two-stage approach.

```text
Rule-based expert winning trajectories
        ↓
BC dataset of state-action pairs
        ↓
Supervised learning / behavior cloning
        ↓
Pretrained Saboteur policy
        ↓
PPO fine-tuning
        ↓
Final Saboteur policy
```

### 1. Collect Saboteur Winning Data

Script:

```text
python/collect_saboteur_win_dataset.py
```

This script runs a rule-based Saboteur expert against three rule-based players.

It only stores trajectories where the Saboteur side wins. Each sample contains:

```text
obs          encoded observation
action_feats encoded legal actions
mask         legal action mask
label        expert action index
```

Example command:

```powershell
python python/collect_saboteur_win_dataset.py --target-win-episodes 200 --max-total-episodes 5000 --out data/saboteur_win_bc_dataset.npz
```

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

### 3. PPO Fine-Tuning

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
+ policy regularization toward the pretrained BC policy
```

The Saboteur reward is sparse and terminal-oriented:

```text
Non-terminal step: 0
Saboteur win: +1
Saboteur lose: -1
Early lose: additional small penalty
```

Example command:

```powershell
python python/ppo_train_dual_role.py --saboteur --pretrained-path checkpoints/saboteur_bc.pt --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10
```

The best current Saboteur checkpoint from evaluation was:

```text
checkpoints/saboteur_bc_ppo_kl_http_update_100.pt
```

with:

```text
BC + PPO Saboteur win rate: 75.6%
```

over 500 deterministic evaluation games.

---

## Evaluation

Evaluation is role-conditioned. Player 3 is forced into the target role, and the learned policy is evaluated against three Java heuristic agents.

### Miner Evaluation

During Miner evaluation:

```text
Player 3: learned Gold Miner policy
Players 0, 1, 2: Java HeuristicsAI
Player 3 role: forced to Gold Miner
Evaluation action selection: greedy
```

Miner evaluation reports:

```text
valid episodes
Miner wins
Miner win rate
winner counts
average episode length
action type statistics
```

The reported Miner results use the update-300 PPO checkpoint and 3000 held-out greedy evaluation episodes.

### Saboteur Evaluation

During Saboteur evaluation:

```text
Player 3: learned Saboteur policy
Players 0, 1, 2: Java HeuristicsAI
Player 3 role: forced to Saboteur
Evaluation action selection: greedy
```

Script:

```text
python/evaluate_saboteur_policy.py
```

Example command:

```powershell
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500
```

To save the result into a text file:

```powershell
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500 | Tee-Object -FilePath eval_logs/eval_kl_update_100.txt
```

Saboteur evaluation reports:

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
miner win rate: 65.2%
saboteur win rate: 34.8%
```

This baseline is not exactly the same setting as the Python model evaluation, because the model evaluation fixes player 3 to the target role. However, it provides a useful reference for the strength of the pure rule-based setup.

---

## Important Results

| Setting | Training Method | Evaluation Target | Observed Result |
|---|---|---|---:|
| Four Java `HeuristicsAI` players | Rule-based baseline | Miner side | 65.2% Miner win rate |
| Four Java `HeuristicsAI` players | Rule-based baseline | Saboteur side | 34.8% Saboteur win rate |
| Miner PPO without trust features | Pure action-conditioned PPO | Miner | 75.47% Miner win rate |
| Miner PPO with trust features | Pure action-conditioned PPO + MinerTrustNet features | Miner | 75.97% Miner win rate |
| Saboteur SFT / BC | Behavior cloning from winning Saboteur trajectories | Saboteur | ~72% Saboteur win rate |
| Saboteur SFT + PPO fine-tuning | BC initialization followed by PPO | Saboteur | 75.6% Saboteur win rate |

The main results are:

- The learned Miner policy improves over the heuristic Miner-side baseline using pure PPO.
- MinerTrustNet gives a small additional improvement over the no-trust Miner PPO baseline.
- The Saboteur policy benefits strongly from SFT / behavior cloning before PPO.
- The project therefore supports the main role-specific conclusion: **Miner can be trained with pure RL, while Saboteur requires SFT warm start before RL fine-tuning.**

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

This prints the top-k actions from each model at each Saboteur decision state. The tool is useful for analyzing whether PPO fine-tuning changed the model's strategy.

In observed cases, most decisions remained similar to the BC policy, while differences mainly appeared in:

```text
which player to block
which card to discard
small local action preferences
```

---

## Main Python Files

### `python/ppo_train_dual_role.py`

Main CLI entry point for training.

It selects whether to train:

```text
Miner PPO
Saboteur PPO
Miner trust model
```

### `python/miner_ppo.py`

Shared RL infrastructure and Miner PPO implementation.

This file includes:

```text
HTTP environment wrapper
observation encoding
action encoding
action-conditioned actor-critic model
rollout buffer
GAE calculation
Miner PPO training components
```

### `python/miner_trust.py`

Auxiliary trust model for Miner-side reasoning.

This file implements MinerTrustNet, which converts public action-history features into suspiciousness / harmfulness trust features for other players.

### `python/evaluate_miner_policy.py`

Evaluates trained Miner checkpoints against Java heuristic opponents.

### `python/saboteur_ppo.py`

Saboteur PPO fine-tuning logic.

This file implements:

```text
loading pretrained BC checkpoint
frozen BC model for policy regularization
Saboteur sparse terminal reward
PPO update loop
checkpoint saving
```

### `python/collect_saboteur_win_dataset.py`

Collects winning Saboteur expert trajectories for supervised training.

### `python/train_saboteur_bc.py`

Trains the behavior cloning / SFT Saboteur policy.

### `python/evaluate_saboteur_policy.py`

Evaluates a trained Saboteur checkpoint over many games.

### `python/compare_saboteur_policies_in_game.py`

Compares two Saboteur models on the same game states and prints their top-k decisions.

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

# 2. Train Miner directly with PPO
python python/ppo_train_dual_role.py --miner --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10

# 3. Optionally train / use MinerTrustNet features
python python/ppo_train_dual_role.py --miner --use-trust --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10

# 4. Evaluate Miner checkpoint
python python/evaluate_miner_policy.py --checkpoint checkpoints/miner_ppo_update_300.pt --episodes 3000

# 5. Collect Saboteur expert winning data
python python/collect_saboteur_win_dataset.py --target-win-episodes 200 --max-total-episodes 5000 --out data/saboteur_win_bc_dataset.npz

# 6. Train Saboteur behavior cloning model
python python/train_saboteur_bc.py --data data/saboteur_win_bc_dataset.npz --out checkpoints/saboteur_bc.pt --epochs 50

# 7. Fine-tune Saboteur with PPO
python python/ppo_train_dual_role.py --saboteur --pretrained-path checkpoints/saboteur_bc.pt --updates 300 --rollout-steps 512 --save-every 20 --debug-every 10

# 8. Evaluate best Saboteur checkpoint
python python/evaluate_saboteur_policy.py --checkpoint checkpoints/saboteur_bc_ppo_kl_http_update_100.pt --episodes 500
```

---

## Main Takeaways

1. **Legal-action scoring works well.**  
   Encoding and scoring only the currently legal moves avoids a large fixed global action space.

2. **The two roles require different training recipes.**  
   Miner can learn directly with pure action-conditioned PPO, while Saboteur benefits from SFT / behavior cloning before PPO.

3. **Trust is useful but modest.**  
   MinerTrustNet slightly improves Miner performance, suggesting that public action history contains useful but noisy opponent information.

4. **SFT stabilizes difficult hidden-role learning.**  
   For Saboteur, behavior cloning provides a strong initial policy and makes later RL fine-tuning more stable.

5. **Later checkpoints are not always better.**  
   In current Saboteur experiments, the best PPO checkpoint was not necessarily the final checkpoint, suggesting that excessive PPO fine-tuning can still cause drift or overfitting.

---

## Limitations and Future Work

Current limitations:

```text
The experiments focus on the basic 4-player Saboteur setting.
Only one learned player is evaluated at a time.
Opponents are fixed Java heuristic agents.
The trust improvement is small.
Evaluation currently uses role-conditioned player-3 testing.
Multiple random seeds would strengthen the conclusion.
```

Future work may include:

```text
larger games
multiple Saboteurs
self-play
adaptive learned opponents
multi-role unified training
stronger opponent modeling
larger-scale statistical evaluation
```

---

## Conclusion

This project implements a role-specific reinforcement learning pipeline for the hidden-role card game *Saboteur*.

The key design is to let the neural policy score legal actions generated by the Java game engine instead of predicting from a fixed global action space.

The final training setup uses two parallel role-specific pipelines:

```text
Miner:
pure action-conditioned PPO

Saboteur:
SFT / behavior cloning followed by PPO fine-tuning
```

The learned Miner policy improves over the heuristic Miner baseline using pure RL. The learned Saboteur policy improves over the heuristic Saboteur baseline by first imitating successful Saboteur trajectories and then applying PPO fine-tuning.

Overall, the project shows that role-specific training is important in hidden-role games: cooperative Miner behavior and adversarial Saboteur behavior benefit from different learning strategies.
