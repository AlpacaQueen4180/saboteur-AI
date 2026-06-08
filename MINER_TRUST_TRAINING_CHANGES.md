# Miner Trust Training Changes

## Summary

Updated `python/miner_trust.py` so the miner trust model is no longer trained as an effectively early-only classifier.

The new setup still emphasizes early-game trust inference, because that is when the miner policy has the least role evidence, but it now keeps snapshots from every collected step and assigns training weights by per-episode progress phase.

## What Changed

- Changed trust dataset sampling from every 2 steps to every step.
- Converted collected snapshot progress to per-episode relative progress after each game finishes.
- Added phase-based sample weights:
  - early progress `< 0.25`: `1.00`
  - mid progress `< 0.65`: `0.70`
  - late progress `>= 0.65`: `0.40`
- Applied the phase sample weights to:
  - role classification loss
  - harmful behavior loss
  - early prior regularization loss
- Added sample-weight statistics to the training output.
- Saved the new progress and phase-weighting metadata in the trust checkpoint.
- Changed checkpoint saving to keep the best validation-accuracy epoch instead of always saving the final epoch.
- Added best-epoch validation metrics to the saved checkpoint metadata.

## Why

The previous training output showed `progress max = 0.213`, which meant validation phases were all categorized as early-game under the old `global_step / max_steps_per_game` progress definition.

That is useful for early suspicion, but too narrow for an RL agent that may use trust features throughout the full episode. The new setup keeps the early-game bias while making midgame and late-game states available when games last that long.

Runtime trust features remain compatible with `miner_ppo.py`.

## Checkpoint Selection

`train_miner_trust` now tracks `val_acc` after each epoch and saves the checkpoint from the best epoch to the requested `--trust-save-path`.

The final printed trust examples are generated from the restored best checkpoint, not necessarily the last training epoch.
