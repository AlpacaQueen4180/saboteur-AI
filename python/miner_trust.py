import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTROLLED_PLAYER = 3
TARGET_ROLE = "GOLD_MINER"

SAMPLE_INTERVAL_STEPS = 1

# Train on the full collected episode, but keep the trust model early-game
# biased because that is when uncertainty matters most to the miner policy.
EARLY_PROGRESS_CUTOFF = 0.25
MID_PROGRESS_CUTOFF = 0.65
EARLY_SAMPLE_WEIGHT = 1.00
MID_SAMPLE_WEIGHT = 0.70
LATE_SAMPLE_WEIGHT = 0.40

# From a miner perspective, among players 0/1/2 there is usually 1 saboteur.
PRIOR_P_SABOTEUR = 1.0 / 3.0

# Keep this mild. Low-evidence samples should not become too confident,
# but rule-based opponents may still reveal themselves early.
EARLY_PRIOR_COEF = 0.03

# If a player has made this many observable actions, the role target can
# move close to the true hidden-role label.
EVIDENCE_FULL_COUNT = 4.0

# Loss weights.
ROLE_LOSS_COEF = 1.0
HARMFUL_LOSS_COEF = 0.7

MOVE_TYPES = [
    "DISCARD",
    "PLAY_PATH",
    "PLAY_PLAYER",
    "PLAY_MAP",
    "PLAY_ROCKFALL",
]

CARD_TYPES = [
    "PATHWAY",
    "DEADEND",
    "MAP",
    "ROCKFALL",
    "BLOCK",
    "REPAIR",
]

TOOLS = [
    "CART",
    "LANTERN",
    "PICKAXE",
]


# ============================================================
# Utils
# ============================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def sigmoid_np(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def role_of_controlled_player(state: Dict[str, Any]) -> str:
    return get_nested(state, ["observation", "private", "role"], "")


def player_role_from_full_state(state: Dict[str, Any], player_index: int) -> str:
    players = state.get("fullState", {}).get("players", [])
    for p in players:
        if safe_int(p.get("index"), -1) == player_index:
            return str(p.get("role", "UNKNOWN"))
    return "UNKNOWN"


def get_player_status(state: Dict[str, Any], player_index: int) -> Dict[str, Any]:
    statuses = get_nested(state, ["observation", "playerStatus"], [])
    for p in statuses:
        if safe_int(p.get("index"), -1) == player_index:
            return p
    return {}


def get_path_features(state: Dict[str, Any]) -> Dict[str, Any]:
    return get_nested(state, ["observation", "board", "path_features"], {})


# ============================================================
# HTTP Env
# ============================================================

class SaboteurHttpEnv:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self) -> Dict[str, Any]:
        r = requests.post(self.base_url + "/reset", json={}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset_until_role(self, target_role: str) -> Dict[str, Any]:
        tries = 0

        while True:
            tries += 1
            state = self.reset()
            role = role_of_controlled_player(state)

            if role == target_role:
                if tries > 1:
                    print(f"[reset_until_role] target={target_role}, tries={tries}")
                return state

            if tries % 20 == 0:
                print(f"[reset_until_role] waiting target={target_role}, tries={tries}, last_role={role}")

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.base_url + "/step", json=action, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ============================================================
# Data collection policy
# ============================================================

def choose_data_collection_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only used to advance games while collecting trust-training data.
    This is not PPO.
    """
    actions = state.get("legalActions", [])
    if not actions:
        raise RuntimeError("No legal actions available.")

    best_path = None
    best_delta = -999.0

    for a in actions:
        if a.get("type") == "PLAY_PATH":
            delta = safe_float(a.get("delta_target_distance"), 0.0)
            if delta > best_delta:
                best_delta = delta
                best_path = a

    if best_path is not None and best_delta > 0:
        return best_path

    pf = get_path_features(state)
    known_goals = pf.get("known_goals", ["UNKNOWN", "UNKNOWN", "UNKNOWN"])
    gold_known = bool(pf.get("gold_known", False))

    if not gold_known:
        for a in actions:
            if a.get("type") == "PLAY_MAP":
                gi = safe_int(a.get("goal_index", a.get("goalIndex", -1)), -1)
                if 0 <= gi < 3 and known_goals[gi] == "UNKNOWN":
                    return a

    for a in actions:
        if a.get("type") == "PLAY_PLAYER" and a.get("card_type", a.get("cardType")) == "REPAIR":
            return a

    if best_path is not None:
        return best_path

    for a in actions:
        if a.get("type") == "DISCARD":
            return a

    return actions[0]


# ============================================================
# Feature aggregation
# ============================================================

FEATURE_NAMES = [
    # Move counts
    "discard_count",
    "play_path_count",
    "play_player_count",
    "play_map_count",
    "play_rockfall_count",

    # Card type counts
    "pathway_card_count",
    "deadend_card_count",
    "map_card_count",
    "rockfall_card_count",
    "block_card_count",
    "repair_card_count",

    # Action-specific counts
    "block_count",
    "repair_count",
    "targeted_block_count",
    "targeted_repair_count",

    # Path / rockfall semantic statistics
    # These become more useful if Java events are enriched later.
    "sum_delta_target_distance",
    "avg_delta_target_distance",
    "positive_delta_count",
    "negative_delta_count",

    "sum_remove_delta",
    "avg_remove_delta",
    "positive_remove_delta_count",

    "sum_ideal_fill_delta",
    "avg_ideal_fill_delta",
    "positive_ideal_fill_count",

    # Current public status
    "hand_size",
    "is_sabotaged",
    "cart_broken",
    "lantern_broken",
    "pickaxe_broken",

    # Evidence amount
    "player_event_count",
    "global_event_count",
    "evidence_ratio",

    # Temporal context
    "global_step",
    "progress_ratio",

    # Board / game context
    "deck_size",
    "target_distance",
    "reachable_count",
    "destroyable_count",
    "frontier_count",
    "gold_known",
]

FEATURE_DIM = len(FEATURE_NAMES)


class PlayerFeatureAccumulator:
    def __init__(self) -> None:
        self.data = {name: 0.0 for name in FEATURE_NAMES}
        self.delta_values: List[float] = []
        self.remove_values: List[float] = []
        self.ideal_values: List[float] = []

    def add_actor_event(self, event: Dict[str, Any]) -> None:
        move_type = str(event.get("type", ""))
        self.data["player_event_count"] += 1.0

        if move_type == "DISCARD":
            self.data["discard_count"] += 1.0
        elif move_type == "PLAY_PATH":
            self.data["play_path_count"] += 1.0
        elif move_type == "PLAY_PLAYER":
            self.data["play_player_count"] += 1.0
        elif move_type == "PLAY_MAP":
            self.data["play_map_count"] += 1.0
        elif move_type == "PLAY_ROCKFALL":
            self.data["play_rockfall_count"] += 1.0

        card = event.get("card")
        card_type = card.get("type") if isinstance(card, dict) else None

        if card_type == "PATHWAY":
            self.data["pathway_card_count"] += 1.0
        elif card_type == "DEADEND":
            self.data["deadend_card_count"] += 1.0
        elif card_type == "MAP":
            self.data["map_card_count"] += 1.0
        elif card_type == "ROCKFALL":
            self.data["rockfall_card_count"] += 1.0
        elif card_type == "BLOCK":
            self.data["block_card_count"] += 1.0
            self.data["block_count"] += 1.0
        elif card_type == "REPAIR":
            self.data["repair_card_count"] += 1.0
            self.data["repair_count"] += 1.0

        delta = safe_float(event.get("delta_target_distance"), 0.0)
        remove_delta = safe_float(event.get("remove_delta"), 0.0)
        ideal_delta = safe_float(event.get("ideal_fill_delta"), 0.0)

        if abs(delta) > 1e-8:
            self.delta_values.append(delta)
            self.data["sum_delta_target_distance"] += delta
            if delta > 0:
                self.data["positive_delta_count"] += 1.0
            elif delta < 0:
                self.data["negative_delta_count"] += 1.0

        if abs(remove_delta) > 1e-8:
            self.remove_values.append(remove_delta)
            self.data["sum_remove_delta"] += remove_delta
            if remove_delta > 0:
                self.data["positive_remove_delta_count"] += 1.0

        if abs(ideal_delta) > 1e-8:
            self.ideal_values.append(ideal_delta)
            self.data["sum_ideal_fill_delta"] += ideal_delta
            if ideal_delta > 0:
                self.data["positive_ideal_fill_count"] += 1.0

    def add_targeted_event(self, card_type: str) -> None:
        if card_type == "BLOCK":
            self.data["targeted_block_count"] += 1.0
        elif card_type == "REPAIR":
            self.data["targeted_repair_count"] += 1.0

    @staticmethod
    def extract_target_player(event: Dict[str, Any]) -> int:
        if "targetPlayer" in event:
            return safe_int(event.get("targetPlayer"), -1)
        if "target_player" in event:
            return safe_int(event.get("target_player"), -1)

        args = event.get("args", [])
        move_type = str(event.get("type", ""))
        if move_type == "PLAY_PLAYER" and isinstance(args, list) and len(args) >= 1:
            return safe_int(args[0], -1)

        return -1

    def update_context(
        self,
        state: Dict[str, Any],
        player_index: int,
        global_step: int,
        max_steps_per_game: int,
        global_event_count: int,
    ) -> None:
        status = get_player_status(state, player_index)

        self.data["hand_size"] = safe_float(status.get("handSize"), 0.0)
        self.data["is_sabotaged"] = 1.0 if bool(status.get("sabotaged", False)) else 0.0

        blocked = set(status.get("blockedTools", []))
        self.data["cart_broken"] = 1.0 if "CART" in blocked else 0.0
        self.data["lantern_broken"] = 1.0 if "LANTERN" in blocked else 0.0
        self.data["pickaxe_broken"] = 1.0 if "PICKAXE" in blocked else 0.0

        pf = get_path_features(state)

        self.data["global_step"] = float(global_step)
        self.data["progress_ratio"] = float(global_step) / max(1.0, float(max_steps_per_game))
        self.data["global_event_count"] = float(global_event_count)

        player_events = self.data["player_event_count"]
        self.data["evidence_ratio"] = player_events / max(1.0, float(global_event_count))

        self.data["deck_size"] = safe_float(get_nested(state, ["observation", "public", "deckSize"], 0.0), 0.0)
        self.data["target_distance"] = safe_float(pf.get("target_distance"), 0.0)
        self.data["reachable_count"] = safe_float(pf.get("reachable_count"), 0.0)
        self.data["destroyable_count"] = safe_float(pf.get("destroyable_count"), 0.0)
        self.data["frontier_count"] = safe_float(pf.get("frontier_count"), 0.0)
        self.data["gold_known"] = 1.0 if bool(pf.get("gold_known", False)) else 0.0

    def raw_harm_score(self) -> float:
        """
        Weak behavioral-threat target.

        This is NOT used as a hand-written trust policy.
        It is only a weak supervision signal for a separate p_harmful head.
        Miner PPO will later receive p_harmful as a soft feature, not a hard rule.
        """
        d = self.data

        harmful = 0.0
        helpful = 0.0

        # Directly suspicious public actions.
        harmful += 1.00 * d["block_count"]
        harmful += 0.35 * d["play_rockfall_count"]
        harmful += 0.15 * d["discard_count"]

        # If Java events later include semantic deltas, these become important.
        harmful += 0.90 * d["negative_delta_count"]
        harmful += 0.90 * d["positive_remove_delta_count"]

        # Positive / cooperative behavior.
        helpful += 0.65 * d["positive_delta_count"]
        helpful += 0.45 * d["repair_count"]
        helpful += 0.55 * d["positive_ideal_fill_count"]
        helpful += 0.20 * d["play_map_count"]

        # Being targeted by block may indicate others suspect this player,
        # but it can also be noisy. Keep it weak.
        harmful += 0.15 * d["targeted_block_count"]
        helpful += 0.10 * d["targeted_repair_count"]

        # Normalize by evidence amount. This prevents high confidence from
        # a single weak action.
        player_events = max(1.0, d["player_event_count"])
        score = (harmful - helpful) / max(1.0, np.sqrt(player_events))

        return float(score)

    def weak_harmful_label(self) -> float:
        """
        Continuous weak label in [0, 1].
        p_harmful is behavior threat, not hidden role.
        """
        score = self.raw_harm_score()
        return float(sigmoid_np(score))

    def to_vector(self) -> np.ndarray:
        data = dict(self.data)

        if self.delta_values:
            data["avg_delta_target_distance"] = data["sum_delta_target_distance"] / len(self.delta_values)
        if self.remove_values:
            data["avg_remove_delta"] = data["sum_remove_delta"] / len(self.remove_values)
        if self.ideal_values:
            data["avg_ideal_fill_delta"] = data["sum_ideal_fill_delta"] / len(self.ideal_values)

        x = np.asarray([data[name] for name in FEATURE_NAMES], dtype=np.float32)

        player_events = max(1.0, data["player_event_count"])
        global_events = max(1.0, data["global_event_count"])

        count_names_player_norm = [
            "discard_count",
            "play_path_count",
            "play_player_count",
            "play_map_count",
            "play_rockfall_count",
            "pathway_card_count",
            "deadend_card_count",
            "map_card_count",
            "rockfall_card_count",
            "block_card_count",
            "repair_card_count",
            "block_count",
            "repair_count",
            "positive_delta_count",
            "negative_delta_count",
            "positive_remove_delta_count",
            "positive_ideal_fill_count",
        ]

        for name in count_names_player_norm:
            x[FEATURE_NAMES.index(name)] /= player_events

        for name in ["targeted_block_count", "targeted_repair_count"]:
            x[FEATURE_NAMES.index(name)] /= global_events

        x[FEATURE_NAMES.index("hand_size")] /= 6.0
        x[FEATURE_NAMES.index("player_event_count")] /= 50.0
        x[FEATURE_NAMES.index("global_event_count")] /= 100.0
        x[FEATURE_NAMES.index("global_step")] /= 80.0
        x[FEATURE_NAMES.index("deck_size")] /= 70.0
        x[FEATURE_NAMES.index("target_distance")] /= 12.0
        x[FEATURE_NAMES.index("reachable_count")] /= 45.0
        x[FEATURE_NAMES.index("destroyable_count")] /= 45.0
        x[FEATURE_NAMES.index("frontier_count")] /= 45.0

        for name in [
            "sum_delta_target_distance",
            "avg_delta_target_distance",
            "sum_remove_delta",
            "avg_remove_delta",
            "sum_ideal_fill_delta",
            "avg_ideal_fill_delta",
        ]:
            x[FEATURE_NAMES.index(name)] /= 12.0

        return x


def update_accumulators_from_events(
    accumulators: Dict[int, PlayerFeatureAccumulator],
    state: Dict[str, Any],
) -> int:
    events = get_nested(state, ["observation", "events"], [])
    event_count = 0

    for event in events:
        event_count += 1

        actor = safe_int(event.get("playerIndex"), -1)
        card = event.get("card")
        card_type = card.get("type") if isinstance(card, dict) else None

        if actor in accumulators:
            accumulators[actor].add_actor_event(event)

        target = PlayerFeatureAccumulator.extract_target_player(event)
        if target in accumulators and actor != target:
            accumulators[target].add_targeted_event(card_type or "")

    return event_count


# ============================================================
# Dataset collection
# ============================================================

@dataclass
class TrustDataset:
    x: np.ndarray
    y_role: np.ndarray
    y_harmful: np.ndarray
    progress: np.ndarray
    player_event_count: np.ndarray
    sample_weight: np.ndarray


def sample_weight_from_progress(progress: float) -> float:
    if progress < EARLY_PROGRESS_CUTOFF:
        return EARLY_SAMPLE_WEIGHT
    if progress < MID_PROGRESS_CUTOFF:
        return MID_SAMPLE_WEIGHT
    return LATE_SAMPLE_WEIGHT


def add_snapshot_samples(
    xs: List[np.ndarray],
    y_roles: List[int],
    y_harmfuls: List[float],
    progresses: List[float],
    player_event_counts: List[float],
    accumulators: Dict[int, PlayerFeatureAccumulator],
    state: Dict[str, Any],
    global_step: int,
    max_steps_per_game: int,
    global_event_count: int,
) -> None:
    for player_index in [0, 1, 2]:
        role = player_role_from_full_state(state, player_index)
        if role not in ["GOLD_MINER", "SABOTEUR"]:
            continue

        acc = accumulators[player_index]
        acc.update_context(
            state=state,
            player_index=player_index,
            global_step=global_step,
            max_steps_per_game=max_steps_per_game,
            global_event_count=global_event_count,
        )

        role_label = 1 if role == "SABOTEUR" else 0
        harmful_label = acc.weak_harmful_label()

        # Store the absolute step for now. collect_one_game converts this to
        # per-episode relative progress after the final episode length is known.
        progress = float(global_step)
        raw_player_event_count = float(acc.data["player_event_count"])

        xs.append(acc.to_vector())
        y_roles.append(role_label)
        y_harmfuls.append(harmful_label)
        progresses.append(progress)
        player_event_counts.append(raw_player_event_count)


def collect_one_game(
    env: SaboteurHttpEnv,
    max_steps_per_game: int,
) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[float], List[float], Dict[str, Any]]:
    state = env.reset_until_role(TARGET_ROLE)

    accumulators = {
        0: PlayerFeatureAccumulator(),
        1: PlayerFeatureAccumulator(),
        2: PlayerFeatureAccumulator(),
    }

    xs: List[np.ndarray] = []
    y_roles: List[int] = []
    y_harmfuls: List[float] = []
    progresses: List[float] = []
    player_event_counts: List[float] = []
    sample_weights: List[float] = []

    global_step = 0
    global_event_count = 0

    global_event_count += update_accumulators_from_events(accumulators, state)

    add_snapshot_samples(
        xs, y_roles, y_harmfuls, progresses, player_event_counts,
        accumulators, state,
        global_step, max_steps_per_game, global_event_count,
    )

    done = bool(state.get("done", False))

    while not done and global_step < max_steps_per_game:
        actions = state.get("legalActions", [])
        if not actions:
            break

        action = choose_data_collection_action(state)

        try:
            state = env.step(action)
        except requests.HTTPError as e:
            print("[collect_one_game] HTTP step error:", e)
            break

        global_step += 1
        global_event_count += update_accumulators_from_events(accumulators, state)

        add_snapshot_samples(
            xs, y_roles, y_harmfuls, progresses, player_event_counts,
            accumulators, state,
            global_step, max_steps_per_game, global_event_count,
        )

        done = bool(state.get("done", False))

    add_snapshot_samples(
        xs, y_roles, y_harmfuls, progresses, player_event_counts,
        accumulators, state,
        global_step, max_steps_per_game, global_event_count,
    )

    # Convert stored absolute step positions into per-episode progress for
    # phase metrics and sample weighting. Runtime features still use the
    # original max_steps_per_game scale inside PlayerFeatureAccumulator.
    final_step = max(1.0, float(global_step))
    progresses[:] = [min(1.0, float(step) / final_step) for step in progresses]
    sample_weights.extend(sample_weight_from_progress(p) for p in progresses)

    return xs, y_roles, y_harmfuls, progresses, player_event_counts, sample_weights, state


def collect_dataset(
    base_url: str,
    num_games: int,
    max_steps_per_game: int,
) -> TrustDataset:
    env = SaboteurHttpEnv(base_url)

    all_x: List[np.ndarray] = []
    all_y_role: List[int] = []
    all_y_harmful: List[float] = []
    all_progress: List[float] = []
    all_player_event_counts: List[float] = []
    all_sample_weights: List[float] = []

    miner_labels = 0
    sab_labels = 0

    for game_idx in range(1, num_games + 1):
        xs, y_roles, y_harmfuls, progresses, player_event_counts, sample_weights, final_state = collect_one_game(
            env, max_steps_per_game
        )

        all_x.extend(xs)
        all_y_role.extend(y_roles)
        all_y_harmful.extend(y_harmfuls)
        all_progress.extend(progresses)
        all_player_event_counts.extend(player_event_counts)
        all_sample_weights.extend(sample_weights)

        miner_labels += sum(1 for y in y_roles if y == 0)
        sab_labels += sum(1 for y in y_roles if y == 1)

        if game_idx % 20 == 0:
            print(
                f"[collect] games={game_idx}/{num_games} "
                f"samples={len(all_y_role)} "
                f"miner_labels={miner_labels} "
                f"saboteur_labels={sab_labels} "
                f"last_done={final_state.get('done')} "
                f"last_winner={final_state.get('winner')}"
            )

    return TrustDataset(
        x=np.asarray(all_x, dtype=np.float32),
        y_role=np.asarray(all_y_role, dtype=np.float32),
        y_harmful=np.asarray(all_y_harmful, dtype=np.float32),
        progress=np.asarray(all_progress, dtype=np.float32),
        player_event_count=np.asarray(all_player_event_counts, dtype=np.float32),
        sample_weight=np.asarray(all_sample_weights, dtype=np.float32),
    )


# ============================================================
# Model
# ============================================================

class MinerTrustNet(nn.Module):
    """
    Dual-head opponent assessment model.

    role_logit:
        Hidden-role suspicion. label 1 = SABOTEUR.

    harmful_logit:
        Behavioral threat. This is trained from weak public-behavior labels.
    """
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        self.role_head = nn.Linear(64, 1)
        self.harmful_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        role_logit = self.role_head(z).squeeze(-1)
        harmful_logit = self.harmful_head(z).squeeze(-1)
        return role_logit, harmful_logit


def trust_score_from_role_logit(role_logit: torch.Tensor) -> torch.Tensor:
    p_sab = torch.sigmoid(role_logit)
    return 1.0 - 2.0 * p_sab


def build_trust_checkpoint(
    model: nn.Module,
    input_dim: int,
    epoch: int,
    selection_metric: str,
    selection_score: float,
    val_metrics: Dict[str, float],
    val_role_bce: float,
    val_harm_bce: float,
    val_harm_mae: float,
) -> Dict[str, Any]:
    return {
        "model": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "input_dim": input_dim,
        "feature_names": FEATURE_NAMES,
        "target_role": "role label 1 = SABOTEUR, role label 0 = GOLD_MINER",
        "target_harmful": "weak behavioral threat label in [0, 1]",
        "trust_score_definition": "trust_score = 1 - 2 * sigmoid(role_logit)",
        "p_saboteur_definition": "p_saboteur = sigmoid(role_logit)",
        "p_harmful_definition": "p_harmful = sigmoid(harmful_logit)",
        "temporal": True,
        "evidence_weighted": True,
        "dual_head": True,
        "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
        "progress_definition": "per-episode relative progress for training metrics and sample weights",
        "phase_sample_weighting": True,
        "early_progress_cutoff": EARLY_PROGRESS_CUTOFF,
        "mid_progress_cutoff": MID_PROGRESS_CUTOFF,
        "early_sample_weight": EARLY_SAMPLE_WEIGHT,
        "mid_sample_weight": MID_SAMPLE_WEIGHT,
        "late_sample_weight": LATE_SAMPLE_WEIGHT,
        "early_prior_coef": EARLY_PRIOR_COEF,
        "prior_p_saboteur": PRIOR_P_SABOTEUR,
        "evidence_full_count": EVIDENCE_FULL_COUNT,
        "role_loss_coef": ROLE_LOSS_COEF,
        "harmful_loss_coef": HARMFUL_LOSS_COEF,
        "saved_epoch": epoch,
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "val_metrics": val_metrics,
        "val_role_bce": val_role_bce,
        "val_harm_bce": val_harm_bce,
        "val_harm_mae": val_harm_mae,
    }


# ============================================================
# Train / Eval
# ============================================================

def split_dataset(dataset: TrustDataset, val_ratio: float = 0.2) -> Tuple[TrustDataset, TrustDataset]:
    n = len(dataset.y_role)
    indices = np.arange(n)
    np.random.shuffle(indices)

    val_n = max(1, int(n * val_ratio))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    return (
        TrustDataset(
            dataset.x[train_idx],
            dataset.y_role[train_idx],
            dataset.y_harmful[train_idx],
            dataset.progress[train_idx],
            dataset.player_event_count[train_idx],
            dataset.sample_weight[train_idx],
        ),
        TrustDataset(
            dataset.x[val_idx],
            dataset.y_role[val_idx],
            dataset.y_harmful[val_idx],
            dataset.progress[val_idx],
            dataset.player_event_count[val_idx],
            dataset.sample_weight[val_idx],
        ),
    )


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    labels_f = labels.float()

    tp = ((preds == 1) & (labels_f == 1)).sum().item()
    tn = ((preds == 0) & (labels_f == 0)).sum().item()
    fp = ((preds == 1) & (labels_f == 0)).sum().item()
    fn = ((preds == 0) & (labels_f == 1)).sum().item()

    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    miner_acc = tn / max(1, tn + fp)
    sab_acc = tp / max(1, tp + fn)

    return {
        "accuracy": acc,
        "saboteur_precision": precision,
        "saboteur_recall": recall,
        "miner_accuracy": miner_acc,
        "saboteur_accuracy": sab_acc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def phase_metrics(
    model: nn.Module,
    x: torch.Tensor,
    y_role: torch.Tensor,
    progress: torch.Tensor,
    player_event_count: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    phases = {
        "early": progress < 0.25,
        "mid": (progress >= 0.25) & (progress < 0.65),
        "late": progress >= 0.65,
        "low_evidence": player_event_count < 2.0,
        "enough_evidence": player_event_count >= 2.0,
    }

    out: Dict[str, Dict[str, float]] = {}

    model.eval()
    with torch.no_grad():
        role_logits, _ = model(x)

    for name, mask in phases.items():
        if mask.sum().item() == 0:
            continue
        out[name] = metrics_from_logits(role_logits[mask], y_role[mask])

    return out


def train_miner_trust(
    base_url: str,
    num_games: int,
    max_steps_per_game: int,
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: str,
) -> None:
    print("Collecting dual-head Miner trust dataset...")
    print(f"target role: controlled player must be {TARGET_ROLE}")
    print(f"num_games={num_games}, max_steps_per_game={max_steps_per_game}")
    print(f"sample_interval_steps={SAMPLE_INTERVAL_STEPS}")
    print(
        "phase sample weights: "
        f"early(<{EARLY_PROGRESS_CUTOFF})={EARLY_SAMPLE_WEIGHT}, "
        f"mid(<{MID_PROGRESS_CUTOFF})={MID_SAMPLE_WEIGHT}, "
        f"late={LATE_SAMPLE_WEIGHT}"
    )
    print(f"prior_p_saboteur={PRIOR_P_SABOTEUR:.3f}")
    print(f"evidence_full_count={EVIDENCE_FULL_COUNT}")

    dataset = collect_dataset(
        base_url=base_url,
        num_games=num_games,
        max_steps_per_game=max_steps_per_game,
    )

    if len(dataset.y_role) == 0:
        raise RuntimeError("No samples collected.")

    print("Dataset collected.")
    print("x shape:", dataset.x.shape)
    print("y_role shape:", dataset.y_role.shape)
    print("y_harmful shape:", dataset.y_harmful.shape)
    print("progress shape:", dataset.progress.shape)
    print("player_event_count shape:", dataset.player_event_count.shape)
    print("feature_dim:", FEATURE_DIM)
    print("role labels: miner=", int((dataset.y_role == 0).sum()), "saboteur=", int((dataset.y_role == 1).sum()))
    print(
        "harmful label: "
        f"mean={dataset.y_harmful.mean():.3f}, "
        f"min={dataset.y_harmful.min():.3f}, "
        f"max={dataset.y_harmful.max():.3f}"
    )
    print(
        "progress: "
        f"mean={dataset.progress.mean():.3f}, "
        f"min={dataset.progress.min():.3f}, "
        f"max={dataset.progress.max():.3f}"
    )
    print(
        "player_event_count: "
        f"mean={dataset.player_event_count.mean():.3f}, "
        f"min={dataset.player_event_count.min():.3f}, "
        f"max={dataset.player_event_count.max():.3f}"
    )
    print(
        "sample_weight: "
        f"mean={dataset.sample_weight.mean():.3f}, "
        f"min={dataset.sample_weight.min():.3f}, "
        f"max={dataset.sample_weight.max():.3f}"
    )

    train_set, val_set = split_dataset(dataset, val_ratio=0.2)

    model = MinerTrustNet(input_dim=dataset.x.shape[1]).to(DEVICE)

    train_pos = max(1.0, float((train_set.y_role == 1).sum()))
    train_neg = max(1.0, float((train_set.y_role == 0).sum()))
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32, device=DEVICE)

    role_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    harmful_criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train = torch.tensor(train_set.x, dtype=torch.float32, device=DEVICE)
    y_role_train = torch.tensor(train_set.y_role, dtype=torch.float32, device=DEVICE)
    y_harmful_train = torch.tensor(train_set.y_harmful, dtype=torch.float32, device=DEVICE)
    progress_train = torch.tensor(train_set.progress, dtype=torch.float32, device=DEVICE)
    event_count_train = torch.tensor(train_set.player_event_count, dtype=torch.float32, device=DEVICE)
    sample_weight_train = torch.tensor(train_set.sample_weight, dtype=torch.float32, device=DEVICE)

    x_val = torch.tensor(val_set.x, dtype=torch.float32, device=DEVICE)
    y_role_val = torch.tensor(val_set.y_role, dtype=torch.float32, device=DEVICE)
    y_harmful_val = torch.tensor(val_set.y_harmful, dtype=torch.float32, device=DEVICE)
    progress_val = torch.tensor(val_set.progress, dtype=torch.float32, device=DEVICE)
    event_count_val = torch.tensor(val_set.player_event_count, dtype=torch.float32, device=DEVICE)

    n = len(train_set.y_role)
    indices = np.arange(n)

    print("Training MinerTrustNet dual-head model...")
    print("device:", DEVICE)
    print("pos_weight:", float(pos_weight.item()))
    print("role_loss_coef:", ROLE_LOSS_COEF)
    print("harmful_loss_coef:", HARMFUL_LOSS_COEF)
    print("early prior regularization coefficient:", EARLY_PRIOR_COEF)
    print("checkpoint selection metric: val_acc")

    best_ckpt: Dict[str, Any] = {}
    best_epoch = 0
    best_score = -1.0
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        model.train()

        total_losses = []
        role_losses = []
        harmful_losses = []
        prior_losses = []

        for start in range(0, n, batch_size):
            mb_idx = indices[start:start + batch_size]

            xb = x_train[mb_idx]
            y_role_b = y_role_train[mb_idx]
            y_harm_b = y_harmful_train[mb_idx]
            event_count_b = event_count_train[mb_idx]
            sample_weight_b = sample_weight_train[mb_idx]
            weight_norm = sample_weight_b.sum().clamp_min(1e-6)

            role_logit, harmful_logit = model(xb)

            # Evidence-weighted soft label for hidden-role suspicion.
            evidence_weight = torch.clamp(event_count_b / EVIDENCE_FULL_COUNT, 0.0, 1.0)
            prior_target = torch.full_like(y_role_b, PRIOR_P_SABOTEUR)
            soft_role_target = (1.0 - evidence_weight) * prior_target + evidence_weight * y_role_b

            role_loss = (
                role_criterion(role_logit, soft_role_target) * sample_weight_b
            ).sum() / weight_norm
            harmful_loss = (
                harmful_criterion(harmful_logit, y_harm_b) * sample_weight_b
            ).sum() / weight_norm

            p_sab = torch.sigmoid(role_logit)
            low_evidence_weight = (1.0 - evidence_weight).clamp(0.0, 1.0)
            prior_loss = (
                low_evidence_weight
                * (p_sab - PRIOR_P_SABOTEUR).pow(2)
                * sample_weight_b
            ).sum() / weight_norm

            loss = (
                ROLE_LOSS_COEF * role_loss
                + HARMFUL_LOSS_COEF * harmful_loss
                + EARLY_PRIOR_COEF * prior_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_losses.append(float(loss.item()))
            role_losses.append(float(role_loss.item()))
            harmful_losses.append(float(harmful_loss.item()))
            prior_losses.append(float(prior_loss.item()))

        model.eval()
        with torch.no_grad():
            role_train_logit, harmful_train_logit = model(x_train)
            role_val_logit, harmful_val_logit = model(x_val)

            role_val_hard_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                role_val_logit, y_role_val
            ).item()

            harmful_val_bce = nn.BCEWithLogitsLoss()(
                harmful_val_logit, y_harmful_val
            ).item()

            train_metrics = metrics_from_logits(role_train_logit, y_role_train)
            val_metrics = metrics_from_logits(role_val_logit, y_role_val)
            val_phase_metrics = phase_metrics(model, x_val, y_role_val, progress_val, event_count_val)

            p_harmful_val = torch.sigmoid(harmful_val_logit)
            harmful_mae = torch.mean(torch.abs(p_harmful_val - y_harmful_val)).item()

        selection_score = float(val_metrics["accuracy"])
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_ckpt = build_trust_checkpoint(
                model=model,
                input_dim=dataset.x.shape[1],
                epoch=epoch,
                selection_metric="val_acc",
                selection_score=selection_score,
                val_metrics=best_val_metrics,
                val_role_bce=role_val_hard_bce,
                val_harm_bce=harmful_val_bce,
                val_harm_mae=harmful_mae,
            )

        phase_text = ""
        for phase_name in ["early", "mid", "late", "low_evidence", "enough_evidence"]:
            if phase_name in val_phase_metrics:
                m = val_phase_metrics[phase_name]
                phase_text += (
                    f" | {phase_name}:"
                    f"acc={m['accuracy']:.2f},"
                    f"sab_rec={m['saboteur_recall']:.2f},"
                    f"miner_acc={m['miner_accuracy']:.2f}"
                )

        print(
            f"[trust epoch {epoch:03d}] "
            f"loss={np.mean(total_losses):.5f} "
            f"role_loss={np.mean(role_losses):.5f} "
            f"harm_loss={np.mean(harmful_losses):.5f} "
            f"prior={np.mean(prior_losses):.5f} "
            f"val_role_bce={role_val_hard_bce:.5f} "
            f"val_harm_bce={harmful_val_bce:.5f} "
            f"val_harm_mae={harmful_mae:.5f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_sab_precision={val_metrics['saboteur_precision']:.3f} "
            f"val_sab_recall={val_metrics['saboteur_recall']:.3f} "
            f"val_miner_acc={val_metrics['miner_accuracy']:.3f} "
            f"tp={int(val_metrics['tp'])} "
            f"tn={int(val_metrics['tn'])} "
            f"fp={int(val_metrics['fp'])} "
            f"fn={int(val_metrics['fn'])}"
            f" best_epoch={best_epoch} "
            f"best_val_acc={best_score:.3f}"
            f"{' *best*' if is_best else ''}"
            f"{phase_text}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not best_ckpt:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_ckpt["model"])
    torch.save(best_ckpt, save_path)

    print(
        "saved best dual-head miner trust model:",
        save_path,
        f"(epoch={best_epoch}, val_acc={best_score:.3f}, "
        f"sab_precision={best_val_metrics.get('saboteur_precision', 0.0):.3f}, "
        f"sab_recall={best_val_metrics.get('saboteur_recall', 0.0):.3f}, "
        f"miner_acc={best_val_metrics.get('miner_accuracy', 0.0):.3f})"
    )

    model.eval()
    with torch.no_grad():
        sample_n = min(12, len(x_val))
        role_logit, harmful_logit = model(x_val[:sample_n])

        p_sab = torch.sigmoid(role_logit)
        trust = trust_score_from_role_logit(role_logit)
        p_harm = torch.sigmoid(harmful_logit)

    print("\n[dual-head trust examples]")
    for i in range(sample_n):
        label = int(y_role_val[i].item())
        print(
            f"sample={i} "
            f"progress={float(progress_val[i].item()):.3f} "
            f"player_events={float(event_count_val[i].item()):.1f} "
            f"role_label={'SABOTEUR' if label == 1 else 'GOLD_MINER'} "
            f"p_saboteur={float(p_sab[i].item()):.3f} "
            f"trust_score={float(trust[i].item()):.3f} "
            f"p_harmful={float(p_harm[i].item()):.3f} "
            f"harmful_label={float(y_harmful_val[i].item()):.3f}"
        )
