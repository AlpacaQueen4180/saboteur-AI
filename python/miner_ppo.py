import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

from miner_trust import (
    MinerTrustNet,
    PlayerFeatureAccumulator,
    update_accumulators_from_events,
    FEATURE_NAMES as TRUST_FEATURE_NAMES,
)


# ============================================================
# Config
# ============================================================

MAX_ACTIONS = 300

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
LR = 1e-4

PPO_EPOCHS = 4
MINIBATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRUST_MODEL_PATH = "checkpoints/miner_trust_model.pt"
USE_TRUST_FEATURES = True

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

SIDE_TYPES = [
    "PATH",
    "ROCK",
    "EMPTY",
]

TOOLS = [
    "CART",
    "LANTERN",
    "PICKAXE",
]

GOAL_TYPES = [
    "UNKNOWN",
    "ROCK",
    "GOLD",
]


# ============================================================
# Utilities
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def one_hot(index: int, size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


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


def bool_float(x: Any) -> float:
    return 1.0 if bool(x) else 0.0


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
            role = role_of(state)

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
# Learned Miner Trust / Opponent Assessment
# ============================================================

class MinerTrustInference:
    """
    Runtime wrapper for the learned dual-head miner trust model.

    Outputs per player:
        p_saboteur
        trust_score = 1 - 2 * p_saboteur
        p_harmful
        player_event_count_norm

    This is only used as state feature for Miner PPO.
    It is not directly used in the reward.
    """

    def __init__(self, model_path: str = TRUST_MODEL_PATH, device: torch.device = DEVICE):
        self.model_path = model_path
        self.device = device
        self.available = False

        self.model: Optional[MinerTrustNet] = None
        self.feature_names: List[str] = []
        self.input_dim: int = 0

        self.accumulators: Dict[int, PlayerFeatureAccumulator] = {}
        self.global_step: int = 0
        self.global_event_count: int = 0
        self.max_steps_per_game: int = 80

        self._load_model()

    def _load_model(self) -> None:
        if not USE_TRUST_FEATURES:
            print("[miner trust] disabled by USE_TRUST_FEATURES=False")
            return

        if not os.path.exists(self.model_path):
            print(f"[miner trust] checkpoint not found: {self.model_path}")
            print("[miner trust] PPO will use neutral trust features.")
            return

        ckpt = torch.load(self.model_path, map_location=self.device)

        self.input_dim = int(ckpt.get("input_dim", 0))
        self.feature_names = list(ckpt.get("feature_names", TRUST_FEATURE_NAMES))

        if self.input_dim <= 0:
            print("[miner trust] invalid checkpoint input_dim. Using neutral trust features.")
            return

        self.model = MinerTrustNet(input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.available = True
        print(f"[miner trust] loaded: {self.model_path}")
        print(f"[miner trust] input_dim={self.input_dim}, dual_head={ckpt.get('dual_head', False)}")

    def reset(self, state: Dict[str, Any], max_steps_per_game: int = 80) -> None:
        self.accumulators = {
            0: PlayerFeatureAccumulator(),
            1: PlayerFeatureAccumulator(),
            2: PlayerFeatureAccumulator(),
        }
        self.global_step = 0
        self.global_event_count = 0
        self.max_steps_per_game = max_steps_per_game

        self.update_from_state(state, increment_step=False)

    def update_from_state(self, state: Dict[str, Any], increment_step: bool = True) -> None:
        if not self.accumulators:
            self.reset(state, self.max_steps_per_game)
            return

        if increment_step:
            self.global_step += 1

        self.global_event_count += update_accumulators_from_events(self.accumulators, state)

    def infer_for_player(self, state: Dict[str, Any], player_index: int) -> Tuple[float, float, float, float]:
        if player_index == 3:
            return 0.0, 1.0, 0.0, 1.0

        if player_index not in self.accumulators:
            return 1.0 / 3.0, 1.0 - 2.0 * (1.0 / 3.0), 0.5, 0.0

        acc = self.accumulators[player_index]

        acc.update_context(
            state=state,
            player_index=player_index,
            global_step=self.global_step,
            max_steps_per_game=self.max_steps_per_game,
            global_event_count=self.global_event_count,
        )

        player_event_count = float(acc.data.get("player_event_count", 0.0))
        event_count_norm = clamp(player_event_count / 20.0, 0.0, 1.0)

        if not self.available or self.model is None:
            p_saboteur = 1.0 / 3.0
            trust_score = 1.0 - 2.0 * p_saboteur
            p_harmful = 0.5
            return p_saboteur, trust_score, p_harmful, event_count_norm

        x = acc.to_vector()

        if len(x) != self.input_dim:
            # Feature mismatch means trust checkpoint is incompatible with current miner_trust.py.
            p_saboteur = 1.0 / 3.0
            trust_score = 1.0 - 2.0 * p_saboteur
            p_harmful = 0.5
            return p_saboteur, trust_score, p_harmful, event_count_norm

        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            role_logit, harmful_logit = self.model(xt)
            p_saboteur = float(torch.sigmoid(role_logit)[0].item())
            p_harmful = float(torch.sigmoid(harmful_logit)[0].item())
            trust_score = 1.0 - 2.0 * p_saboteur

        return p_saboteur, trust_score, p_harmful, event_count_norm

    def encode_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        4 players × 4 features:
            p_saboteur
            trust_score
            p_harmful
            player_event_count_norm

        player 3 is self:
            p_saboteur=0
            trust_score=1
            p_harmful=0
            event_count_norm=1
        """
        vec: List[float] = []

        for player_index in range(4):
            p_saboteur, trust_score, p_harmful, event_count_norm = self.infer_for_player(
                state, player_index
            )
            vec.extend([
                p_saboteur,
                trust_score,
                p_harmful,
                event_count_norm,
            ])

        return np.asarray(vec, dtype=np.float32)

    def debug_summary(self, state: Dict[str, Any]) -> str:
        parts = []
        for player_index in [0, 1, 2]:
            p_sab, trust, p_harm, ev = self.infer_for_player(state, player_index)
            parts.append(
                f"p{player_index}:p_sab={p_sab:.2f},trust={trust:.2f},harm={p_harm:.2f},ev={ev:.2f}"
            )
        return " | ".join(parts)


# ============================================================
# State / Action Encoding
# ============================================================

def role_of(state: Dict[str, Any]) -> str:
    return get_nested(state, ["observation", "private", "role"], "GOLD_MINER")


def encode_role(state: Dict[str, Any]) -> np.ndarray:
    role = role_of(state)
    if role == "GOLD_MINER":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if role == "SABOTEUR":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return np.asarray([0.0, 0.0], dtype=np.float32)


def encode_card(card: Optional[Dict[str, Any]]) -> np.ndarray:
    vec: List[float] = []

    if card is None:
        card_type = ""
        sides = []
        effects = []
    else:
        card_type = card.get("type", "")
        sides = card.get("sides", [])
        effects = card.get("effects", [])

    vec.extend(one_hot(CARD_TYPES.index(card_type) if card_type in CARD_TYPES else -1, len(CARD_TYPES)))

    for i in range(4):
        side = sides[i] if i < len(sides) else "EMPTY"
        vec.extend(one_hot(SIDE_TYPES.index(side) if side in SIDE_TYPES else -1, len(SIDE_TYPES)))

    effects_set = set(effects or [])
    for tool in TOOLS:
        vec.append(1.0 if tool in effects_set else 0.0)

    return np.asarray(vec, dtype=np.float32)


def encode_hand(state: Dict[str, Any], max_hand: int = 6) -> np.ndarray:
    hand = get_nested(state, ["observation", "private", "hand"], [])
    card_dim = len(encode_card(None))
    out = np.zeros((max_hand, card_dim), dtype=np.float32)

    for i, card in enumerate(hand[:max_hand]):
        out[i] = encode_card(card)

    return out.flatten()


def encode_path_features(state: Dict[str, Any]) -> np.ndarray:
    pf = get_nested(state, ["observation", "board", "path_features"], {})

    known_goals = pf.get("known_goals", ["UNKNOWN", "UNKNOWN", "UNKNOWN"])
    known_goal_vec: List[float] = []

    for i in range(3):
        g = known_goals[i] if i < len(known_goals) else "UNKNOWN"
        known_goal_vec.extend(one_hot(GOAL_TYPES.index(g) if g in GOAL_TYPES else 0, len(GOAL_TYPES)))

    vals = [
        bool_float(pf.get("gold_known", False)),
        safe_float(pf.get("known_gold_index", -1), -1.0) / 2.0,

        safe_float(pf.get("distance_to_top_goal", -1), -1.0) / 12.0,
        safe_float(pf.get("distance_to_middle_goal", -1), -1.0) / 12.0,
        safe_float(pf.get("distance_to_bottom_goal", -1), -1.0) / 12.0,
        safe_float(pf.get("average_distance_to_all_goals", -1), -1.0) / 12.0,
        safe_float(pf.get("distance_to_known_gold", -1), -1.0) / 12.0,
        safe_float(pf.get("target_distance", -1), -1.0) / 12.0,

        safe_float(pf.get("frontier_count", 0), 0.0) / 20.0,
        safe_float(pf.get("reachable_count", 0), 0.0) / 45.0,
        safe_float(pf.get("destroyable_count", 0), 0.0) / 45.0,
    ]

    return np.asarray(vals + known_goal_vec, dtype=np.float32)


def encode_players(state: Dict[str, Any]) -> np.ndarray:
    players = get_nested(state, ["observation", "playerStatus"], [])
    out = np.zeros((4, 5), dtype=np.float32)

    for p in players[:4]:
        idx = safe_int(p.get("index"), -1)
        if not (0 <= idx < 4):
            continue

        blocked = set(p.get("blockedTools", []))
        out[idx, 0] = safe_float(p.get("handSize", 0), 0.0) / 6.0
        out[idx, 1] = bool_float(p.get("sabotaged", False))
        out[idx, 2] = 1.0 if "CART" in blocked else 0.0
        out[idx, 3] = 1.0 if "LANTERN" in blocked else 0.0
        out[idx, 4] = 1.0 if "PICKAXE" in blocked else 0.0

    return out.flatten()


def encode_events(state: Dict[str, Any]) -> np.ndarray:
    events = get_nested(state, ["observation", "events"], [])
    out = np.zeros((4, len(MOVE_TYPES)), dtype=np.float32)

    for e in events:
        p = safe_int(e.get("playerIndex"), -1)
        mt = e.get("type", "")
        if 0 <= p < 4 and mt in MOVE_TYPES:
            out[p, MOVE_TYPES.index(mt)] += 1.0

    denom = max(1.0, float(len(events)))
    out /= denom
    return out.flatten()


def encode_observation(
    state: Dict[str, Any],
    trust_inference: Optional[MinerTrustInference] = None,
) -> np.ndarray:
    parts = [
        encode_role(state),
        encode_hand(state),
        encode_path_features(state),
        encode_players(state),
        encode_events(state),
    ]

    if trust_inference is not None:
        parts.append(trust_inference.encode_features(state))
    else:
        # Neutral features, same dim as trust_inference output.
        neutral = []
        for i in range(4):
            if i == 3:
                neutral.extend([0.0, 1.0, 0.0, 1.0])
            else:
                p = 1.0 / 3.0
                neutral.extend([p, 1.0 - 2.0 * p, 0.5, 0.0])
        parts.append(np.asarray(neutral, dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def encode_action(action: Dict[str, Any]) -> np.ndarray:
    vec: List[float] = []

    move_type = action.get("move_type", action.get("type", ""))
    card_type = action.get("card_type", action.get("cardType", ""))

    vec.extend(one_hot(MOVE_TYPES.index(move_type) if move_type in MOVE_TYPES else -1, len(MOVE_TYPES)))
    vec.extend(one_hot(CARD_TYPES.index(card_type) if card_type in CARD_TYPES else -1, len(CARD_TYPES)))

    hand_index = safe_float(action.get("handIndex", action.get("hand_index", -1)), -1.0)
    vec.append(hand_index / 6.0)

    x = safe_float(action.get("x", -1), -1.0)
    y = safe_float(action.get("y", -1), -1.0)
    rotated = action.get("rotated", False)

    vec.append(x / 8.0)
    vec.append(y / 4.0)
    vec.append(bool_float(rotated))

    target_player = safe_int(action.get("target_player", action.get("targetPlayer", -1)), -1)
    vec.extend(one_hot(target_player, 4))

    goal_index = safe_int(action.get("goal_index", action.get("goalIndex", -1)), -1)
    vec.extend(one_hot(goal_index, 3))

    before = safe_float(action.get("before_target_distance", action.get("beforeTargetDistance", -1)), -1.0)
    after = safe_float(action.get("after_target_distance", action.get("afterTargetDistance", -1)), -1.0)
    delta = safe_float(action.get("delta_target_distance", action.get("deltaTargetDistance", 0)), 0.0)

    after_remove = safe_float(action.get("after_remove_distance", action.get("afterRemoveDistance", -1)), -1.0)
    after_ideal = safe_float(action.get("after_ideal_fill_distance", action.get("afterIdealFillDistance", -1)), -1.0)
    remove_delta = safe_float(action.get("remove_delta", action.get("removeDelta", 0)), 0.0)
    ideal_delta = safe_float(action.get("ideal_fill_delta", action.get("idealFillDelta", 0)), 0.0)

    after_reachable = safe_float(action.get("after_reachable_count", action.get("afterReachableCount", 0)), 0.0)
    after_destroyable = safe_float(action.get("after_destroyable_count", action.get("afterDestroyableCount", 0)), 0.0)

    vec.extend([
        before / 12.0,
        after / 12.0,
        delta / 12.0,
        after_remove / 12.0,
        after_ideal / 12.0,
        remove_delta / 12.0,
        ideal_delta / 12.0,
        after_reachable / 45.0,
        after_destroyable / 45.0,
    ])

    effects = set(action.get("effects", []))
    for tool in TOOLS:
        vec.append(1.0 if tool in effects else 0.0)

    vec.append(1.0 if delta > 0 else 0.0)
    vec.append(1.0 if delta < 0 else 0.0)
    vec.append(1.0 if remove_delta > 0 else 0.0)
    vec.append(1.0 if ideal_delta > 0 else 0.0)

    return np.asarray(vec, dtype=np.float32)


def encode_action_matrix(actions: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if len(actions) == 0:
        raise RuntimeError("No legal actions to encode.")

    action_dim = len(encode_action(actions[0]))
    mat = np.zeros((MAX_ACTIONS, action_dim), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

    n = min(len(actions), MAX_ACTIONS)
    for i in range(n):
        mat[i] = encode_action(actions[i])
        mask[i] = 1.0

    return mat, mask


def state_to_tensors(
    state: Dict[str, Any],
    trust_inference: Optional[MinerTrustInference] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = encode_observation(state, trust_inference)
    action_feats, mask = encode_action_matrix(state.get("legalActions", []))
    return obs, action_feats, mask


# ============================================================
# Miner Reward
# ============================================================

def gold_known(state: Dict[str, Any]) -> bool:
    return bool(get_nested(state, ["observation", "board", "path_features", "gold_known"], False))


def known_goals(state: Dict[str, Any]) -> List[str]:
    return get_nested(
        state,
        ["observation", "board", "path_features", "known_goals"],
        ["UNKNOWN", "UNKNOWN", "UNKNOWN"],
    )


def best_path_improvement(actions: List[Dict[str, Any]]) -> float:
    best = 0.0
    for a in actions:
        if a.get("type") == "PLAY_PATH":
            best = max(best, safe_float(a.get("delta_target_distance"), 0.0))
    return best


def compute_miner_reward(
    prev_state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    role = role_of(prev_state)
    assert role == "GOLD_MINER", f"Miner PPO received non-miner role: {role}"

    done = bool(next_state.get("done", False))
    winner = next_state.get("winner", None)

    parts: Dict[str, float] = {
        "step": -0.001,
        "terminal": 0.0,
        "path": 0.0,
        "player_action": 0.0,
        "rockfall": 0.0,
        "map": 0.0,
        "fold": 0.0,
    }

    reward = parts["step"]

    if done and winner is not None:
        parts["terminal"] = 1.0 if winner == "GOLD_MINER" else -1.0
        reward += parts["terminal"]

    move_type = action.get("type")
    card_type = action.get("card_type", action.get("cardType"))

    delta = safe_float(action.get("delta_target_distance"), 0.0)
    remove_delta = safe_float(action.get("remove_delta"), 0.0)
    ideal_delta = safe_float(action.get("ideal_fill_delta"), 0.0)

    if move_type == "PLAY_PATH":
        parts["path"] = 0.03 * delta
        reward += parts["path"]

    if move_type == "PLAY_PLAYER":
        # Trust is only a state feature in this version.
        # Do not reward directly by p_saboteur / p_harmful yet.
        if card_type == "REPAIR":
            parts["player_action"] = 0.015
        elif card_type == "BLOCK":
            parts["player_action"] = -0.015

        reward += parts["player_action"]

    if move_type == "PLAY_ROCKFALL":
        parts["rockfall"] = 0.03 * ideal_delta
        if remove_delta > 0 and ideal_delta <= 0:
            parts["rockfall"] -= 0.03

        reward += parts["rockfall"]

    if move_type == "PLAY_MAP":
        goal_index = safe_int(action.get("goal_index", action.get("goalIndex", -1)), -1)
        prev_known = known_goals(prev_state)
        next_known = known_goals(next_state)

        if gold_known(prev_state):
            parts["map"] = -0.01
        elif 0 <= goal_index < 3 and prev_known[goal_index] == "UNKNOWN":
            parts["map"] = 0.01
            if next_known[goal_index] == "GOLD":
                parts["map"] += 0.03
        else:
            parts["map"] = -0.005

        reward += parts["map"]

    if move_type == "DISCARD":
        legal_actions = prev_state.get("legalActions", [])
        imp = best_path_improvement(legal_actions)

        if imp > 0:
            parts["fold"] = -min(0.05, 0.03 * imp)
        else:
            parts["fold"] = -0.001

        reward += parts["fold"]

    reward = clamp(reward, -1.05, 1.05)
    parts["total"] = reward
    return reward, parts


# ============================================================
# Model
# ============================================================

class ActionConditionedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.Tanh(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, action_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_emb = self.obs_encoder(obs)
        value = self.value_head(obs_emb).squeeze(-1)

        act_emb = self.action_encoder(action_feats)
        obs_expand = obs_emb.unsqueeze(1).expand(-1, act_emb.shape[1], -1)
        joint = torch.cat([obs_expand, act_emb], dim=-1)

        scores = self.scorer(joint).squeeze(-1)
        return scores, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, value = self.forward(obs, action_feats)
        masked_scores = scores.masked_fill(mask <= 0, -1e9)

        dist = torch.distributions.Categorical(logits=masked_scores)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_feats: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, value = self.forward(obs, action_feats)
        masked_scores = scores.masked_fill(mask <= 0, -1e9)

        dist = torch.distributions.Categorical(logits=masked_scores)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()

        return logprob, entropy, value


# ============================================================
# PPO Buffer
# ============================================================

@dataclass
class RolloutBuffer:
    obs: List[np.ndarray]
    action_feats: List[np.ndarray]
    masks: List[np.ndarray]
    actions: List[int]
    logprobs: List[float]
    rewards: List[float]
    dones: List[bool]
    values: List[float]

    def clear(self) -> None:
        self.obs.clear()
        self.action_feats.clear()
        self.masks.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


def compute_gae(buffer: RolloutBuffer, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    dones = np.asarray(buffer.dones, dtype=np.float32)
    values = np.asarray(buffer.values + [next_value], dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t + 1] * nonterminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def infer_checkpoint_update(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    marker = "update_"
    idx = stem.rfind(marker)
    if idx < 0:
        return 0

    raw = stem[idx + len(marker):]
    digits = []
    for ch in raw:
        if not ch.isdigit():
            break
        digits.append(ch)

    if not digits:
        return 0
    return int("".join(digits))


# ============================================================
# Debug
# ============================================================

def print_debug(
    state: Dict[str, Any],
    action: Dict[str, Any],
    reward_parts: Dict[str, float],
    trust_inference: Optional[MinerTrustInference],
) -> None:
    print("\n[MINER DEBUG ACTION]")
    print(
        f"type={action.get('type')} card={action.get('card_name')} "
        f"hand={action.get('handIndex')} x={action.get('x')} y={action.get('y')} "
        f"rotated={action.get('rotated')} targetPlayer={action.get('target_player')} "
        f"goalIndex={action.get('goal_index')}"
    )
    print(
        f"targetDist={action.get('before_target_distance')} -> "
        f"{action.get('after_target_distance')} "
        f"delta={action.get('delta_target_distance')}"
    )
    print(
        f"removeDelta={action.get('remove_delta')} "
        f"idealFillDelta={action.get('ideal_fill_delta')}"
    )

    if trust_inference is not None:
        print("opponent assessment:", trust_inference.debug_summary(state))

    print("reward_parts:", {k: round(v, 4) for k, v in reward_parts.items()})


# ============================================================
# Train Miner PPO
# ============================================================

def train_miner(
    base_url: str,
    initial_state: Dict[str, Any],
    total_updates: int,
    rollout_steps: int,
    save_every: int,
    debug_every: int,
    resume_path: str = "",
) -> None:
    env = SaboteurHttpEnv(base_url)

    trust_inference = MinerTrustInference(TRUST_MODEL_PATH, DEVICE)

    state = initial_state
    if role_of(state) != "GOLD_MINER":
        state = env.reset_until_role("GOLD_MINER")

    trust_inference.reset(state, max_steps_per_game=80)
    obs, action_feats, mask = state_to_tensors(state, trust_inference)

    obs_dim = obs.shape[0]
    action_dim = action_feats.shape[1]

    print("Miner PPO training started.")
    print("device:", DEVICE)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("initial legal_actions:", int(mask.sum()))
    print("trust_features_enabled:", trust_inference.available)
    print("trust_model_path:", TRUST_MODEL_PATH)

    model = ActionConditionedActorCritic(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer([], [], [], [], [], [], [], [])

    episode_count = 0
    win_count = 0
    start_update = 0

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Miner resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=DEVICE)
        ckpt_obs_dim = int(ckpt.get("obs_dim", -1))
        ckpt_action_dim = int(ckpt.get("action_dim", -1))
        ckpt_role = ckpt.get("role", "")

        if ckpt_role != "GOLD_MINER":
            raise ValueError(f"Miner resume checkpoint has role={ckpt_role!r}, expected 'GOLD_MINER'.")
        if ckpt_obs_dim != obs_dim or ckpt_action_dim != action_dim:
            raise ValueError(
                "Miner resume checkpoint dimension mismatch: "
                f"checkpoint obs/action=({ckpt_obs_dim}, {ckpt_action_dim}), "
                f"current obs/action=({obs_dim}, {action_dim})"
            )

        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[miner] resumed optimizer state:", resume_path)
        else:
            print("[miner] resume checkpoint has no optimizer state; using fresh Adam.")

        start_update = int(ckpt.get("update", infer_checkpoint_update(resume_path)))
        episode_count = int(ckpt.get("episode_count", 0))
        win_count = int(ckpt.get("win_count", 0))
        print(
            f"[miner] resumed model: {resume_path} "
            f"start_update={start_update} episodes={episode_count} wins={win_count}"
        )

    for local_update in range(1, total_updates + 1):
        update = start_update + local_update
        buffer.clear()
        rollout_reward = 0.0
        steps_collected = 0

        while steps_collected < rollout_steps:
            if role_of(state) != "GOLD_MINER":
                state = env.reset_until_role("GOLD_MINER")
                trust_inference.reset(state, max_steps_per_game=80)
                obs, action_feats, mask = state_to_tensors(state, trust_inference)

            legal_actions = state.get("legalActions", [])
            if len(legal_actions) == 0:
                state = env.reset_until_role("GOLD_MINER")
                trust_inference.reset(state, max_steps_per_game=80)
                obs, action_feats, mask = state_to_tensors(state, trust_inference)
                continue

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action(obs_t, act_t, mask_t)

            action_id = int(action_t.item())
            if action_id >= len(legal_actions):
                action_id = 0

            selected_action = legal_actions[action_id]
            prev_state = state

            try:
                next_state = env.step(selected_action)
            except requests.HTTPError as e:
                print("[miner] HTTP step error:", e)
                state = env.reset_until_role("GOLD_MINER")
                trust_inference.reset(state, max_steps_per_game=80)
                obs, action_feats, mask = state_to_tensors(state, trust_inference)
                continue

            reward, reward_parts = compute_miner_reward(prev_state, selected_action, next_state)
            done = bool(next_state.get("done", False))

            buffer.obs.append(obs)
            buffer.action_feats.append(action_feats)
            buffer.masks.append(mask)
            buffer.actions.append(action_id)
            buffer.logprobs.append(float(logprob_t.item()))
            buffer.rewards.append(float(reward))
            buffer.dones.append(done)
            buffer.values.append(float(value_t.item()))

            rollout_reward += reward
            steps_collected += 1

            if update % debug_every == 0 and steps_collected == 1:
                print_debug(prev_state, selected_action, reward_parts, trust_inference)

            if done:
                episode_count += 1
                if next_state.get("winner") == "GOLD_MINER":
                    win_count += 1

                state = env.reset_until_role("GOLD_MINER")
                trust_inference.reset(state, max_steps_per_game=80)
                obs, action_feats, mask = state_to_tensors(state, trust_inference)
            else:
                state = next_state
                trust_inference.update_from_state(state, increment_step=True)
                obs, action_feats, mask = state_to_tensors(state, trust_inference)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act_t = torch.tensor(action_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, next_value_t = model.forward(obs_t, act_t)
            next_value = float(next_value_t.item())

        advantages, returns = compute_gae(buffer, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=DEVICE)
        act_batch = torch.tensor(np.asarray(buffer.action_feats), dtype=torch.float32, device=DEVICE)
        mask_batch = torch.tensor(np.asarray(buffer.masks), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)
        old_logprob_batch = torch.tensor(buffer.logprobs, dtype=torch.float32, device=DEVICE)
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret_batch = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        n = len(buffer.actions)
        indices = np.arange(n)

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx = indices[start:start + MINIBATCH_SIZE]

                new_logprob, entropy, value = model.evaluate_actions(
                    obs_batch[mb_idx],
                    act_batch[mb_idx],
                    mask_batch[mb_idx],
                    action_batch[mb_idx],
                )

                ratio = torch.exp(new_logprob - old_logprob_batch[mb_idx])
                surr1 = ratio * adv_batch[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_batch[mb_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - ret_batch[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))

        win_rate = win_count / max(1, episode_count)
        avg_step_reward = rollout_reward / max(1, steps_collected)

        print(
            f"[miner update {update:04d}] "
            f"episodes={episode_count} "
            f"rollout_reward={rollout_reward:.3f} "
            f"avg_step_reward={avg_step_reward:.5f} "
            f"policy_loss={np.mean(policy_losses):.5f} "
            f"value_loss={np.mean(value_losses):.5f} "
            f"entropy={np.mean(entropies):.5f} "
            f"miner_win={win_rate:.3f}"
        )

        if update % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/miner_ppo_http_trust_update_{update}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "source": "miner_ppo_trust_pure_ppo",
                    "update": update,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "max_actions": MAX_ACTIONS,
                    "role": "GOLD_MINER",
                    "move_types": MOVE_TYPES,
                    "card_types": CARD_TYPES,
                    "episode_count": episode_count,
                    "win_count": win_count,
                    "win_rate": win_rate,
                    "rollout_steps": rollout_steps,
                    "total_updates_requested": total_updates,
                    "resume_path": resume_path,
                    "trust_features_enabled": trust_inference.available,
                    "trust_model_path": TRUST_MODEL_PATH,
                    "trust_features": [
                        "p_saboteur",
                        "trust_score",
                        "p_harmful",
                        "player_event_count_norm",
                    ],
                    "ppo_hyperparameters": {
                        "gamma": GAMMA,
                        "gae_lambda": GAE_LAMBDA,
                        "clip_eps": CLIP_EPS,
                        "entropy_coef": ENTROPY_COEF,
                        "value_coef": VALUE_COEF,
                        "lr": LR,
                        "ppo_epochs": PPO_EPOCHS,
                        "minibatch_size": MINIBATCH_SIZE,
                    },
                },
                ckpt_path,
            )
            print("saved checkpoint:", ckpt_path)
