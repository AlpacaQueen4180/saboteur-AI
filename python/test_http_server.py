import requests
from typing import Any, Dict, List, Optional

BASE = "http://localhost:8000"


def get_path_features(state: Dict[str, Any]) -> Dict[str, Any]:
    return state["observation"]["board"].get("path_features", {})


def get_private(state: Dict[str, Any]) -> Dict[str, Any]:
    return state["observation"].get("private", {})


def get_player_status(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return state["observation"].get("playerStatus", [])


def get_events(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return state["observation"].get("events", [])


def print_hand(state: Dict[str, Any]) -> None:
    private = get_private(state)
    hand = private.get("hand", [])

    print("\n[HAND]")
    print(f"playerIndex={private.get('playerIndex')} role={private.get('role')}")
    for i, card in enumerate(hand):
        name = card.get("name")
        ctype = card.get("type")
        path_type = card.get("pathType")
        effects = card.get("effects")

        extra = ""
        if path_type:
            extra = f", pathType={path_type}"
        if effects:
            extra = f", effects={effects}"

        print(f"  hand[{i}] {name} ({ctype}{extra})")


def print_path_summary(state: Dict[str, Any]) -> None:
    board = state["observation"]["board"]
    pf = board.get("path_features", {})

    gold_known = pf.get("gold_known")
    known_goals = pf.get("known_goals")
    known_gold_index = pf.get("known_gold_index")

    d_top = pf.get("distance_to_top_goal")
    d_mid = pf.get("distance_to_middle_goal")
    d_bot = pf.get("distance_to_bottom_goal")
    avg = pf.get("average_distance_to_all_goals")
    d_gold = pf.get("distance_to_known_gold")
    target = pf.get("target_distance")

    reachable = board.get("reachable_count")
    destroyable = board.get("destroyable_count")
    frontier = pf.get("frontier_count")

    print("\n[PATH / BOARD SUMMARY]")
    print(f"board={board.get('width')}x{board.get('height')}")
    print(f"reachable_count={reachable}, destroyable_count={destroyable}, frontier_count={frontier}")
    print(f"gold_known={gold_known}, known_gold_index={known_gold_index}, known_goals={known_goals}")
    print(f"d_top={d_top}, d_middle={d_mid}, d_bottom={d_bot}")
    print(f"avg_distance_to_all_goals={avg}")
    print(f"distance_to_known_gold={d_gold}")
    print(f"target_distance={target}")

    if gold_known:
        print("target rule: gold is known → target_distance = distance_to_known_gold")
    else:
        print("target rule: gold unknown → target_distance = average distance to TOP/MIDDLE/BOTTOM")


def print_players(state: Dict[str, Any]) -> None:
    players = get_player_status(state)

    print("\n[PLAYER TOOL STATUS]")
    for p in players:
        idx = p.get("index")
        name = p.get("name")
        hand_size = p.get("handSize")
        sabotaged = p.get("sabotaged")
        blocked_tools = p.get("blockedTools", [])
        print(
            f"  player {idx} {name}: "
            f"handSize={hand_size}, sabotaged={sabotaged}, blockedTools={blocked_tools}"
        )


def print_events(state: Dict[str, Any]) -> None:
    events = get_events(state)

    print("\n[RECENT EVENTS]")
    if not events:
        print("  no events")
        return

    for i, e in enumerate(events):
        card = e.get("card")
        card_name = card.get("name") if isinstance(card, dict) else None
        print(
            f"  event[{i}] "
            f"player={e.get('playerIndex')} "
            f"type={e.get('type')} "
            f"handIndex={e.get('handIndex')} "
            f"args={e.get('args')} "
            f"card={card_name}"
        )


def summarize_action(action: Dict[str, Any]) -> str:
    action_id = action.get("action_id")
    move_type = action.get("move_type", action.get("type"))
    hand_index = action.get("handIndex")
    card_name = action.get("card_name", action.get("cardName"))
    card_type = action.get("card_type", action.get("cardType"))

    x = action.get("x")
    y = action.get("y")
    rotated = action.get("rotated")
    target_player = action.get("target_player", action.get("targetPlayer"))
    goal_index = action.get("goal_index", action.get("goalIndex"))

    before = action.get("before_target_distance")
    after = action.get("after_target_distance")
    delta = action.get("delta_target_distance")
    remove_delta = action.get("remove_delta")
    ideal_fill_delta = action.get("ideal_fill_delta")

    return (
        f"id={action_id} type={move_type} hand={hand_index} "
        f"card={card_name}({card_type}) "
        f"x={x} y={y} rotated={rotated} "
        f"targetPlayer={target_player} goalIndex={goal_index} "
        f"targetDist={before}->{after} delta={delta} "
        f"removeDelta={remove_delta} idealFillDelta={ideal_fill_delta}"
    )


def print_legal_actions(state: Dict[str, Any], max_actions: int = 20) -> None:
    actions = state.get("legalActions", [])

    print("\n[LEGAL ACTIONS SUMMARY]")
    print(f"legal_actions={len(actions)}")

    if not actions:
        return

    print(f"showing first {min(max_actions, len(actions))} actions:")
    for a in actions[:max_actions]:
        print("  " + summarize_action(a))

    best_path = None
    best_path_delta = -999.0
    for a in actions:
        if a.get("type") == "PLAY_PATH":
            delta = float(a.get("delta_target_distance", 0.0))
            if delta > best_path_delta:
                best_path_delta = delta
                best_path = a

    if best_path is not None:
        print("\n[BEST PATH ACTION BY delta_target_distance]")
        print("  " + summarize_action(best_path))

    best_rockfall_miner = None
    best_ideal = -999.0
    best_rockfall_sab = None
    best_remove = -999.0

    for a in actions:
        if a.get("type") == "PLAY_ROCKFALL":
            ideal = float(a.get("ideal_fill_delta", 0.0))
            remove = float(a.get("remove_delta", 0.0))

            if ideal > best_ideal:
                best_ideal = ideal
                best_rockfall_miner = a

            if remove > best_remove:
                best_remove = remove
                best_rockfall_sab = a

    if best_rockfall_miner is not None:
        print("\n[BEST ROCKFALL FOR MINER BY ideal_fill_delta]")
        print("  " + summarize_action(best_rockfall_miner))

    if best_rockfall_sab is not None:
        print("\n[BEST ROCKFALL FOR SABOTEUR BY remove_delta]")
        print("  " + summarize_action(best_rockfall_sab))


def print_state_summary(title: str, state: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"reward={state.get('reward')} done={state.get('done')} winner={state.get('winner')}")

    print_hand(state)
    print_path_summary(state)
    print_players(state)
    print_events(state)
    print_legal_actions(state, max_actions=20)


def choose_interesting_action(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    For smoke test only:
    Prefer a path action that improves target distance.
    If none exists, choose first legal action.
    """
    actions = state.get("legalActions", [])
    if not actions:
        return None

    best = None
    best_delta = 0.0

    for a in actions:
        if a.get("type") == "PLAY_PATH":
            delta = float(a.get("delta_target_distance", 0.0))
            if delta > best_delta:
                best_delta = delta
                best = a

    if best is not None:
        return best

    return actions[0]


def main() -> None:
    print("GET /health")
    health = requests.get(BASE + "/health", timeout=5)
    print("status:", health.status_code, health.json())

    print("\nPOST /reset")
    reset = requests.post(BASE + "/reset", json={}, timeout=10)
    print("status:", reset.status_code)

    state = reset.json()
    print_state_summary("RESET SUMMARY", state)

    action = choose_interesting_action(state)
    if action is None:
        print("\nNo legal action available.")
        return

    print("\n" + "=" * 80)
    print("CHOSEN ACTION FOR SMOKE TEST")
    print("=" * 80)
    print(summarize_action(action))

    print("\nPOST /step")
    step = requests.post(BASE + "/step", json=action, timeout=10)
    print("status:", step.status_code)

    next_state = step.json()
    print_state_summary("AFTER STEP SUMMARY", next_state)


if __name__ == "__main__":
    main()