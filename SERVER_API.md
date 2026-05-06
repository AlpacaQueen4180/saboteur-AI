# Saboteur Server API

The Java server runs one headless Saboteur environment for Python/RL control.
It always creates a 4-player game where player `3` is controlled by Python and
players `0`, `1`, and `2` are `HeuristicsAI` opponents.

## Run

```powershell
mvn package
mvn exec:java
```

The default URL is `http://localhost:8000`. To use another port:

```powershell
mvn exec:java -Dexec.args="9000"
```

## Endpoints

### `GET /health`

Returns server status.

```json
{
  "ok": true,
  "started": false,
  "finished": false
}
```

### `POST /reset`

Starts a new 4-player game. The request body is optional and ignored.

```json
{}
```

Response:

```json
{
  "observation": {},
  "fullState": {},
  "legalActions": [],
  "reward": 0,
  "done": false,
  "winner": null
}
```

### `GET /state?view=observation|full|both`

Returns the current state. The default view is `both`.

- `observation`: controlled player training view.
- `full`: debug state with all player roles and hands.
- `both`: includes both fields.

### `GET /legal-actions`

Returns legal actions for player `3` if it is player `3`'s turn. Returns an
empty list if the game is finished or the heuristic players are still resolving.

### `POST /step`

Applies one player `3` action. The server infers `playerIndex: 3`, then advances
heuristic players until the next player `3` turn or terminal state.

Move examples:

```json
{ "type": "DISCARD", "handIndex": 0 }
```

```json
{ "type": "PLAY_PATH", "handIndex": 1, "x": 2, "y": 2, "rotated": false }
```

```json
{ "type": "PLAY_PLAYER", "handIndex": 2, "targetPlayer": 1 }
```

```json
{ "type": "PLAY_MAP", "handIndex": 0, "goal": "TOP" }
```

```json
{ "type": "PLAY_ROCKFALL", "handIndex": 4, "x": 3, "y": 2 }
```

Response:

```json
{
  "observation": {},
  "fullState": {},
  "legalActions": [],
  "reward": 0,
  "done": false,
  "winner": null
}
```

## Observation

`observation` contains only player `3`'s private information plus public game
state.

- `board`: width, height, start/goal positions, and all cells with side layout
  and placed card data.
- `revealedGoals`: goals known to player `3`.
- `connectivity`: reachable cells, destroyable cells, and placeable cells per
  path card in player `3`'s hand.
- `private`: player `3` index, role, and hand cards.
- `public`: current player, deck size, game status, and public player status.
- `playerStatus`: each player's hand size and blocked tools.
- `events`: latest transition moves, including player `3`'s applied move and
  all heuristic moves before control returns to player `3`.

Python should store cumulative history itself by appending `observation.events`
after each reset or step.

## Errors

Errors return JSON:

```json
{ "error": "message" }
```

Status codes:

- `400`: invalid JSON, invalid request body, or unknown action.
- `405`: wrong HTTP method.
- `409`: game not reset, game finished, or not player `3`'s turn.

## Python Smoke Test

```python
import requests

base = "http://localhost:8000"
print(requests.get(f"{base}/health").json())

state = requests.post(f"{base}/reset", json={}).json()
assert state["observation"]["private"]["playerIndex"] == 3
assert len(state["fullState"]["players"]) == 4

actions = requests.get(f"{base}/legal-actions").json()
step = requests.post(f"{base}/step", json=actions[0]).json()
print(step["reward"], step["done"], step["observation"]["events"])
```
