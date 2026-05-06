# Saboteur

Final project for Game Theory/AI course. This repo is dedicated completely for the development of the AI. The base game can be found [Here](https://github.com/nickylogan/saboteur). The AI will approach the game with an unsupervised reinforcement learning approach, competing against itself using a shared curriculum.

## Model
* ### Inputs
```will be added```
* ### Outputs
```will be added```
* ### Reward System
```will be added```

## Controls

* To select a card, left-click on any of the card on the bottom pane
* To place a card on the board, right-click on the desired position
* To target a player (repair/block), click on the player name on the right pane
* To rotate a path card, press `R`
* To discard the selected card, press `D`

## Documentation

Please read the javadoc.

## Headless Python/RL Server

The project includes a headless HTTP server for Python/RL control. It runs a
fixed 4-player game where Python controls player `3` and the other players use
`HeuristicsAI`.

To run the server, install Maven first. Then run the server with `mvn exec:java`.
Test the server with `curl http://localhost:8000/health`.

See [SERVER_API.md](SERVER_API.md) for endpoints and JSON schemas.

From this `saboteur-AI` folder, install the Python client dependency and run
the smoke test:

```sh
pip install -r requirements.txt
python test.py
```
