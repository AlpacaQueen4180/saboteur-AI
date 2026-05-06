import random
from pprint import pprint

from SaboteurEnv import SaboteurEnv, default_server_command


def choose_action(actions):
    """Simple baseline: play the first legal action returned by the server."""
    if not actions:
        raise RuntimeError("No legal actions available before the game ended")
    return random.choice(actions)


def main() -> None:
    env = SaboteurEnv(server_command=default_server_command())
    env.make(start_server=True)
    obs = env.reset()
    assert obs["private"]["playerIndex"] == 3

    done = False
    total_reward = 0
    steps = 0
    winner = None

    while not done:
        actions = env.legal_actions(refresh=False)
        action = choose_action(actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        winner = info["winner"]

        print(
            f"step={steps} action={action['type']} "
            f"events={len(info['events'])} reward={reward} done={done}"
        )

    print("\nGame finished")
    print("winner:", winner)
    print("total_reward:", total_reward)
    print("agent_steps:", steps)
    print("recorded_events:", len(env.history))
    print("\nFinal board:")
    print(env.render_text(obs))
    print("\nLast observation private info:")
    pprint(obs["private"])


if __name__ == "__main__":
    main()
