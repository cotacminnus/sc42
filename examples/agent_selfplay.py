"""Run with: python -m examples.agent_selfplay (no optional dependencies)."""
from texas42 import SelfPlayEnv
from texas42.agents import RandomAgent


def main():
    agents = [RandomAgent(seed=seat) for seat in range(4)]
    env = SelfPlayEnv(seed=42)
    obs = env.reset()
    for seat, agent in enumerate(agents):
        agent.reset(seat)
    decisions = 0
    while obs is not None:
        transition = env.step(agents[obs.seat].act(obs))
        decisions += 1
        obs = transition.observation
    print(f'{decisions} decisions; hand rewards: {transition.rewards}')


if __name__ == '__main__':
    main()
