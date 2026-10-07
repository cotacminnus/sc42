"""Sample OpenSpiel rollouts using the engine-backed Texas 42 adapter."""
import argparse
import random

import pyspiel
import texas42.openspiel  # Register the optional adapter.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", choices=("hand", "match"), default="hand")
    parser.add_argument("--target-marks", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    game = pyspiel.load_game("python_texas42", {
        "episode": args.episode, "target_marks": args.target_marks,
    })
    rng = random.Random(args.seed)
    state = game.new_initial_state()
    decisions = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probabilities = zip(*state.chance_outcomes())
            action = rng.choices(outcomes, weights=probabilities, k=1)[0]
        else:
            # Policies can consume this tensor plus state.legal_actions_mask().
            observation = state.information_state_tensor(state.current_player())
            assert len(observation) == game.information_state_tensor_size()
            action = rng.choice(state.legal_actions())
            decisions += 1
        state.apply_action(action)
    print(f"Player decisions: {decisions}; returns: {state.returns()}")


if __name__ == "__main__":
    main()
