from __future__ import annotations

import argparse
import json
import random
from contextlib import ExitStack
from pathlib import Path
from collections.abc import Sequence

from .engine import Game, GameConfig
from .players import HumanPlayer, Player, RandomPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Texas 42 against three placeholder bots.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="reveal every player's dominoes at the beginning of each hand",
    )
    parser.add_argument("--seed", type=int, help="use a reproducible shuffle and bot seed")
    parser.add_argument(
        "--target-marks",
        type=int,
        default=7,
        help="marks needed to win (default: 7)",
    )
    parser.add_argument(
        "--all-ai",
        action="store_true",
        help="run self-play with four placeholder bots",
    )
    parser.add_argument(
        "--self-play-log", nargs="?", const=Path("logs/selfplay.jsonl"), type=Path,
        metavar="PATH", help="write self-play states as JSONL (default: logs/selfplay.jsonl); requires --all-ai",
    )
    return parser


def make_players(seed: int | None, all_ai: bool = False) -> list[Player]:
    # Separate bot RNGs keep one player's choices from consuming another's stream.
    master = random.Random(seed)
    players: list[Player] = []
    for seat in range(4):
        if seat == 0 and not all_ai:
            players.append(HumanPlayer())
        else:
            players.append(RandomPlayer(f"Dummy AI {seat + 1}", random.Random(master.getrandbits(64))))
    return players


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_play_log is not None and not args.all_ai:
        parser.error("--self-play-log requires --all-ai")
    try:
        config = GameConfig(
            target_marks=args.target_marks,
            debug=args.debug,
            seed=args.seed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    with ExitStack() as stack:
        log_state = None
        if args.self_play_log is not None:
            args.self_play_log.parent.mkdir(parents=True, exist_ok=True)
            log_file = stack.enter_context(args.self_play_log.open("w", encoding="utf-8"))

            def log_state(record: dict) -> None:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

        game = Game(make_players(args.seed, args.all_ai), config, log_state=log_state)
        try:
            game.play_match()
        except (EOFError, KeyboardInterrupt):
            print("\nGame ended.")
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
