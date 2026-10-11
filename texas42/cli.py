from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from .agents import AgentPlayer, VARIANTS, variant_factory
from .engine import Game, GameConfig
from .players import HumanPlayer, Player, RandomPlayer
from .rulesets import RULESET_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Texas 42 with humans and selectable policy bots.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="reveal every player's dominoes at the beginning of each hand",
    )
    parser.add_argument("--seed", type=int, help="use a reproducible shuffle and bot seed")
    parser.add_argument("--ruleset-id", default=RULESET_ID, choices=(RULESET_ID,),
                        help="versioned house rules (currently only cadets_base_v1)")
    parser.add_argument(
        "--target-marks",
        type=int,
        default=7,
        help="marks needed to win (default: 7)",
    )
    parser.add_argument(
        "--all-ai",
        action="store_true",
        help="run self-play without humans (four placeholder bots unless --players is supplied)",
    )
    parser.add_argument(
        "--players", nargs=4, choices=("human", "placeholder", "neural", *VARIANTS),
        metavar=("P1", "P2", "P3", "P4"),
        help="player types in seat order: human, placeholder, neural, or "
             + ", ".join(VARIANTS) + "; seats 1/3 and 2/4 are partners; "
             "neural defaults to neural bid/play and heuristic trump",
    )
    for number in range(1, 5):
        group = parser.add_argument_group(f"Player {number} phase composition")
        group.add_argument(f"--player-{number}-checkpoint", type=Path,
                           help="default checkpoint for this player's neural phases")
        for phase in ("bid", "trump", "play"):
            choices = ("neural", "heuristic", "random", "conservative") if phase == "bid" else (
                "neural", "heuristic", "random")
            group.add_argument(f"--player-{number}-{phase}", choices=choices,
                               help="override this phase of the selected bot policy")
            group.add_argument(f"--player-{number}-{phase}-checkpoint", type=Path,
                               help="checkpoint override for this neural phase only")
    # nargs="?" + const: a bare flag uses the default path, a value overrides it
    # (https://docs.python.org/3/library/argparse.html#nargs).
    parser.add_argument(
        "--self-play-log", nargs="?", const=Path("logs/selfplay.jsonl"), type=Path,
        metavar="PATH", help="write self-play states as JSONL (default: logs/selfplay.jsonl); requires --all-ai",
    )
    return parser


def make_players(seed: int | None, all_ai: bool = False,
                 player_types: Sequence[str] | None = None,
                 policy_options: Sequence[dict] | None = None) -> list[Player]:
    if player_types is None:
        player_types = ("placeholder",) * 4 if all_ai else ("human",) + ("placeholder",) * 3
    if len(player_types) != 4 or any(kind not in ("human", "placeholder", "neural", *VARIANTS)
                                    for kind in player_types):
        raise ValueError("Choose four player types: human, placeholder, neural, or a policy variant")
    if all_ai and "human" in player_types:
        raise ValueError("--all-ai cannot include human players")
    # Separate bot RNGs keep one player's choices from consuming another's stream.
    master = random.Random(seed)
    players: list[Player] = []
    options = policy_options if policy_options is not None else ({},) * 4
    if len(options) != 4:
        raise ValueError("Specify policy options for four seats")
    for seat, kind in enumerate(player_types):
        overrides = options[seat]
        if overrides and kind in ("human", "placeholder"):
            raise ValueError(f"Player {seat + 1}: phase options require neural or a policy variant in --players")
        if kind == "human":
            players.append(HumanPlayer() if seat == 0 else HumanPlayer(f"Human {seat + 1}"))
        else:
            bot_seed = master.getrandbits(64)
            if kind == "placeholder":
                players.append(RandomPlayer(f"Dummy AI {seat + 1}", random.Random(bot_seed)))
            elif kind == "neural" or overrides:
                # Optional research imports stay at the composition boundary.
                from research.ppo.composition import CompositeAgentFactory, resolve_policy_spec
                defaults = ("neural", "heuristic", "neural") if kind == "neural" else VARIANTS[kind]
                strategies = tuple(overrides.get(phase) or default for phase, default in
                                   zip(("bid", "trump", "play"), defaults))
                spec = resolve_policy_spec(f"Player {seat + 1}", strategies,
                    overrides.get("checkpoint"),
                    tuple(overrides.get(phase + "_checkpoint") for phase in ("bid", "trump", "play")))
                factory = CompositeAgentFactory(spec)
                factory.prepare()
                players.append(AgentPlayer(f"Policy AI {seat + 1}", factory(seat, bot_seed)))
            else:
                players.append(AgentPlayer(f"{kind} AI {seat + 1}",
                                           variant_factory(kind)(seat, bot_seed)))
    return players


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_play_log is not None and not args.all_ai:
        parser.error("--self-play-log requires --all-ai")
    if args.all_ai and args.players is not None and "human" in args.players:
        parser.error("--all-ai cannot include human players")
    try:
        config = GameConfig(
            target_marks=args.target_marks,
            debug=args.debug,
            seed=args.seed,
            ruleset_id=args.ruleset_id,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    options = []
    for number in range(1, 5):
        prefix = f"player_{number}_"
        options.append({key: value for key in ("checkpoint", "bid", "trump", "play",
                       "bid_checkpoint", "trump_checkpoint", "play_checkpoint")
                       if (value := getattr(args, prefix + key)) is not None})
    try:
        players = make_players(args.seed, args.all_ai, args.players, options)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    # ExitStack closes the log file only if one was opened
    # (https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack).
    with ExitStack() as stack:
        neural = False
        for player in players:
            spec = getattr(getattr(player, "agent", None), "spec", None)
            if spec is not None:
                print(f"{spec.name}: {spec.label()}")
                for phase, policy in spec.items():
                    if policy.checkpoint is not None:
                        neural = True
                        print(f"  {phase}: {policy.checkpoint.path} [SHA256 {policy.checkpoint.sha256}]")
        if neural:
            import torch
            stack.callback(torch.set_num_threads, torch.get_num_threads())
            torch.set_num_threads(1)
        log_state = None
        if args.self_play_log is not None:
            args.self_play_log.parent.mkdir(parents=True, exist_ok=True)
            log_file = stack.enter_context(args.self_play_log.open("w", encoding="utf-8"))

            def log_state(record: dict) -> None:
                log_file.write(json.dumps(record) + "\n")
                # Preserve completed transitions if a long match is interrupted.
                log_file.flush()

        game = Game(players, config, log_state=log_state)
        try:
            game.play_match()
        except (EOFError, KeyboardInterrupt):
            print("\nGame ended.")
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
