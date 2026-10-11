import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from texas42 import (
    ACTION_REGISTRY, RULESET_ID, SelfPlayEnv, upgrade_selfplay_record, BID_OFFSET, TRUMP_OFFSET,
    observation_from_record,
)
from texas42.agents import AgentPlayer, HybridAgent, VARIANTS
from texas42.cli import build_parser, main, make_players
from texas42.players import HumanPlayer, RandomPlayer


class SelfPlayLogTests(unittest.TestCase):
    def test_cli_log_replays_as_a_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "match.jsonl"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "--all-ai", "--seed", "42", "--target-marks", "3",
                    "--self-play-log", str(path),
                ]), 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        env = SelfPlayEnv(episode="match", target_marks=3, seed=42)
        observation = env.reset()
        hand = 1
        for record in records:
            self.assertEqual(observation_from_record(record), observation)
            self.assertEqual(record["schema_version"], 5)
            self.assertEqual(record["ruleset_id"], RULESET_ID)
            self.assertEqual(record["action_key"], ACTION_REGISTRY[record["action_id"]].key)
            schema4 = deepcopy(record)
            schema4["schema_version"] = 4
            del schema4["feature_schema_version"]
            del schema4["observation"]["feature_schema_version"]
            self.assertEqual(upgrade_selfplay_record(schema4), record)
            global_legacy = deepcopy(schema4)
            global_legacy["schema_version"] = 3
            del global_legacy["ruleset_id"], global_legacy["action_key"]
            del global_legacy["observation"]["ruleset_id"]
            del global_legacy["observation"]["view"]["contract"]
            if global_legacy["hand_result"]:
                del global_legacy["hand_result"]["contract"]
            self.assertEqual(upgrade_selfplay_record(global_legacy), record)
            # A full old-format dataset can be upgraded and replayed without
            # changing rewards or private/public decision information.
            legacy = deepcopy(global_legacy)
            legacy["schema_version"] = 2
            view = legacy["observation"]
            offset, size = {"play": (0, 28), "bid": (BID_OFFSET, 18),
                            "trump": (TRUMP_OFFSET, 9)}[view["phase"]]
            legacy["action_id"] -= offset
            view["legal_action_ids"] = [i - offset for i in view["legal_action_ids"]]
            view["action_mask"] = view["action_mask"][offset:offset + size]
            del view["action_space_version"], view["num_actions"]
            original = deepcopy(legacy)
            self.assertEqual(upgrade_selfplay_record(legacy), record)
            self.assertEqual(legacy, original)
            if record["observation"]["phase"] == "play":
                self.assertEqual(record["action"], record["action_id"])
            self.assertEqual(record["hand"], hand)
            self.assertEqual(record["shaker"], env.shaker)
            self.assertEqual(record["observation"], json.loads(json.dumps(asdict(observation))))
            transition = env.step(record["action_id"])
            expected_result = asdict(transition.hand_result) if transition.hand_result else None
            self.assertEqual(record["hand_result"], json.loads(json.dumps(expected_result)))
            self.assertEqual(record["match_marks"], list(env.match_marks))
            self.assertEqual(record["rewards"], list(transition.rewards))
            self.assertEqual(record["terminated"], transition.terminated)
            self.assertEqual(record["match_winner"], transition.match_winner)
            if transition.hand_result:
                hand += 1
            observation = transition.observation
        self.assertGreater(hand, 2)
        self.assertTrue(records[-1]["terminated"])
        self.assertIsNone(observation)

    def test_logging_requires_all_ai(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--self-play-log"])
        self.assertEqual(error.exception.code, 2)


class PolicyPlayerTests(unittest.TestCase):
    def test_defaults_preserve_placeholder_seeds(self):
        for all_ai in (False, True):
            players = make_players(42, all_ai)
            master = random.Random(42)
            if not all_ai:
                self.assertIsInstance(players[0], HumanPlayer)
                self.assertEqual(players[0].name, "You")
            for seat in range(0 if all_ai else 1, 4):
                self.assertIsInstance(players[seat], RandomPlayer)
                self.assertEqual(players[seat].name, f"Dummy AI {seat + 1}")
                self.assertEqual(players[seat].rng.getstate(),
                                 random.Random(master.getrandbits(64)).getstate())

    def test_mixed_seats_use_existing_independent_agents(self):
        players = make_players(42, player_types=("human", "HHH", "CHH", "HHR"))
        self.assertIsInstance(players[0], HumanPlayer)
        for player, variant in zip(players[1:], ("HHH", "CHH", "HHR")):
            self.assertIsInstance(player, AgentPlayer)
            self.assertIsInstance(player.agent, HybridAgent)
            self.assertIn(variant, player.name)
        self.assertEqual(len({id(p.agent) for p in players[1:]}), 3)
        self.assertIsInstance(make_players(42, player_types=("HHH", "human", "human", "HHH"))[1],
                              HumanPlayer)

    def test_all_variants_complete_matches_with_replayable_legal_actions(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "match.jsonl"
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([
                        "--players", *([variant] * 4), "--all-ai", "--seed", "42",
                        "--target-marks", "2", "--self-play-log", str(path),
                    ]), 0)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                env = SelfPlayEnv(episode="match", target_marks=2, seed=42)
                obs = env.reset()
                # Independently reconstruct policies: CLI and native env must
                # choose identical global IDs from identical private snapshots.
                players = make_players(42, True, [variant] * 4)
                for seat, player in enumerate(players):
                    player.reset(seat)
                hand_decisions = 0
                for record in records:
                    self.assertEqual(observation_from_record(record), obs)
                    action_id = players[obs.seat].agent.act(obs)
                    self.assertEqual(action_id, record["action_id"])
                    self.assertIn(action_id, obs.legal_action_ids)
                    self.assertTrue(obs.action_mask[action_id])
                    self.assertLess(action_id, 55)
                    transition = env.step(action_id)
                    hand_decisions += 1
                    if transition.hand_result:
                        self.assertEqual(hand_decisions, 33)
                        self.assertEqual(sum(transition.hand_result.team_points), 42)
                        self.assertEqual(record["hand_result"],
                                         json.loads(json.dumps(asdict(transition.hand_result))))
                        hand_decisions = 0
                    self.assertEqual(record["rewards"], list(transition.rewards))
                    self.assertEqual(record["match_marks"], list(env.match_marks))
                    obs = transition.observation
                self.assertIsNone(obs)
                self.assertTrue(records[-1]["terminated"])

    def test_interactive_cli_preserves_human_and_selected_opponents(self):
        # Stop at the first human decision rather than require terminal input.
        human = HumanPlayer(read=lambda _: (_ for _ in ()).throw(EOFError()))
        with patch("texas42.cli.HumanPlayer", return_value=human), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--players", "human", "HHH", "CHH", "HHR", "--seed", "42"]), 130)
        self.assertIn("Game ended.", output.getvalue())

    def test_invalid_player_selections_fail_before_gameplay(self):
        for args in (["--players", "human", "HHH"],
                     ["--players", "human", "HHH", "bad-policy", "CHH"],
                     ["--all-ai", "--players", "human", "HHH", "CHH", "HHR"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
        self.assertIn("--players", build_parser().format_help())

    def test_phase_overrides_reuse_composition_without_optional_dependencies(self):
        # The interactive entry point stays usable even when RL packages are blocked.
        code = '''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch', 'numpy', 'pyspiel'): raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from texas42.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
        result = subprocess.run([sys.executable, '-c', code, '--all-ai', '--players',
            'HHH', 'CHH', 'HHR', 'CRR', '--player-1-play', 'random', '--seed', '42',
            '--target-marks', '1'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[Heuristic Bid | Heuristic Trump | Random Play]', result.stdout)
        self.assertIn('Trick 7:', result.stdout)
        self.assertIn('win the match', result.stdout)

    def test_invalid_compositions_fail_before_opening_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log.jsonl'
            for options in (
                ['--players', 'neural', 'HHH', 'HHH', 'HHH'],
                ['--players', 'HHH', 'HHH', 'HHH', 'HHH', '--player-1-trump', 'conservative'],
                ['--players', 'human', 'HHH', 'HHH', 'HHH', '--player-1-play', 'neural'],
                ['--players', 'HHH', 'HHH', 'HHH', 'HHH', '--player-1-play-checkpoint', 'missing.pt'],
            ):
                with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(options + ['--all-ai', '--self-play-log', str(path)])
                self.assertFalse(path.exists())
