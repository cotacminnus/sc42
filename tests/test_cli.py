import io
import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path

from texas42 import (
    ACTION_REGISTRY, RULESET_ID, SelfPlayEnv, upgrade_selfplay_record, BID_OFFSET, TRUMP_OFFSET,
    observation_from_record,
)
from texas42.cli import main


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
