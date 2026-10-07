import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path

from texas42 import SelfPlayEnv
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
            self.assertEqual(record["schema_version"], 2)
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
