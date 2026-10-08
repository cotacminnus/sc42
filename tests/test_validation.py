from copy import deepcopy
import io
import json
import unittest
from contextlib import redirect_stdout
import tempfile
from pathlib import Path

from texas42 import (
    ACTION_REGISTRY, SelfPlayEnv, TrainingCollector, ValidationError, validate_episode,
)
from texas42.cli import main


def collect(*, privileged=True, episode='hand', target_marks=3):
    records = []
    env = SelfPlayEnv(episode=episode, target_marks=target_marks, seed=42,
                      collector=TrainingCollector(records.append, include_initial_deal=privileged))
    obs = env.reset()
    while obs:
        obs = env.step(obs.legal_action_ids[0]).observation
    return json.loads(json.dumps(records))


class ValidationTests(unittest.TestCase):
    def test_verified_hand_and_match_replay_without_mutating_input(self):
        for mode in ('hand', 'match'):
            records = collect(episode=mode)
            before = deepcopy(records)
            result = validate_episode(iter(records), episode=mode)
            self.assertEqual(result.status, 'verified_replay')
            self.assertEqual(result.records, len(records))
            self.assertEqual(result.hands, records[-1]['hand'])
            self.assertEqual(result.limitations, ())
            self.assertEqual(records, before)
            if mode == 'match':
                self.assertGreater(result.hands, 1)

    def test_without_deals_and_mixed_deals_remain_explicitly_partial(self):
        for mode in ('hand', 'match'):
            records = collect(episode=mode, privileged=False)
            result = validate_episode(records, episode=mode)
            self.assertEqual(result.status, 'partial')
            self.assertTrue(result.limitations)
            # Existing schema-5 records need not contain the optional sidecar.
            for record in records:
                del record['public_history']
            self.assertEqual(validate_episode(records, episode=mode).status, 'partial')
        records = collect()
        del records[0]['privileged']
        self.assertEqual(validate_episode(records, episode='hand').status, 'partial')

    def test_existing_cli_schema5_match_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'match.jsonl'
            with redirect_stdout(io.StringIO()):
                main(['--all-ai', '--seed', '42', '--target-marks', '2', '--self-play-log', str(path)])
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(validate_episode(records, episode='match').status, 'partial')

    def test_corrupt_actions_rewards_history_contracts_and_labels_fail(self):
        for privileged in (False, True):
            source = collect(privileged=privileged)
            def action(records):
                # A different legal initial call, with consistent decoded labels,
                # still disagrees with the next record's observed auction.
                records[0]['action_id'] = 29
                records[0]['action_key'] = ACTION_REGISTRY[29].key
                records[0]['action'] = {'points': 30, 'marks': None}
            def reward(records):
                records[-1]['rewards'][0] += 1
            def history(records):
                records[1]['public_history']['events'][0]['seat'] = (records[0]['observation']['seat'] + 1) % 4
            def contract(records):
                records[5]['observation']['view']['contract']['bidder'] = (records[5]['observation']['seat'] + 1) % 4
            def marks(records):
                records[-1]['match_marks'][0] += 1
            def result(records):
                records[-1]['hand_result']['marks_awarded'] += 1
            def termination(records):
                records[-1]['terminated'] = False
            def mask(records):
                records[0]['observation']['action_mask'][29] = False
                records[0]['observation']['legal_action_ids'].remove(29)
            def illegal_action(records):
                # The fourth seat must bid after three passes. The ID itself is
                # valid and in the correct phase, but the engine must reject it.
                records[3]['action_id'] = 28
                records[3]['action_key'] = ACTION_REGISTRY[28].key
                records[3]['action'] = None
            for corrupt in (action, illegal_action, reward, history, contract, marks, result, termination, mask):
                records = deepcopy(source)
                corrupt(records)
                with self.subTest(privileged=privileged, corruption=corrupt.__name__):
                    with self.assertRaisesRegex(ValidationError, 'record'):
                        validate_episode(records, episode='hand')

    def test_mark_rewards_and_all_trump_choices_replay(self):
        for trump_id in range(46, 55):
            records = []
            env = SelfPlayEnv(seed=trump_id, collector=TrainingCollector(
                records.append, include_initial_deal=True))
            obs = env.reset()
            while obs:
                action = trump_id if obs.phase == 'trump' else obs.legal_action_ids[-1]
                obs = env.step(action).observation
            self.assertEqual(abs(records[-1]['rewards'][0]), 5)
            self.assertEqual(validate_episode(records, episode='hand').status, 'verified_replay')

    def test_corrupt_initial_deals_fail_even_when_permutations_remain_valid(self):
        source = collect()
        for corruption in ('duplicate', 'wrong_owner', 'changed_copy', 'bool_id', 'missing'):
            records = deepcopy(source)
            if corruption == 'duplicate':
                records[0]['privileged']['initial_deal'][0][0] = records[0]['privileged']['initial_deal'][1][0]
            elif corruption == 'wrong_owner':
                # Corrupt every copy identically: cannot rely on copy consistency.
                for record in records:
                    deal = record['privileged']['initial_deal']
                    deal[0][0], deal[1][0] = deal[1][0], deal[0][0]
            elif corruption == 'changed_copy':
                deal = records[8]['privileged']['initial_deal']
                deal[0], deal[1] = deal[1], deal[0]
            elif corruption == 'bool_id':
                records[0]['privileged']['initial_deal'][0][0] = True
            else:
                del records[0]['privileged']['initial_deal']
            with self.subTest(corruption=corruption), self.assertRaises(ValidationError):
                validate_episode(records, episode='hand')

    def test_truncation_ordering_versions_and_mode_rejected(self):
        source = collect()
        for records in ([], source[:-1], source[1:], source + source[-1:]):
            with self.assertRaises(ValidationError):
                validate_episode(records, episode='hand')
        for key, value in (('hand', 2), ('episode', 999), ('schema_version', 6)):
            records = deepcopy(source)
            records[1][key] = value
            with self.assertRaises(ValidationError):
                validate_episode(records, episode='hand')
        with self.assertRaises(ValidationError):
            validate_episode(source, episode='match')

    def test_collector_failure_aborts_rollout_without_retry_or_rollback(self):
        for fail_at in (1, 33):
            written = []
            failed = False
            def sink(record):
                nonlocal failed
                written.append(record)  # A sink may write before reporting failure.
                if len(written) == fail_at and not failed:
                    failed = True
                    raise OSError('disk full')
            env = SelfPlayEnv(episode='match', target_marks=3, seed=42,
                              collector=TrainingCollector(sink, include_initial_deal=True))
            obs = env.reset()
            for _ in range(fail_at - 1):
                obs = env.step(obs.legal_action_ids[0]).observation
            action = obs.legal_action_ids[0]
            with self.assertRaisesRegex(OSError, 'disk full'):
                env.step(action)
            # The action was applied exactly once, even though the caller got no
            # transition. The same pass would otherwise be legal for the next seat.
            state = env._session.hand
            self.assertEqual(len(state.bid_history), 1 if fail_at == 1 else 4)
            if fail_at == 33:
                self.assertIsNotNone(state.result)
                self.assertTrue(env._session.needs_deal)
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'rollout aborted'):
                    env.step(action)
                with self.assertRaisesRegex(RuntimeError, 'rollout aborted'):
                    env.observe()
            self.assertEqual(len(written), fail_at)
            obs = env.reset(seed=12)
            env.step(obs.legal_action_ids[0])
            self.assertEqual(written[-1]['episode'], 2)
            self.assertEqual(written[-1]['hand'], 1)


if __name__ == '__main__':
    unittest.main()
