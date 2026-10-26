"""Search inside training (docs/search_training.md, P1/P3/P4).

Tiny hidden_size=8 fixtures in the style of test_paired_selfplay. These check
mechanics, invariants and reproducibility, not whether search improves play.
"""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import contextlib
    from dataclasses import replace
    import io
    import json
    import tempfile
    from pathlib import Path
    import torch
    from research.ppo.trainer import Config, Trainer
    from research.ppo.checkpoint import save as save_legacy, load_model, read
    from research.ppo.opponents import PolicySpec
    from research.ppo.search import (InMemoryGuide, TrainingSearch, masked_log_softmax, search_minibatch_loss,
                                     search_update_targets)
    from research.ppo.selfplay_checkpoint import save, config_from_dict
    from research.ppo.selfplay_rollout import collect_phase_hand
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer, PairedSelfPlayTrainer
    from texas42.agents import VARIANTS


def parameters(model):
    return [p.detach().clone() for p in model.parameters()]


def same(first, second):
    return all(torch.equal(a, b) for a, b in zip(first, second))


def without_search(paths):
    return tuple(tuple(replace(r, search_q=None) for r in path) for path in paths)


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class SearchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        self.addCleanup(legacy.close)
        self.anchor_path = self.path / 'anchor.pt'
        save_legacy(self.anchor_path, legacy)
        self.initial, _ = load_model(self.anchor_path)
        self.anchor = PolicySpec.checkpoint('anchor', self.anchor_path)
        self.policies = tuple(PolicySpec(n) for n in VARIANTS) + (self.anchor,)

    def config(self, **changes):
        c = SelfPlayConfig(total_hands=4, hidden_size=8, device='cpu', seed=42, learn_phases=('bid', 'play'),
            rollout_decisions=16, ppo_epochs=2, minibatch_size=8, anchor=self.anchor,
            opponent_policies=self.policies, opponent_probabilities=(('HHH', '1'),),
            eval_deals=2, bootstrap_samples=5, forgetting_hands=1)
        return replace(c, **changes)

    def trainer(self, **changes):
        config = self.config(**changes)
        trainer = (PairedSelfPlayTrainer if config.partner_mode == 'paired' else SelfPlayTrainer)(config, self.initial)
        self.addCleanup(trainer.close)
        return trainer

    def test_configuration_rules(self):
        self.assertEqual(self.config().search_fraction, 0.0)  # Off by default.
        for changes in (dict(search_fraction=-0.1), dict(search_fraction=1.5), dict(search_samples=0),
                        dict(search_weight=-1.0), dict(search_target='argmax'), dict(search_clip=0.0),
                        dict(search_warmup_hands=-1), dict(search_fraction=0.5, learn_phases=('bid',))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        with self.assertRaises(ValueError):
            TrainingSearch(0.0, 2)
        t = self.trainer(search_fraction=0.5, search_warmup_hands=4)
        self.assertIsNone(t.training_search())  # Still warming up.
        t.hands = 4
        self.assertEqual(t.training_search(), TrainingSearch(0.5, 16))

    def test_search_never_changes_the_hand(self):
        for mode in ('shared', 'paired'):
            t = self.trainer(partner_mode=mode)
            models = t.model if mode == 'shared' else (t.model, t.partner.model)
            pool = t.opponent_pool()
            for episode in range(3):
                plain = collect_phase_hand(models, 42, episode, ('bid', 'play'), pool)
                full = collect_phase_hand(models, 42, episode, ('bid', 'play'), pool,
                                          search=TrainingSearch(1.0, 2))
                half = collect_phase_hand(models, 42, episode, ('bid', 'play'), pool,
                                          search=TrainingSearch(0.5, 2))
                with self.subTest(mode=mode, episode=episode):
                    # Identical hands and records: search only adds search_q.
                    self.assertEqual(without_search(full['paths']), plain['paths'])
                    self.assertEqual(without_search(half['paths']), plain['paths'])
                    self.assertEqual(full['reward'], plain['reward'])
                    records = [r for path in full['paths'] for r in path]
                    searched = [r for r in records if r.search_q is not None]
                    self.assertEqual(full['searched_decisions'], len(searched))
                    for r in records:
                        is_play = r.observation_features[2] == 1.0
                        legal = sum(r.legal_action_mask)
                        self.assertEqual(r.search_q is not None, is_play and legal > 1)
                        if r.search_q is not None:
                            self.assertTrue(all(v == 0.0 for v, ok in zip(r.search_q, r.legal_action_mask)
                                                if not ok))
                    # A searched decision's values depend only on its seed, never on the fraction.
                    by_key = {(r.seat, i): r.search_q for path in full['paths'] for i, r in enumerate(path)}
                    for path in half['paths']:
                        for i, r in enumerate(path):
                            if r.search_q is not None:
                                self.assertEqual(r.search_q, by_key[(r.seat, i)])
            again = collect_phase_hand(models, 42, 0, ('bid', 'play'), pool, search=TrainingSearch(1.0, 2))
            first = collect_phase_hand(models, 42, 0, ('bid', 'play'), pool, search=TrainingSearch(1.0, 2))
            self.assertEqual(again['paths'], first['paths'])  # Deterministic.

    def test_in_memory_guide_plays_like_the_learner(self):
        # P1: the guide plays the in-memory network greedily on its neural phases
        # (masked argmax) and Heuristic V1 on the others, consuming no RNG.
        from texas42.agents import HeuristicAgent
        from texas42.encoding import encode
        from texas42.selfplay import SelfPlayEnv
        t = self.trainer()
        guide = InMemoryGuide(t.model)
        env, neural, heuristic = SelfPlayEnv(seed=3), 0, 0
        obs = env.reset()
        while obs is not None:
            agent = guide(obs.seat, 0)
            agent.reset(obs.seat)
            action = agent.act(obs)
            if obs.phase in t.model.neural_phases:
                logits, _ = t.model(torch.tensor([encode(obs)]))
                expected = int(masked_log_softmax(logits, torch.tensor([obs.action_mask])).argmax(-1))
                neural += 1
            else:
                reference = HeuristicAgent()
                reference.reset(obs.seat)
                expected = reference.act(obs)
                heuristic += 1
            self.assertEqual(action, expected)
            obs = env.step(action).observation
        self.assertGreater(neural, 5)
        self.assertGreater(heuristic, 0)  # Trump is heuristic in a bid+play learner.

    def test_zero_weight_search_is_bitwise_plain_ppo(self):
        plain = self.trainer()
        buffer, _ = plain.collect()
        plain.update(buffer)
        searched = self.trainer(search_fraction=1.0, search_samples=2, search_weight=0.0)
        buffer, _ = searched.collect()
        self.assertTrue(any(r.search_q is not None for r in buffer.records))
        metrics = searched.update(buffer)
        self.assertGreater(metrics['searched_states'], 0)
        self.assertTrue(same(parameters(plain.model), parameters(searched.model)))

    def test_search_losses_update_the_policy(self):
        for target in ('cmpo', 'advantage'):
            with self.subTest(target=target):
                t = self.trainer(search_fraction=1.0, search_samples=2, search_weight=1.0, search_target=target)
                start = parameters(t.model.networks['play'])
                buffer, _ = t.collect()
                self.assertGreater(t.rollout_performance['searched_decisions'], 0)
                self.assertGreater(t.rollout_performance['search_seconds'], 0)
                metrics = t.update(buffer)
                self.assertTrue(all(key in metrics for key in
                                    ('searched_states', 'search_loss', 'search_disagreement', 'search_abs_advantage')))
                self.assertEqual('search_target_kl' in metrics, target == 'cmpo')
                if target == 'cmpo':
                    self.assertLessEqual(metrics['search_target_kl'], 2 * t.config.search_clip + 1e-6)
                self.assertFalse(same(start, parameters(t.model.networks['play'])))

    def test_advantage_loss_gradient_is_the_all_action_gradient(self):
        # docs/search_training.md 4.3: at θ_old the all-action clipped surrogate's
        # gradient is Σ_a ∇π_θ(a) Â(a) (per state), with Â the normalized advantage.
        t = self.trainer(search_fraction=1.0, search_samples=2)
        buffer, _ = t.collect()
        data = buffer.tensors()
        search_update_targets(t.model, data, 'advantage', 2.0)
        rows = data['searched']
        net = t.model.networks['play']
        net.zero_grad()
        summed, count = search_minibatch_loss(t.model, data, 'advantage', 0.2)
        summed.backward()

        def gradients():  # The value head gets no gradient from either policy loss.
            return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in net.parameters()]

        surrogate = gradients()
        net.zero_grad()
        logits, _ = t.model(data['features'][rows])
        probabilities = masked_log_softmax(logits, data['masks'][rows]).exp()
        (-(probabilities * data['search_advantage'][rows]).sum()).backward()
        reference = gradients()
        self.assertTrue(any(g.abs().sum() > 0 for g in reference))
        self.assertEqual(count, int(rows.sum()))
        for a, b in zip(surrogate, reference):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_worker_count_invariance_with_search(self):
        results = []
        for threads in (1, 2):
            t = self.trainer(threads=threads, search_fraction=0.5, search_samples=2)
            buffer, reward = t.collect()
            records = list(buffer.records)  # The update clears the buffer.
            metrics = t.update(buffer)
            results.append((records, reward, parameters(t.model), metrics['searched_states']))
        (records_1, reward_1, parameters_1, searched_1), (records_2, reward_2, parameters_2, searched_2) = results
        self.assertEqual((records_1, reward_1, searched_1), (records_2, reward_2, searched_2))
        self.assertTrue(any(r.search_q is not None for r in records_1))
        self.assertTrue(same(parameters_1, parameters_2))

    def test_checkpoint_round_trip_and_old_configs(self):
        t = self.trainer(search_fraction=0.25, search_samples=4, search_target='advantage',
                         search_warmup_hands=2)
        path = self.path / 'search.pt'
        save(path, t)
        config = config_from_dict(read(path)['config'])
        self.assertEqual((config.search_fraction, config.search_samples, config.search_target,
                          config.search_warmup_hands), (0.25, 4, 'advantage', 2))
        stored = dict(read(path)['config'])
        for key in [k for k in stored if k.startswith('search_')]:
            del stored[key]  # As written before search inside training existed.
        self.assertEqual(config_from_dict(stored).search_fraction, 0.0)

    def test_cli_smoke(self):
        from examples.train_selfplay_ppo import main as train
        run = self.path / 'run'
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train(['--init-play-checkpoint', str(self.anchor_path), '--learn-phases', 'bid,play',
                   '--total-hands', '2', '--rollout-decisions', '8', '--ppo-epochs', '1', '--minibatch-size', '4',
                   '--eval-deals', '2', '--bootstrap-samples', '5', '--forgetting-hands', '1',
                   '--eval-opponents', 'HHH,anchor', '--opponent-pool', 'HHH:1',
                   '--search-fraction', '1', '--search-samples', '2', '--search-target', 'cmpo',
                   '--output-dir', str(run), '--log-dir', str(self.path / 'log')])
        row = json.loads((run / 'progress.jsonl').read_text().splitlines()[-1])
        self.assertGreater(row['rollout_performance']['searched_decisions'], 0)
        self.assertGreater(row['metrics']['searched_states'], 0)
        self.assertIn('search:', output.getvalue())
        self.assertEqual(json.loads((run / 'config.json').read_text())['search_fraction'], 1.0)


if __name__ == '__main__':
    unittest.main()
