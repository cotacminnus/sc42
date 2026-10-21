"""Centralized critic, return estimators, oracle auxiliary data and checkpoint compatibility."""
import importlib.util
import unittest
from dataclasses import replace
import tempfile
from pathlib import Path

from research.ppo.oracle_dataset import (deal_split, label_positions, load_dataset, position_from_record,
                                         sample_positions)
from texas42.encoding import encode
from texas42.oracle import SearchLimits
from texas42.phases import Phase
from texas42.privileged import PRIVILEGED_SIZE, PRIVILEGED_SLICES, encode_privileged
from texas42.selfplay import SelfPlayEnv

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from research.ppo.agent import PPOAgent
    from research.ppo.buffer import Experience, RolloutBuffer, gae, monte_carlo
    from research.ppo.checkpoint import load_model, read, save as save_legacy
    from research.ppo.critic import CentralCritic, interval_loss
    from research.ppo.opponents import HistoricalPool, PolicySpec
    from research.ppo.selfplay_checkpoint import config_from_dict, resume, save
    from research.ppo.selfplay_rollout import collect_phase_hand
    from research.ppo.selfplay_trainer import SelfPlayConfig, SelfPlayTrainer
    from research.ppo.trainer import Config, Trainer
    from texas42.agents import VARIANTS


def play_observation(seed=3, plays=6):
    env = SelfPlayEnv(seed=seed)
    obs = env.reset()
    from texas42.agents import HeuristicAgent
    agents = [HeuristicAgent() for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(seat)
    while obs.phase != Phase.PLAY or len(obs.view.completed_tricks) * 4 + len(obs.view.trick) < plays:
        obs = env.step(agents[obs.seat].act(obs)).observation
    return env, obs


def swapped_hidden(hands, actor):
    """Exchange one domino between the two opponents; the actor's view is unchanged."""
    left, right = (actor + 1) % 4, (actor + 3) % 4
    hands = [list(h) for h in hands]
    hands[left][0], hands[right][0] = hands[right][0], hands[left][0]
    return tuple(tuple(sorted(h)) for h in hands)


class PrivilegedFeatureTests(unittest.TestCase):
    def test_privileged_hands_change_only_critic_features(self):
        env, obs = play_observation()
        hands = env.remaining_hands()
        other = swapped_hidden(hands, obs.seat)
        self.assertNotEqual(hands, other)
        critic_a, critic_b = encode_privileged(obs, hands), encode_privileged(obs, other)
        self.assertEqual(len(critic_a), PRIVILEGED_SIZE)
        self.assertNotEqual(critic_a, critic_b)
        public = PRIVILEGED_SLICES['public']
        self.assertEqual(critic_a[public], encode(obs))
        self.assertEqual(critic_a[PRIVILEGED_SLICES['perspective']], [1.0, 0.0])
        self.assertEqual(critic_a[PRIVILEGED_SLICES['next_player']], [1.0, 0.0, 0.0, 0.0])
        own = PRIVILEGED_SLICES['hands'].start
        self.assertEqual([i for i in range(28) if critic_a[own + i]], sorted(obs.view.hand))
        with self.assertRaises(ValueError):
            encode_privileged(obs, (hands[1], hands[0], hands[2], hands[3]))


@unittest.skipUnless(TORCH, 'optional rl extra is not installed')
class CriticTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        legacy = Trainer(Config(total_hands=1, hidden_size=8, device='cpu'))
        self.anchor_path = self.path / 'anchor.pt'
        save_legacy(self.anchor_path, legacy)
        self.initial, _ = load_model(self.anchor_path)
        self.anchor = PolicySpec.checkpoint('anchor', self.anchor_path)
        self.policies = tuple(PolicySpec(n) for n in VARIANTS) + (self.anchor,)
        positions = sample_positions(deals=12, seed=5, min_tricks=4, max_tricks=6, per_deal=2)
        self.dataset = self.path / 'labels.json'
        label_positions(positions, self.dataset, limits=SearchLimits(max_nodes=20000), fractions=(.6, .2, .2))

    def config(self, **changes):
        c = SelfPlayConfig(total_hands=8, hidden_size=8, device='cpu', seed=42, learn_phases=('bid', 'play'),
                           rollout_decisions=32, ppo_epochs=2, minibatch_size=8, anchor=self.anchor,
                           opponent_policies=self.policies, eval_deals=2, bootstrap_samples=5,
                           forgetting_hands=1, snapshot_every_hands=64, snapshot_pool_size=2)
        return replace(c, **changes)

    def central(self, **changes):
        from research.ppo.oracle_dataset import dataset_sha256
        return self.config(critic_mode='centralized', oracle_aux_coef=.5, oracle_dataset=str(self.dataset),
                           oracle_dataset_sha256=dataset_sha256(self.dataset), **changes)

    def trainer(self, config, **kwargs):
        trainer = SelfPlayTrainer(config, self.initial, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_actor_outputs_ignore_privileged_inputs(self):
        local = self.trainer(self.config())
        central = self.trainer(self.config(critic_mode='centralized'))
        # Identical actor initialization across critic modes.
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(local.model.parameters(), central.model.parameters())))
        pool = HistoricalPool((('HHH', '1'),), self.policies, self.anchor)
        for episode in range(3):
            plain = collect_phase_hand(local.model, 42, episode, ('bid', 'play'), pool)
            privileged = collect_phase_hand(local.model, 42, episode, ('bid', 'play'), pool, central.critic)
            for a, b in zip(plain['paths'], privileged['paths']):
                self.assertEqual([(r.action_id, r.old_log_probability, r.observation_features) for r in a],
                                 [(r.action_id, r.old_log_probability, r.observation_features) for r in b])
                self.assertTrue(all(r.critic_features is None for r in a))
                self.assertTrue(all(len(r.critic_features) == PRIVILEGED_SIZE for r in b))
        # Two deals that differ only in the opponents' hidden hands give the
        # same actor observation and action, but different critic inputs.
        from texas42.models import double_six_set
        from texas42.state import HandState
        from texas42.selfplay import observation_from_view
        deck = list(double_six_set())
        hands = [deck[i * 7:(i + 1) * 7] for i in range(4)]
        other = [list(h) for h in hands]
        other[2][0], other[3][0] = other[3][0], other[2][0]  # Shaker 0: seat 1 acts first.
        observations, critic_inputs, actions = [], [], []
        for deal in (hands, other):
            state = HandState(deal, 0, (0, 0), 7)
            obs = observation_from_view(state.view())
            observations.append(obs)
            ids = tuple(tuple(d.id for d in h) for h in state.hands)
            critic_inputs.append(torch.tensor([encode_privileged(obs, ids)]))
            x = torch.tensor([encode(obs)], dtype=torch.float32)
            mask = torch.tensor([obs.action_mask])
            with torch.no_grad():
                action, logp, _ = central.model.select(x, mask, generator=torch.Generator().manual_seed(3))
            actions.append((int(action), float(logp)))
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(actions[0], actions[1])
        self.assertFalse(torch.equal(*critic_inputs))
        self.assertNotEqual(central.critic(critic_inputs[0]).item(), central.critic(critic_inputs[1]).item())

    def test_partnership_reward_signs_and_value_perspective(self):
        t = self.trainer(self.config(critic_mode='centralized'))
        buffer, _ = t.collect()
        by_episode = {}
        for r in buffer.records:
            if r.done:
                by_episode.setdefault(r.episode_id, {})[r.seat] = r.reward
        for episode, rewards in by_episode.items():
            team = (42 + episode) % 2
            self.assertEqual(set(rewards), {team, team + 2})
            self.assertEqual(rewards[team], rewards[team + 2])
            self.assertNotEqual(rewards[team], 0)
        for r in buffer.records:
            self.assertEqual(r.critic_features[PRIVILEGED_SLICES['perspective']], (1.0, 0.0))
        env = SelfPlayEnv(seed=11)
        obs = env.reset()
        while True:
            result = env.step(obs.legal_action_ids[0])
            if result.terminated:
                break
            obs = result.observation
        r = result.rewards
        self.assertTrue(r[0] == r[2] == -r[1] == -r[3] != 0)

    def test_returns_across_teams_phases_terminals_and_truncation(self):
        bid, trump, play = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)
        def record(seat, features, reward=0., done=False, value=0.):
            return Experience(features, (True, True), 0, 0., value, reward, done, seat, 0)
        buffer = RolloutBuffer()
        # Seat 0 (declarer) crosses bid -> trump -> play; seat 1 is an opposing learner.
        buffer.add_trajectory([record(0, bid, value=.2), record(0, trump, value=.4),
                               record(0, play, 2., True, .9)], estimator='monte-carlo')
        buffer.add_trajectory([record(1, bid, value=-.3), record(1, play, -2., True, -1.)],
                              estimator='monte-carlo')
        self.assertEqual(buffer.returns, [2., 2., 2., -2., -2.])
        self.assertEqual([round(a, 6) for a in buffer.advantages], [1.8, 1.6, 1.1, -1.7, -1.])
        for values in ([.2, .4, .9], [0., 0., 0.]):
            self.assertEqual([round(x, 9) for x in gae([0., 0., 2.], values, 1., 1.)[1]],
                             monte_carlo([0., 0., 2.], values)[1])
        # Truncation bootstraps from the same team's next value; terminals bootstrap zero.
        self.assertEqual(monte_carlo([0., 0.], [0., 0.], bootstrap_value=1.5)[1], [1.5, 1.5])
        self.assertEqual(gae([0.], [.5], 1., .95, bootstrap_value=1.)[0], [.5])
        truncated = RolloutBuffer()
        truncated.add_trajectory([record(2, play, value=.1), record(2, play, value=.3)], bootstrap_value=.7)
        self.assertEqual(truncated.returns, gae([0., 0.], [.1, .3], 1., .95, .7)[1])
        self.assertAlmostEqual(truncated.returns[-1], .7)
        with self.assertRaises(ValueError):
            RolloutBuffer().add_trajectory([record(0, play)])
        with self.assertRaises(ValueError):
            RolloutBuffer().add_trajectory([record(0, play, 1., True)], bootstrap_value=0.)
        with self.assertRaises(ValueError):
            RolloutBuffer().add_trajectory([record(0, play, 1., True)], estimator='td')
        t = self.trainer(self.config(return_estimator='monte-carlo'))
        buffer, _ = t.collect()
        for episode in {r.episode_id for r in buffer.records}:
            for seat in {r.seat for r in buffer.records if r.episode_id == episode}:
                rows = [i for i, r in enumerate(buffer.records) if (r.episode_id, r.seat) == (episode, seat)]
                final = buffer.records[rows[-1]].reward
                self.assertEqual([buffer.returns[i] for i in rows], [final] * len(rows))

    def test_interval_loss(self):
        v = torch.tensor([.1, .5, .9, .3])
        lower, upper = torch.tensor([.2, .4, .4, .3]), torch.tensor([.6, .6, .8, .3])
        loss = interval_loss(v, lower, upper)
        self.assertTrue(torch.allclose(loss, torch.tensor([.01, 0., .01, 0.])))
        exact = torch.tensor([.25, .25, .25, .25])
        self.assertTrue(torch.allclose(interval_loss(v, exact, exact), (v - exact).square()))
        with self.assertRaises(ValueError):
            interval_loss(v, upper + 1, upper)

    def test_dataset_splits_cache_replay_and_compatibility(self):
        dataset = load_dataset(self.dataset)
        splits = {}
        for record in dataset.records:
            splits.setdefault(record['position']['deal_seed'], set()).add(record['split'])
            position = position_from_record(record)
            state = position.state()
            self.assertEqual(state.seat, position.seat)
            self.assertEqual(record['reference_team'], position.seat % 2)
            self.assertTrue(0 <= record['lower'] <= record['upper'] <= 42)
        self.assertTrue(all(len(s) == 1 for s in splits.values()))
        self.assertEqual({deal_split(d, 0, (.6, .2, .2)) for d in range(200)}, {'train', 'validation', 'test'})
        positions = [position_from_record(r) for r in dataset.records]
        again = label_positions(positions, self.dataset, limits=SearchLimits(max_nodes=20000), fractions=(.6, .2, .2))
        self.assertEqual(again.reused, len(positions))
        with self.assertRaises(ValueError):
            label_positions(positions, self.dataset, limits=SearchLimits(max_nodes=5), fractions=(.6, .2, .2))
        import json
        content = json.loads(self.dataset.read_text())
        for key, value in (('schema_version', 99), ('privileged_layout_version', 0), ('objective', 'contract')):
            broken = self.path / f'broken-{key}.json'
            broken.write_text(json.dumps(dict(header={**content['header'], key: value}, records=content['records'])))
            with self.assertRaises(ValueError):
                load_dataset(broken)
        crossing = self.path / 'crossing.json'
        records = [dict(r) for r in content['records']]
        same = [r for r in records if r['position']['deal_seed'] == records[0]['position']['deal_seed']]
        if len(same) > 1:
            same[1]['split'] = next(s for s in ('train', 'validation', 'test') if s != same[0]['split'])
            crossing.write_text(json.dumps(dict(header=content['header'], records=records)))
            with self.assertRaises(ValueError):
                load_dataset(crossing)
        with self.assertRaises(ValueError):
            load_dataset(self.dataset, sha256='0' * 64)

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            self.config(critic_mode='global')
        with self.assertRaises(ValueError):
            self.config(return_estimator='td')
        with self.assertRaises(ValueError):
            self.config(oracle_aux_coef=.1)  # requires centralized critic and a dataset
        with self.assertRaises(ValueError):
            self.config(critic_mode='centralized', oracle_aux_coef=.1)
        with self.assertRaises(ValueError):
            self.config(critic_mode='centralized', oracle_dataset=str(self.dataset), oracle_dataset_sha256='x')
        self.assertEqual((Config().gamma, SelfPlayConfig.__dataclass_fields__['critic_mode'].default,
                          SelfPlayConfig.__dataclass_fields__['oracle_aux_coef'].default), (1.0, 'local', 0.0))

    def test_auxiliary_update_resume_and_actor_only_inference(self):
        continuous = self.trainer(self.central())
        self.assertEqual(continuous.critic_initialization, 'fresh')
        before = [p.clone() for p in continuous.critic.oracle.parameters()]
        metrics = continuous.update(continuous.collect()[0])
        for key in ('oracle_aux_loss', 'oracle_aux_point_error', 'critic_gradient_norm', 'value_loss'):
            self.assertIn(key, metrics)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, continuous.critic.oracle.parameters())))
        self.assertIn('explained_variance', continuous.value_diagnostics)
        self.assertIn('play', continuous.value_diagnostics)
        path = self.path / 'central.pt'
        save(path, continuous)
        continuous.update(continuous.collect()[0])
        resumed = self.trainer(self.central())
        resume(path, resumed)
        resumed.update(resumed.collect()[0])
        for a, b in ((continuous.model, resumed.model), (continuous.critic, resumed.critic)):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))
        # Actor-only inference: no critic, oracle dataset or search needed.
        model, data = load_model(path)
        self.assertEqual(data['metadata']['critic_mode'], 'centralized')
        self.dataset.rename(self.path / 'moved.json')
        _, obs = play_observation(seed=4)
        agent = PPOAgent(model, seed=0)
        agent.reset(obs.seat)
        self.assertIn(agent.act(obs), obs.legal_action_ids)
        # Missing critic weights in a centralized checkpoint are rejected.
        broken = torch.load(path, weights_only=True)
        del broken['critic']
        torch.save(broken, self.path / 'broken.pt')
        with self.assertRaises(ValueError):
            read(self.path / 'broken.pt')

    def test_old_checkpoints_and_explicit_critic_initialization(self):
        local = self.trainer(self.config())
        self.assertIsNone(local.critic)
        old = self.path / 'local.pt'
        save(old, local)
        data = torch.load(old, weights_only=True)
        data['metadata'].pop('critic_mode')  # As written before critic modes existed.
        torch.save(data, old)
        model, data = load_model(old)
        config = config_from_dict(data['config'])
        self.assertEqual((config.critic_mode, config.return_estimator, config.oracle_aux_coef), ('local', 'gae', 0.0))
        fresh = SelfPlayTrainer(replace(self.config(), critic_mode='centralized'), model)
        self.addCleanup(fresh.close)
        self.assertEqual(fresh.critic_initialization, 'fresh')
        central = self.path / 'central.pt'
        save(central, fresh)
        _, saved = load_model(central)
        copied = SelfPlayTrainer(self.config(critic_mode='centralized'), model, initial_critic=saved['critic'])
        self.addCleanup(copied.close)
        self.assertEqual(copied.critic_initialization, 'checkpoint')
        # Exact resume refuses to switch critic modes.
        switched = self.trainer(self.config(critic_mode='centralized'))
        with self.assertRaises(ValueError):
            resume(old, switched)

    def test_cli_smoke_for_local_centralized_and_oracle_auxiliary(self):
        import contextlib
        import io
        import json
        from examples.train_selfplay_ppo import main
        from examples.oracle_labels import main as labels
        dataset = self.path / 'cli-labels.json'
        with contextlib.redirect_stdout(io.StringIO()):
            labels(['--output', str(dataset), '--deals', '6', '--max-nodes', '5000', '--time-seconds', '2',
                    '--log-dir', str(self.path / 'label-logs')])
        self.assertTrue(load_dataset(dataset).records)
        common = ['--init-play-checkpoint', str(self.anchor_path), '--learner-bid', 'neural-train',
                  '--learner-bid-source', 'new', '--learner-trump', 'heuristic', '--learner-play', 'neural-train',
                  '--opponent-pool', 'HHH:1', '--eval-opponents', 'HHH,anchor', '--selection-opponent', 'anchor',
                  '--total-hands', '4', '--rollout-decisions', '16', '--ppo-epochs', '1', '--minibatch-size', '8',
                  '--eval-deals', '1', '--bootstrap-samples', '2', '--forgetting-hands', '1', '--device', 'cpu']
        variants = {'A': [], 'B': ['--critic-mode', 'centralized'],
                    'C': ['--critic-mode', 'centralized', '--oracle-aux-coef', '0.5', '--oracle-dataset', str(dataset)]}
        rewards = {}
        for name, extra in variants.items():
            directory = self.path / name
            with contextlib.redirect_stdout(io.StringIO()):
                t = main(common + extra + ['--output-dir', str(directory), '--log-dir', str(self.path / (name + '-logs'))])
            row = json.loads((directory / 'progress.jsonl').read_text().splitlines()[-1])
            rewards[name] = row['mean_reward']
            self.assertEqual(row['critic_mode'], 'local' if name == 'A' else 'centralized')
            self.assertEqual('oracle_aux_loss' in row['metrics'], name == 'C')
            self.assertEqual(row['oracle_dataset'] is not None, name == 'C')
            self.assertGreater(t.updates, 0)
        self.assertEqual(len(set(rewards.values())), 1)  # Matched first rollouts across A/B/C.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            main(common + ['--oracle-aux-coef', '0.5', '--oracle-dataset', str(dataset),
                           '--output-dir', str(self.path / 'invalid')])


if __name__ == '__main__':
    unittest.main()
