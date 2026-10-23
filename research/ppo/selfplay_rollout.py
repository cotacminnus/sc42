"""Whole-hand CPU rollouts with chronological per-seat multi-phase records."""
from dataclasses import replace
import hashlib
import os
from time import perf_counter
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode
from texas42.privileged import encode_privileged
from texas42.selfplay import SelfPlayEnv
from .agent import PPOAgent
from .buffer import Experience
from .composition import partner_index
from .selfplay_evaluation import PhaseFactory
from .seeds import stream_seed

_FACTORIES = {}
_LEARNER = None
_VERSION = None
_CRITIC = None
_CRITIC_VERSION = None


@torch.no_grad()
def collect_phase_hand(models, seed, episode, learn_phases, pool, critic=None, current_sha256=None):
    """One complete hand; `models` is one shared learner or a (partner 1, partner 2) pair.

    Shared mode passes a single model (or a 1-tuple) and keeps the original
    records and random streams. Paired mode passes two models: each learner seat
    acts and records with its own partner's network. The per-hand sampling
    generator is shared, so draws interleave in decision order in both modes.

    critic=None keeps the local actor-critic values.

    A pool choice without a frozen spec (self-current, or self-snapshots before
    any snapshot in an anchor-free run) seats the learner's own rollout snapshot
    as the opponent partnership: the same fixed weights, its own per-seat action
    streams, stochastic legal sampling on trainable phases, greedy on frozen
    ones and heuristic phases unchanged. Its decisions never enter the buffer.
    In paired mode that opponent partnership is the current pair, routed by the
    same partner rule (first seat -> partner 1, second seat -> partner 2).
    current_sha256 identifies that rollout snapshot in the opponent log.

    With a centralized critic, the actor still sees only `encode(obs)`; the
    privileged features and the critic's value are stored beside the decision
    for the update and never enter action selection.
    """
    models = models if isinstance(models, tuple) else (models,)
    if len(models) not in (1, 2):
        raise ValueError('Expected one shared learner or two paired partners')

    def learner(seat):
        return models[partner_index(seat)] if len(models) == 2 else models[0]

    team = (episode + seed) % 2
    label, spec = pool.choose(seed, episode)
    started = perf_counter()
    if spec is None:
        def factory(seat, agent_seed):
            return PPOAgent(learner(seat), seed=agent_seed, stochastic_phases=learn_phases)
    else:
        if spec not in _FACTORIES:
            created = spec.factory()
            if hasattr(created, 'prepare'):
                created.prepare()
            _FACTORIES[spec] = created
        factory = _FACTORIES[spec]
    loading = perf_counter() - started
    agents = []
    for seat in range(4):
        agent_seed = stream_seed(seed, 'multiphase-opponent-actions-v1', episode * 4 + seat)
        agent = (PPOAgent(learner(seat), seed=agent_seed) if seat % 2 == team else factory(seat, agent_seed))
        agent.reset(seat)
        if isinstance(agent, PPOAgent):
            agent.profile_inference = True
        agents.append(agent)
    generator = torch.Generator().manual_seed(stream_seed(seed, 'multiphase-actions-v1', episode))
    env = SelfPlayEnv(episode='hand', seed=stream_seed(seed, 'deal', episode))
    obs = env.reset()
    paths = {s: [] for s in range(4) if s % 2 == team}
    decisions, inference, neural_inference = 0, 0.0, 0.0
    while True:
        started = perf_counter()
        if obs.seat in paths and obs.phase in learn_phases:
            features = tuple(encode(obs))
            x = torch.tensor([features], dtype=torch.float32)
            mask = torch.tensor([obs.action_mask], dtype=torch.bool)
            neural_started = perf_counter()
            action, logp, value = learner(obs.seat).select(x, mask, generator=generator)
            critic_features = None
            if critic is not None:
                critic_features = tuple(encode_privileged(obs, env.remaining_hands()))
                value = critic(torch.tensor([critic_features], dtype=torch.float32))
            neural_inference += perf_counter() - neural_started
            action = int(action.item())
            paths[obs.seat].append(Experience(features, obs.action_mask, action,
                float(logp.item()), float(value.item()), 0.0, False, obs.seat, episode,
                critic_features))
        else:
            action = agents[obs.seat].act(obs)
        inference += perf_counter() - started
        transition = env.step(action)
        decisions += 1
        if transition.terminated:
            break
        obs = transition.observation
    if decisions != 33 or sum(transition.hand_result.team_points) != 42:
        raise RuntimeError('Incomplete normal hand')
    expected = int('bid' in learn_phases) + 7 * int('play' in learn_phases)
    for seat, path in paths.items():
        if not expected <= len(path) <= expected + int('trump' in learn_phases):
            raise RuntimeError('Incorrect controlled-seat trajectory length')
        if path:
            path[-1] = replace(path[-1], reward=float(transition.rewards[seat]), done=True)
    try:
        from pathlib import Path
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        rss = None
    return dict(worker_rss_bytes=rss, episode=episode, paths=tuple(tuple(p) for p in paths.values()),
                reward=transition.rewards[team], opponent=label,
                opponent_sha256=current_sha256 if spec is None else spec.sha256,
                decisions=decisions, points=transition.hand_result.team_points, bidder=transition.hand_result.bidder,
                inference_seconds=inference, opponent_loading_seconds=loading,
                neural_inference_seconds=neural_inference + sum(getattr(a, 'neural_inference_seconds', 0.0) for a in agents))


def phase_chunk(task):
    global _LEARNER, _VERSION, _CRITIC, _CRITIC_VERSION
    weights, hidden, neural, learn, seed, indices, pool, *optional = task
    critic_weights = optional[0] if optional else None
    # Bound process-local opponent caches to the current roster plus anchor.
    from .checkpoint import _FROZEN_MODELS
    allowed = {s for s in (pool.anchor, *pool.policies, *pool.snapshots) if s is not None}
    for spec in tuple(_FACTORIES):
        if spec not in allowed:
            del _FACTORIES[spec]
    identities = {(i.path, i.sha256) for s in allowed for i in s.identities()}
    for key in tuple(_FROZEN_MODELS):
        if key not in identities:
            del _FROZEN_MODELS[key]
    # weights: one CPU snapshot (shared learner) or a tuple of two (paired
    # partners). The cached learner models are keyed on every payload.
    payloads = weights if isinstance(weights, tuple) else (weights,)
    version = (tuple(hashlib.sha256(w).digest() for w in payloads), hidden, neural)
    if version != _VERSION:
        learners = []
        for payload in payloads:
            factory = PhaseFactory(payload, hidden, neural)
            factory.prepare()
            learners.append(factory.model)
        _LEARNER, _VERSION = tuple(learners), version
    critic = None
    if critic_weights is not None:
        critic_version = (hashlib.sha256(critic_weights).digest(), hidden)
        if critic_version != _CRITIC_VERSION:
            from .critic import CentralCritic
            from io import BytesIO
            with torch.random.fork_rng(devices=[]):
                _CRITIC = CentralCritic(hidden)
            _CRITIC.load_state_dict(torch.load(BytesIO(critic_weights), map_location='cpu', weights_only=True))
            _CRITIC.eval()
            _CRITIC_VERSION = critic_version
        critic = _CRITIC
    # Identity of the learner snapshot(s) for self-current opponent logs; a
    # single payload hashes exactly as before pairing existed.
    rows, current = [], hashlib.sha256(b''.join(payloads)).hexdigest()
    for episode in indices:
        try:
            rows.append(collect_phase_hand(_LEARNER, seed, episode, learn, pool, critic, current))
        except Exception as exc:
            raise RuntimeError(f'Multi-phase episode={episode} worker={os.getpid()}: {exc}') from exc
    return rows, os.getpid()
