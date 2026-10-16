"""Whole-hand CPU rollouts with chronological per-seat multi-phase records."""
from dataclasses import replace
import hashlib
import os
from time import perf_counter
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode
from texas42.selfplay import SelfPlayEnv
from .agent import PPOAgent
from .buffer import Experience
from .composition import partner_index
from .selfplay_evaluation import PhaseFactory
from .seeds import stream_seed

_FACTORIES = {}
_LEARNER = None
_VERSION = None


@torch.no_grad()
def collect_phase_hand(models, seed, episode, learn_phases, pool):
    """One complete hand; `models` is one shared learner or a (partner 1, partner 2) pair.

    Shared mode passes a single model (or a 1-tuple) and keeps the original
    records and random streams. Paired mode passes two models: each learner seat
    acts and records with its own partner's network. The per-hand sampling
    generator is shared, so draws interleave in decision order in both modes.
    """
    models = models if isinstance(models, tuple) else (models,)
    if len(models) not in (1, 2):
        raise ValueError('Expected one shared learner or two paired partners')

    def learner(seat):
        return models[partner_index(seat)] if len(models) == 2 else models[0]

    team = (episode + seed) % 2
    label, spec = pool.choose(seed, episode)
    started = perf_counter()
    key = spec
    if key not in _FACTORIES:
        factory = spec.factory()
        if hasattr(factory, 'prepare'):
            factory.prepare()
        _FACTORIES[key] = factory
    factory = _FACTORIES[key]
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
            neural_inference += perf_counter() - neural_started
            action = int(action.item())
            paths[obs.seat].append(Experience(features, obs.action_mask, action,
                float(logp.item()), float(value.item()), 0.0, False, obs.seat, episode))
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
                reward=transition.rewards[team], opponent=label, opponent_sha256=spec.sha256,
                decisions=decisions, points=transition.hand_result.team_points, bidder=transition.hand_result.bidder,
                inference_seconds=inference, opponent_loading_seconds=loading,
                neural_inference_seconds=neural_inference + sum(getattr(a, 'neural_inference_seconds', 0.0) for a in agents))


def phase_chunk(task):
    global _LEARNER, _VERSION
    weights, hidden, neural, learn, seed, indices, pool = task
    # Bound process-local opponent caches to the current roster plus anchor.
    from .checkpoint import _FROZEN_MODELS
    allowed = {pool.anchor, *pool.policies, *pool.snapshots}
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
    rows = []
    for episode in indices:
        try:
            rows.append(collect_phase_hand(_LEARNER, seed, episode, learn, pool))
        except Exception as exc:
            raise RuntimeError(f'Multi-phase episode={episode} worker={os.getpid()}: {exc}') from exc
    return rows, os.getpid()
