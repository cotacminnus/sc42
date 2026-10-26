"""Whole-hand CPU rollouts with chronological per-seat multi-phase records."""
from dataclasses import dataclass, replace
import hashlib
import os
import random
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
def collect_phase_hand(models, seed, episode, learn_phases, pool, critic=None, current_sha256=None, search=None):
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

    search (research.ppo.search.TrainingSearch, optional): search inside
    training (docs/search_training.md, P3), first pass. For each learner play
    decision, one draw from a dedicated per-hand stream decides whether it is
    searched (forced moves never are). The hand does not search inline: it
    returns row['search_requests'], one SearchRequest per selected decision,
    and the trainer runs them as separate, load-balanced pool tasks
    (search_chunk) before attach_search_values fills Experience.search_q.
    Selection uses its own stream and never touches the hand's action
    generator, so the hand, its actions and its records are exactly those of
    a run without search.
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
    if search is not None:
        search_draws = random.Random(stream_seed(seed, 'search-decisions-v1', episode))
        requests = []
    while True:
        if search is not None and obs.seat in paths and obs.phase == 'play' and 'play' in learn_phases:
            # Draw first, then check legality: the stream advances identically
            # whichever decisions are searched. The key names the record this
            # decision is about to append to its seat's path.
            if search_draws.random() < search.fraction and len(obs.legal_action_ids) > 1:
                requests.append(SearchRequest((episode, obs.seat, len(paths[obs.seat])), obs,
                                              stream_seed(seed, 'search-worlds-v1', episode * 64 + decisions)))
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
    row = dict(worker_rss_bytes=rss, episode=episode, paths=tuple(tuple(p) for p in paths.values()),
               reward=transition.rewards[team], opponent=label,
               opponent_sha256=current_sha256 if spec is None else spec.sha256,
               decisions=decisions, points=transition.hand_result.team_points, bidder=transition.hand_result.bidder,
               inference_seconds=inference, opponent_loading_seconds=loading,
               neural_inference_seconds=neural_inference + sum(getattr(a, 'neural_inference_seconds', 0.0) for a in agents))
    if search is not None:
        row['search_requests'] = tuple(requests)
    return row


def _load_learners(weights, hidden, neural):
    """Restore the rollout snapshot into this process's cached learner models; return the payloads.

    weights: one CPU snapshot (shared learner) or a tuple of two (paired
    partners). The cache is keyed on every payload, so hand and search tasks of
    the same rollout load the networks at most once per worker.
    """
    global _LEARNER, _VERSION
    payloads = weights if isinstance(weights, tuple) else (weights,)
    version = (tuple(hashlib.sha256(w).digest() for w in payloads), hidden, neural)
    if version != _VERSION:
        learners = []
        for payload in payloads:
            factory = PhaseFactory(payload, hidden, neural)
            factory.prepare()
            learners.append(factory.model)
        _LEARNER, _VERSION = tuple(learners), version
    return payloads


@dataclass(frozen=True)
class SearchRequest:
    """One decision selected for search (first pass of search inside training).

    key       (episode, seat, index of the record in that seat's path);
    obs       the learner's own Observation at the decision (no hidden hands);
    seed      the search's world/rollout seed: stream_seed(seed,
              'search-worlds-v1', episode * 64 + decision index in the hand).
    """
    key: tuple
    obs: object
    seed: int


def search_chunk(task):
    """Second pass: run searches with the rollout snapshot; [(key, search_q, seconds)], pid.

    task = (weights, hidden, neural, samples, requests). Every seat is played by
    its own learner network (partner rule in paired mode), as in the hand.
    """
    from .search import InMemoryGuide, search_values
    weights, hidden, neural, samples, requests = task
    _load_learners(weights, hidden, neural)
    models = _LEARNER
    guides = tuple(InMemoryGuide(models[partner_index(seat)] if len(models) == 2 else models[0])
                   for seat in range(4))
    results = []
    for request in requests:
        started = perf_counter()
        try:
            values = search_values(request.obs, guides, samples, request.seed)
        except Exception as exc:
            raise RuntimeError(f'Search key={request.key} worker={os.getpid()}: {exc}') from exc
        results.append((request.key, values, perf_counter() - started))
    return results, os.getpid()


def attach_search_values(rows, values):
    """Copy of hand rows with Experience.search_q filled from {key: search_q}; all keys must be used."""
    remaining = dict(values)
    attached = []
    for row in rows:
        paths = []
        for path in row['paths']:
            paths.append(tuple(replace(record, search_q=remaining.pop((record.episode_id, record.seat, i)))
                               if (record.episode_id, record.seat, i) in remaining else record
                               for i, record in enumerate(path)))
        attached.append({**row, 'paths': tuple(paths)})
    if remaining:
        raise RuntimeError(f'Search results without a decision: {sorted(remaining)[:3]}')
    return attached


def phase_chunk(task):
    global _CRITIC, _CRITIC_VERSION
    # optional: (critic weights or None[, TrainingSearch]); absent keeps the original task.
    weights, hidden, neural, learn, seed, indices, pool, *optional = task
    critic_weights = optional[0] if optional else None
    search = optional[1] if len(optional) > 1 else None
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
    payloads = _load_learners(weights, hidden, neural)
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
            rows.append(collect_phase_hand(_LEARNER, seed, episode, learn, pool, critic, current, search))
        except Exception as exc:
            raise RuntimeError(f'Multi-phase episode={episode} worker={os.getpid()}: {exc}') from exc
    return rows, os.getpid()
