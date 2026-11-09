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
from texas42.actions import ACTION_REGISTRY
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
    learner, label, spec, agents, loading = _episode_setup(
        models, seed, episode, learn_phases, pool, 'multiphase-opponent-actions-v1')
    team = (episode + seed) % 2
    generator = torch.Generator().manual_seed(stream_seed(seed, 'multiphase-actions-v1', episode))
    env = SelfPlayEnv(episode='hand', seed=stream_seed(seed, 'deal', episode))
    obs = env.reset()
    paths = {s: [] for s in range(4) if s % 2 == team}
    decisions, inference, neural_inference = 0, 0.0, 0.0
    calls = []
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
            action, seconds = _learner_decision(learner(obs.seat), obs, generator, critic, env,
                                                paths[obs.seat], episode)
            neural_inference += seconds
        else:
            action = agents[obs.seat].act(obs)
        inference += perf_counter() - started
        if obs.phase == 'bid':
            calls.append(bid_decision(obs, action, team))
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
    _resolve_auction(calls, transition.hand_result)
    row = dict(worker_rss_bytes=_worker_rss(), episode=episode, paths=tuple(tuple(p) for p in paths.values()),
               reward=transition.rewards[team], opponent=label,
               opponent_sha256=current_sha256 if spec is None else spec.sha256,
               decisions=decisions, points=transition.hand_result.team_points, bidder=transition.hand_result.bidder,
               bids=(_bid_record(transition.hand_result, team),),
               contracts=(_contract_record(transition.hand_result, team),),
               bid_decisions=tuple(calls),
               inference_seconds=inference, opponent_loading_seconds=loading,
               neural_inference_seconds=neural_inference + sum(getattr(a, 'neural_inference_seconds', 0.0) for a in agents))
    if search is not None:
        row['search_requests'] = tuple(requests)
    return row


def _episode_setup(models, seed, episode, learn_phases, pool, agent_stream, *, match=False):
    """Learner routing, opponent choice and four fresh agents for one episode.

    Hand episodes keep their original streams and construction order exactly.
    Match episodes (match=True) choose the opponent once per match from
    match-specific streams; agent_stream seeds every seat's own agent.
    """
    models = models if isinstance(models, tuple) else (models,)
    if len(models) not in (1, 2):
        raise ValueError('Expected one shared learner or two paired partners')

    def learner(seat):
        return models[partner_index(seat)] if len(models) == 2 else models[0]

    team = (episode + seed) % 2
    label, spec = pool.choose(seed, episode, match=match)
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
        agent_seed = stream_seed(seed, agent_stream, episode * 4 + seat)
        agent = (PPOAgent(learner(seat), seed=agent_seed) if seat % 2 == team else factory(seat, agent_seed))
        agent.reset(seat)
        if isinstance(agent, PPOAgent):
            agent.profile_inference = True
        agents.append(agent)
    return learner, label, spec, agents, loading


def _learner_decision(model, obs, generator, critic, env, path, episode):
    """Sample one trainable learner decision and append its record; (action, neural seconds).

    The actor sees only encode(obs); privileged critic features are stored
    beside the record and never enter action selection.
    """
    features = tuple(encode(obs))
    x = torch.tensor([features], dtype=torch.float32)
    mask = torch.tensor([obs.action_mask], dtype=torch.bool)
    started = perf_counter()
    action, logp, value = model.select(x, mask, generator=generator)
    critic_features = None
    if critic is not None:
        critic_features = tuple(encode_privileged(obs, env.remaining_hands()))
        value = critic(torch.tensor([critic_features], dtype=torch.float32))
    seconds = perf_counter() - started
    action = int(action.item())
    path.append(Experience(features, obs.action_mask, action, float(logp.item()), float(value.item()),
                           0.0, False, obs.seat, episode, critic_features))
    return action, seconds


BID_CATEGORIES = ('30', '31-41', '42', '2+ marks')
CALL_CATEGORIES = ('pass',) + BID_CATEGORIES


def bid_category(bid):
    """Rollout bid diagnostic bucket: 30, 31-41, 42 (one mark) or 2+ marks."""
    if bid.points is not None:
        return '30' if bid.points == 30 else '31-41'
    return '42' if bid.marks == 1 else '2+ marks'


def _bid_record(result, team):
    """(contract declared by the learner team, bid bucket) for one completed hand."""
    return result.bidder % 2 == team, bid_category(result.bid)


def _contract_record(result, team):
    """(learner team declared, bid bucket, contract made) for one completed hand."""
    return result.bidder % 2 == team, bid_category(result.bid), result.scoring_team == result.bidder % 2


def _call_category(call):
    """Bucket of one auction call: pass, 30, 31-41, 42 or 2+ marks."""
    return 'pass' if call is None else bid_category(call)


def bid_decision(obs, action, team):
    """Diagnostic record of one auction call, from the actor's public observation only.

    Holds the acting seat and partnership, the actor-relative match score
    before the call, whether passing was legal or the bid was forced, the
    selected call (bucket and exact action key), the standing high bid and
    the buckets of every legal alternative. 'auction_won' is filled in when
    the hand's contract resolves. No hidden hand enters the record.
    """
    view = obs.view
    own, other = view.match_marks[obs.seat % 2], view.match_marks[1 - obs.seat % 2]
    legal = [ACTION_REGISTRY[i].value for i in obs.legal_action_ids]
    return dict(seat=obs.seat, learner=obs.seat % 2 == team, own_marks=own, opponent_marks=other,
                score_state='leading' if own > other else 'trailing' if own < other else 'tied',
                pass_legal=None in legal, forced=bool(view.forced),
                call=_call_category(ACTION_REGISTRY[action].value), action=ACTION_REGISTRY[action].key,
                standing=None if view.current_bid is None else _call_category(view.current_bid),
                alternatives=tuple(sorted({_call_category(v) for v in legal}, key=CALL_CATEGORIES.index)),
                auction_won=None)


def _resolve_auction(records, result):
    """Mark each pending auction call with whether its partnership won the contract."""
    for record in records:
        if record['auction_won'] is None:
            record['auction_won'] = record['seat'] % 2 == result.bidder % 2


def _worker_rss():
    try:
        from pathlib import Path
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


@torch.no_grad()
def collect_phase_match(models, seed, match, learn_phases, pool, critic=None, current_sha256=None, *,
                        target_marks=7):
    """One complete match to target_marks; the match-win counterpart of collect_phase_hand.

    The learner partnership, the opponent and every agent are chosen once per
    match and persist across hands; SelfPlayEnv(episode='match') carries the
    marks and rotates the shaker. Each learner seat keeps ONE chronological
    trajectory across all hands (bid, trump and play records): hand
    boundaries are ordinary steps with zero reward, never terminal. At match
    end the signed result (+1 winning partnership, -1 losing) is attached to
    the last record of each nonempty learner-seat trajectory, which alone is
    marked done, so returns span the whole match. A seat without a decision
    in the final hand still receives the outcome on its last earlier record;
    a seat with no record at all (trump-only learner that never declared)
    yields an empty path that the trainer skips.

    All randomness uses match-indexed streams disjoint from the hand
    collector's, so hand-mode sampling is untouched. Search inside training
    is not supported here (SelfPlayConfig rejects it).
    """
    learner, label, spec, agents, loading = _episode_setup(
        models, seed, match, learn_phases, pool, 'multiphase-match-opponent-actions-v1', match=True)
    team = (match + seed) % 2
    generator = torch.Generator().manual_seed(stream_seed(seed, 'multiphase-match-actions-v1', match))
    env = SelfPlayEnv(episode='match', target_marks=target_marks,
                      seed=stream_seed(seed, 'match-deal-v1', match))
    obs = env.reset()
    paths = {s: [] for s in range(4) if s % 2 == team}
    decisions = hand_decisions = 0
    inference = neural_inference = 0.0
    hands, shakers, bids, contracts, calls = [], [env.shaker], [], [], []
    expected = int('bid' in learn_phases) + 7 * int('play' in learn_phases)
    lengths = {s: 0 for s in paths}
    while True:
        started = perf_counter()
        if obs.seat in paths and obs.phase in learn_phases:
            action, seconds = _learner_decision(learner(obs.seat), obs, generator, critic, env,
                                                paths[obs.seat], match)
            neural_inference += seconds
        else:
            action = agents[obs.seat].act(obs)
        inference += perf_counter() - started
        if obs.phase == 'bid':
            calls.append(bid_decision(obs, action, team))
        transition = env.step(action)
        decisions += 1
        hand_decisions += 1
        result = transition.hand_result
        if result is not None:
            if hand_decisions != 33 or sum(result.team_points) != 42:
                raise RuntimeError('Incomplete normal hand')
            for seat, path in paths.items():
                added = len(path) - lengths[seat]
                if not expected <= added <= expected + int('trump' in learn_phases):
                    raise RuntimeError('Incorrect controlled-seat trajectory length')
                lengths[seat] = len(path)
            if not transition.terminated and any(transition.rewards):
                raise RuntimeError('Intermediate match hands must not pay policy rewards')
            hands.append(result.marks_awarded if result.scoring_team == team else -result.marks_awarded)
            bids.append(_bid_record(result, team))
            contracts.append(_contract_record(result, team))
            _resolve_auction(calls, result)
            hand_decisions = 0
            if len(hands) > 2 * target_marks - 1:
                raise RuntimeError('Match exceeded its maximum length')
            if transition.terminated:
                break
            shakers.append(env.shaker)
        obs = transition.observation
    if transition.match_winner is None or set(map(abs, transition.rewards)) != {1}:
        raise RuntimeError('A match must end with signed unit team rewards')
    for seat, path in paths.items():
        if path:
            path[-1] = replace(path[-1], reward=float(transition.rewards[seat]), done=True)
    return dict(worker_rss_bytes=_worker_rss(), episode=match, paths=tuple(tuple(p) for p in paths.values()),
                reward=transition.rewards[team], opponent=label,
                opponent_sha256=current_sha256 if spec is None else spec.sha256,
                decisions=decisions, hands=len(hands), hand_marks=tuple(hands), bids=tuple(bids),
                contracts=tuple(contracts), bid_decisions=tuple(calls),
                shakers=tuple(shakers), final_marks=(env.match_marks[team], env.match_marks[1 - team]),
                inference_seconds=inference, opponent_loading_seconds=loading,
                neural_inference_seconds=neural_inference + sum(getattr(a, 'neural_inference_seconds', 0.0)
                                                                for a in agents))


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
    # optional: (critic weights or None[, TrainingSearch or None[, match target marks]]);
    # absent keeps the original task. A target collects complete matches, not hands.
    weights, hidden, neural, learn, seed, indices, pool, *optional = task
    critic_weights = optional[0] if optional else None
    search = optional[1] if len(optional) > 1 else None
    target_marks = optional[2] if len(optional) > 2 else None
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
            if target_marks is None:
                rows.append(collect_phase_hand(_LEARNER, seed, episode, learn, pool, critic, current, search))
            else:
                rows.append(collect_phase_match(_LEARNER, seed, episode, learn, pool, critic, current,
                                                target_marks=target_marks))
        except Exception as exc:
            kind = 'episode' if target_marks is None else 'match'
            raise RuntimeError(f'Multi-phase {kind}={episode} worker={os.getpid()}: {exc}') from exc
    return rows, os.getpid()
