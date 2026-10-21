"""Offline oracle labels for the critic's auxiliary head (never an actor input).

Workflow: sample_positions() plays seeded hands with fixed policies and keeps
play-phase decision points; label_positions() runs the bounded position oracle
once per position and caches results in one versioned JSON file; load_dataset()
validates compatibility; training_tensors() rebuilds privileged critic features
and normalized bounds for one deal-disjoint split.

Each position stores its full initial deal, shaker and global action prefix, so
the engine replays it exactly. Labels are certified final-point bounds for the
acting seat's team (the critic's perspective), never an estimate. Only torch is
needed by training_tensors(); sampling and labeling are dependency-free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from time import monotonic

from texas42.actions import ACTION_SPACE_VERSION
from texas42.agents import variant_factory
from texas42.features import ENCODER_LAYOUT_VERSION
from texas42.models import double_six_set
from texas42.oracle import SearchLimits, evaluate_position, position_from_state
from texas42.phases import Phase
from texas42.privileged import PRIVILEGED_LAYOUT_VERSION, encode_privileged
from texas42.rulesets import RULESET_ID
from texas42.selfplay import SelfPlayEnv, observation_from_view
from texas42.state import HandState
from .seeds import stream_seed

KIND = 'sc42-oracle-labels'
SCHEMA_VERSION = 1
OBJECTIVE = 'final_points'
SPLITS = ('train', 'validation', 'test')


@dataclass(frozen=True)
class SampledPosition:
    deal_seed: int
    shaker: int
    initial_hands: tuple[tuple[int, ...], ...]
    actions: tuple[int, ...]  # Global action IDs from the start of the hand.
    seat: int
    tricks_done: int
    policy: str

    @property
    def key(self) -> str:
        text = json.dumps([self.initial_hands, self.shaker, self.actions], separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()

    def state(self) -> HandState:
        """Replay through the engine; any inconsistency raises."""
        dominoes = {d.id: d for d in double_six_set()}
        state = HandState([[dominoes[i] for i in hand] for hand in self.initial_hands], self.shaker, (0, 0), 7)
        for action in self.actions:
            state.apply(action)
        if state.phase != Phase.PLAY or state.seat != self.seat or len(state.completed) != self.tricks_done:
            raise ValueError('Stored position does not replay to its recorded decision')
        return state


def deal_split(deal_seed: int, split_seed: int, fractions: tuple[float, float, float]) -> str:
    """Deterministic split of a whole deal; all its positions share the result."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('Split fractions must be three nonnegative values summing to one')
    u = stream_seed(split_seed, 'oracle-deal-split-v1', deal_seed) / 2**63
    return 'train' if u < fractions[0] else 'validation' if u < fractions[0] + fractions[1] else 'test'


def sample_positions(*, deals: int, seed: int, policy: str = 'HHH', min_tricks: int = 4,
                     max_tricks: int = 6, per_deal: int = 1, max_positions: int | None = None
                     ) -> list[SampledPosition]:
    """Play `deals` seeded hands with `policy` on all seats; keep play decisions.

    min/max_tricks select the game stage by completed tricks (0-6); late
    positions give exact labels cheaply. At most per_deal positions per deal.
    """
    if not 0 <= min_tricks <= max_tricks <= 6:
        raise ValueError('Expected 0 <= min_tricks <= max_tricks <= 6')
    if deals <= 0 or per_deal <= 0 or (max_positions is not None and max_positions <= 0):
        raise ValueError('deals, per_deal and max_positions must be positive')
    factory = variant_factory(policy)
    result = []
    for deal in range(deals):
        deal_seed = stream_seed(seed, 'oracle-position-deal-v1', deal)
        env = SelfPlayEnv(episode='hand', seed=deal_seed)
        obs = env.reset()
        initial, shaker = env.remaining_hands(), env.shaker
        agents = [factory(s, stream_seed(seed, 'oracle-position-agent-v1', deal * 4 + s)) for s in range(4)]
        for s, agent in enumerate(agents):
            agent.reset(s)
        prefix, candidates = [], []
        while obs is not None:
            if obs.phase == Phase.PLAY and min_tricks <= len(obs.view.completed_tricks) <= max_tricks:
                candidates.append(SampledPosition(deal_seed, shaker, initial, tuple(prefix), obs.seat,
                                                  len(obs.view.completed_tricks), policy))
            action = agents[obs.seat].act(obs)
            prefix.append(action)
            obs = env.step(action).observation
        chosen = random.Random(stream_seed(seed, 'oracle-position-choice-v1', deal)).sample(
            candidates, min(per_deal, len(candidates)))
        for position in sorted(chosen, key=lambda p: len(p.actions)):
            result.append(position)
            if max_positions is not None and len(result) >= max_positions:
                return result
    return result


@dataclass
class OracleDataset:
    header: dict
    records: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        widths = [r['upper'] - r['lower'] for r in self.records]
        return dict(positions=len(self.records),
                    deals=len({r['position']['deal_seed'] for r in self.records}),
                    splits={s: sum(r['split'] == s for r in self.records) for s in SPLITS},
                    exact_fraction=sum(w == 0 for w in widths) / len(widths) if widths else None,
                    mean_bound_width_points=sum(widths) / len(widths) if widths else None,
                    labeling_seconds=sum(r['elapsed_seconds'] for r in self.records),
                    labeling_nodes=sum(r['nodes'] for r in self.records))


def _header(limits: SearchLimits, move_ordering: str, split_seed: int, fractions) -> dict:
    return dict(kind=KIND, schema_version=SCHEMA_VERSION, objective=OBJECTIVE, units='final_team_points',
                perspective='acting seat team', ruleset_id=RULESET_ID, action_space_version=ACTION_SPACE_VERSION,
                encoder_layout_version=ENCODER_LAYOUT_VERSION, privileged_layout_version=PRIVILEGED_LAYOUT_VERSION,
                search_budget=dict(scope='per action search', max_nodes=limits.max_nodes,
                                   time_seconds=limits.time_seconds, max_cache_entries=limits.max_cache_entries,
                                   move_ordering=move_ordering),
                split=dict(seed=split_seed, fractions=list(fractions), unit='deal'))


_COMPATIBILITY = ('kind', 'schema_version', 'objective', 'ruleset_id', 'action_space_version',
                  'encoder_layout_version', 'privileged_layout_version')


def _check_header(header: dict) -> None:
    expected = _header(SearchLimits(), 'none', 0, (1, 0, 0))
    for key in _COMPATIBILITY:
        if header.get(key) != expected[key]:
            raise ValueError(f'Incompatible oracle dataset {key}: {header.get(key)!r}, expected {expected[key]!r}')


def label_positions(positions, path, *, limits: SearchLimits, move_ordering: str = 'none',
                    split_seed: int = 0, fractions=(0.8, 0.1, 0.1), progress=None) -> OracleDataset:
    """Label positions and write `path`, reusing compatible cached labels.

    A cached label is reused only when the file's compatibility fields, search
    budget and split configuration all match; otherwise the call fails rather
    than mixing labels from different budgets or rules.
    """
    path = Path(path)
    header = _header(limits, move_ordering, split_seed, tuple(fractions))
    cached = {}
    if path.exists():
        old = load_dataset(path)
        if old.header['search_budget'] != header['search_budget'] or old.header['split'] != header['split']:
            raise ValueError(f'{path} uses a different search budget or split; choose another output')
        cached = {r['key']: r for r in old.records}
    records, reused = [], 0
    for index, position in enumerate(positions):
        key = position.key
        if key in cached:
            records.append(cached[key])
            reused += 1
            continue
        state = position.state()
        oracle_position = position_from_state(state)
        result = evaluate_position(oracle_position, limits=limits, move_ordering=move_ordering)
        team = position.seat % 2
        bidders = state.contract.bidder % 2
        # Bidder-team bounds follow from the zero-sum 42-point total.
        low, high = (result.lower, result.upper) if team == bidders else (42 - result.upper, 42 - result.lower)
        target = state.contract.bid.target_points
        records.append(dict(
            key=key, position=asdict(position), split=deal_split(position.deal_seed, split_seed, fractions),
            reference_team=team, lower=result.lower, upper=result.upper, exact=result.exact,
            proof_status='exact' if result.exact else 'bounded',
            contract_target=target,
            contract_status=('proven_makeable' if low >= target else 'proven_unmakeable' if high < target
                             else 'unresolved'),
            terminations=sorted({r.termination for r in result.actions}),
            nodes=result.nodes, elapsed_seconds=result.elapsed_seconds))
        if progress is not None:
            progress(index + 1, records[-1])
    dataset = OracleDataset(header, records)
    _validate_records(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(dict(header=header, records=records), indent=1) + '\n')
    temporary.replace(path)
    dataset.reused = reused
    return dataset


def _validate_records(dataset: OracleDataset) -> None:
    splits = {}
    for r in dataset.records:
        if not 0 <= r['lower'] <= r['upper'] <= 42 or r['exact'] != (r['lower'] == r['upper']):
            raise ValueError('Oracle label bounds are invalid')
        if r['reference_team'] != r['position']['seat'] % 2 or r['split'] not in SPLITS:
            raise ValueError('Oracle label perspective or split is invalid')
        if splits.setdefault(r['position']['deal_seed'], r['split']) != r['split']:
            raise ValueError('Positions from one deal cross dataset splits')


def load_dataset(path, *, sha256: str | None = None) -> OracleDataset:
    path = Path(path)
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f'Oracle dataset identity changed: {path}')
    content = json.loads(data)
    _check_header(content.get('header', {}))
    dataset = OracleDataset(content['header'], content['records'])
    _validate_records(dataset)
    return dataset


def dataset_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def position_from_record(record: dict) -> SampledPosition:
    p = dict(record['position'])
    p['initial_hands'] = tuple(tuple(h) for h in p['initial_hands'])
    p['actions'] = tuple(p['actions'])
    return SampledPosition(**p)


def training_tensors(dataset: OracleDataset, split: str, *, required: bool = True):
    """Privileged features and bounds / 42 for one split, or None if empty and optional."""
    import torch
    from .critic import ORACLE_POINT_SCALE
    rows = [r for r in dataset.records if r['split'] == split]
    if not rows:
        if required:
            raise ValueError(f'Oracle dataset has no {split} positions')
        return None
    features = []
    for r in rows:
        state = position_from_record(r).state()
        observation = observation_from_view(state.view())
        hands = tuple(tuple(d.id for d in hand) for hand in state.hands)
        features.append(encode_privileged(observation, hands))
    return dict(features=torch.tensor(features, dtype=torch.float32),
                lower=torch.tensor([r['lower'] for r in rows], dtype=torch.float32) / ORACLE_POINT_SCALE,
                upper=torch.tensor([r['upper'] for r in rows], dtype=torch.float32) / ORACLE_POINT_SCALE)
