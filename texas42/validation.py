"""Validate one complete, ordered schema-5 episode without changing its records.

Episode mode is an explicit argument: schema 5 does not persist hand versus
match reward semantics. Without privileged deals, completed public plays can
supply a candidate deal for consistency checks, but cannot verify original
ownership against a separately captured deal. Such results are always partial.
"""
from dataclasses import asdict, dataclass, is_dataclass, replace
from itertools import groupby
import json
from typing import Iterable, Literal

from .actions import ACTION_REGISTRY
from .episodes import EpisodeState
from .history import public_history
from .models import Domino, double_six_set
from .phases import Phase
from .serialization import observation_from_record
from .training import TrainingPlayView, training_view


class ValidationError(ValueError):
    """An incompatible, incomplete, or inconsistent episode record."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: Literal['verified_replay', 'partial']
    records: int
    hands: int
    limitations: tuple[str, ...] = ()


def _equal(actual, expected, field):
    # JSON normalizes tuples/lists and StrEnums but keeps bool distinct from int.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValidationError(f'{field} disagrees with replay')


def _normalized_view(view):
    # Collection preserves ownership but intentionally discards private draw
    # order. All chronological fields remain ordered and must match exactly.
    changes = {'hand': tuple(sorted(view.hand))}
    if isinstance(view, TrainingPlayView):
        changes['legal_plays'] = tuple(sorted(view.legal_plays))
    return asdict(replace(view, **changes))


def _deal_ids(deal):
    if (len(deal) != 4 or any(len(hand) != 7 for hand in deal)
            or any(type(i) is not int or not 0 <= i < 28 for hand in deal for i in hand)
            or len({i for hand in deal for i in hand}) != 28):
        raise ValidationError('initial deal must contain four disjoint seven-domino hands')
    return tuple(tuple(sorted(hand)) for hand in deal)


def validate_episode(records: Iterable[dict], *, episode: Literal['hand', 'match']) -> ValidationResult:
    """Validate a complete episode, from reset through its terminal transition.

    Accepts JSON-loaded dictionaries or collector dictionaries. Requires ordered
    records starting at hand 1, with zero initial marks, and a single episode ID
    when IDs are present (CLI logs omit them). Empty/truncated episodes fail.
    Every supplied privileged deal is checked, including repeated copies.
    Observation hand ordering is immaterial; public chronology is exact.

    A partial result checks public chronology, all visible hands, legality and
    transition labels against a deal reconstructed from completed public plays.
    It does not attest to a separately recorded initial deal or RNG provenance.
    Even verified replay verifies consistency, not dataset authenticity.
    """
    if episode not in ('hand', 'match'):
        raise ValidationError("episode must be 'hand' or 'match'")
    parsed = []
    index = 0
    try:
        for index, record in enumerate(records, 1):
            obs = observation_from_record(record)
            if type(record['hand']) is not int or record['hand'] < 1:
                raise ValidationError('hand must be a positive integer')
            if 'episode' in record and (type(record['episode']) is not int or record['episode'] < 1):
                raise ValidationError('episode ID must be a positive integer')
            entry = ACTION_REGISTRY[record['action_id']]
            if not entry.implemented or entry.phase != obs.phase:
                raise ValidationError('action ID is reserved or belongs to another phase')
            _equal(record['action_key'], entry.key, 'action_key')
            action = entry.value
            decoded = action.id if isinstance(action, Domino) else asdict(action) if is_dataclass(action) else action
            _equal(record['action'], decoded, 'decoded action')
            parsed.append((index, record, obs))
        if not parsed:
            raise ValidationError('episode is empty')
        first = parsed[0][1]
        first_obs = parsed[0][2]
        session = EpisodeState(episode, first_obs.view.target_marks, first['shaker'], first_obs.ruleset_id)
        verified = all('privileged' in record for _, record, _ in parsed)
        hands = 0
        for hand_number, grouped in groupby(parsed, key=lambda row: row[1]['hand']):
            group = list(grouped)
            index = group[0][0]
            hands += 1
            if hand_number != hands or not session.needs_deal:
                raise ValidationError('hand boundary is missing, repeated, or out of order')
            supplied = [record['privileged']['initial_deal'] for _, record, _ in group
                        if 'privileged' in record]
            if supplied:
                initial = _deal_ids(supplied[0])
            else:
                # At hand end all 28 played identities and their owners are
                # public. Use them only as a consistency hypothesis, not proof
                # of the original private deal. No hidden labels are invented.
                inferred = [[] for _ in range(4)]
                for _, record, obs in group:
                    if obs.phase == Phase.PLAY:
                        inferred[obs.seat].append(record['action_id'])
                initial = _deal_ids(inferred)
            session.deal([[double_six_set()[i] for i in hand] for hand in initial])
            for index, record, obs in group:
                if session.terminated or session.needs_deal:
                    raise ValidationError('record follows a completed hand or episode')
                _equal(record.get('episode'), first.get('episode'), 'episode ID')
                _equal(record['shaker'], session.shaker, 'shaker')
                _equal(obs.ruleset_id, session.ruleset_id, 'ruleset')
                if 'privileged' in record:
                    _equal(_deal_ids(record['privileged']['initial_deal']), initial, 'initial deal')
                state = session.hand
                expected_view = training_view(state.view())
                _equal(obs.seat, state.seat, 'actor')
                _equal(obs.phase, state.phase, 'phase')
                _equal(_normalized_view(obs.view), _normalized_view(expected_view), 'observation')
                legal = state.legal_action_ids()
                _equal(obs.legal_action_ids, legal, 'legal action IDs')
                _equal(obs.action_mask, ACTION_REGISTRY.mask(legal), 'action mask')
                _equal(asdict(public_history(obs.view)), asdict(public_history(expected_view)), 'public history')
                # EpisodeState delegates action legality and all game rules to
                # HandState, then computes marks, terminal status and rewards.
                transition = session.apply(record['action_id'])
                _equal(record['hand_result'], asdict(transition.hand_result) if transition.hand_result else None,
                       'hand result')
                _equal(record['match_marks'], session.marks, 'match marks')
                _equal(record['rewards'], transition.rewards, 'rewards')
                _equal(record['terminated'], transition.terminated, 'termination')
                _equal(record['match_winner'], transition.match_winner, 'match winner')
        if not session.terminated:
            raise ValidationError('episode is truncated before termination')
        return ValidationResult('verified_replay' if verified else 'partial', len(parsed), hands,
                                () if verified else ('No separately recorded initial deal for every record; '
                                                     'only internal consistency is established.',))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError) as error:
        raise ValidationError(f'record {index}: {error}') from error
