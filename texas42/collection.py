"""Opt-in transition collection, including separately stored teacher labels.

Pass only Observation to an ordinary policy. The collector belongs to the
rollout runner, never to the policy; Python object introspection is not a
security boundary. Sinks receive detached JSON-compatible dictionaries.
"""
from dataclasses import asdict, is_dataclass
from typing import Callable

from .actions import ACTION_REGISTRY
from .features import ENCODER_LAYOUT_VERSION, FEATURE_SCHEMA_VERSION
from .history import public_history
from .models import Domino

PRIVILEGED_DATA_VERSION = 1


class TrainingCollector:
    """Stream schema-5 records to a sink, optionally labeling the initial deal.

    `include_initial_deal=True` explicitly enables privileged data. Each record
    repeats its hand's immutable initial deal, so shuffled records retain their
    labels. The sink controls retention or JSONL storage; nothing is buffered.
    Use one collector per environment. A new reset starts a numbered episode.
    """

    def __init__(self, sink: Callable[[dict], None], *, include_initial_deal: bool = False):
        if not callable(sink):
            raise TypeError("Collector sink must be callable")
        if type(include_initial_deal) is not bool:
            raise ValueError("include_initial_deal must be an explicit boolean")
        self.sink = sink
        self.include_initial_deal = include_initial_deal
        self._episode = 0
        self._hand = 0
        self._initial_deal = None

    def begin_episode(self):
        self._episode += 1
        self._hand = 0
        self._initial_deal = None

    def begin_hand(self, shaker: int, hands=None):
        self._hand += 1
        self._shaker = shaker
        # Copy before play mutates hands. Order within a hand has no meaning.
        self._initial_deal = (tuple(tuple(sorted(d.id for d in hand)) for hand in hands)
                              if self.include_initial_deal else None)

    def record(self, observation, action_id, transition, match_marks):
        action = ACTION_REGISTRY[action_id].value
        record = {
            "schema_version": 5,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "encoder_layout_version": ENCODER_LAYOUT_VERSION,
            "ruleset_id": observation.ruleset_id,
            "episode": self._episode,
            "hand": self._hand,
            "shaker": self._shaker,
            "observation": asdict(observation),
            "public_history": asdict(public_history(observation.view)),
            "action_id": action_id,
            "action_key": ACTION_REGISTRY[action_id].key,
            "action": action.id if isinstance(action, Domino) else
                      asdict(action) if is_dataclass(action) else action,
            "hand_result": asdict(transition.hand_result) if transition.hand_result else None,
            "match_marks": tuple(match_marks),
            "rewards": transition.rewards,
            "terminated": transition.terminated,
            "match_winner": transition.match_winner,
        }
        if self.include_initial_deal:
            record["privileged"] = {
                "version": PRIVILEGED_DATA_VERSION,
                "initial_deal": self._initial_deal,
            }
        self.sink(record)
