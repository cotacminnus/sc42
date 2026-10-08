"""Lossless per-hand public decisions, separate from the summary encoder.

Events precede the observed decision. Seats are absolute for replay; relative
seats can be requested for a policy input. Chance draws and hidden hands never
appear here. Four successive play events form one trick under the base rules.
"""
from dataclasses import dataclass

from .actions import ACTION_REGISTRY
from .features import PUBLIC_HISTORY_VERSION
from .phases import Phase
from .training import TrainingBidView, TrainingPlayView, TrainingView


@dataclass(frozen=True, slots=True)
class PublicEvent:
    phase: Phase
    seat: int
    action_id: int

    def __post_init__(self):
        object.__setattr__(self, "phase", Phase(self.phase))
        if type(self.seat) is not int or self.seat not in range(4):
            raise ValueError("Public event seat must be 0-3")
        entry = ACTION_REGISTRY[self.action_id]
        if entry.phase != self.phase:
            raise ValueError("Public event action and phase disagree")


@dataclass(frozen=True, slots=True)
class PublicHistory:
    events: tuple[PublicEvent, ...]
    version: int = PUBLIC_HISTORY_VERSION

    def __post_init__(self):
        if type(self.version) is not int or self.version != PUBLIC_HISTORY_VERSION:
            raise ValueError("Unsupported public-history version")
        object.__setattr__(self, "events", tuple(self.events))
        if any(not isinstance(event, PublicEvent) for event in self.events):
            raise ValueError("Expected PublicEvent entries")

    def relative_events(self, actor: int) -> tuple[PublicEvent, ...]:
        """Rotate seats without changing action identity or chronology."""
        if type(actor) is not int or actor not in range(4):
            raise ValueError("Actor must be a seat from 0 to 3")
        return tuple(PublicEvent(e.phase, (e.seat - actor) % 4, e.action_id)
                     for e in self.events)

    @classmethod
    def from_dict(cls, data: dict) -> "PublicHistory":
        """Load the JSON form produced by dataclasses.asdict, rejecting versions."""
        try:
            return cls(tuple(PublicEvent(**event) for event in data["events"]), data["version"])
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed public history") from error


def public_history(view: TrainingView) -> PublicHistory:
    """Recover every observed public decision, including passes and trump choice.

    Completed tricks stay chronological, unlike the encoder's played-domino
    summary. No terminal observation is invented: append the final action when
    reconstructing a complete hand from transition records.
    """
    events = [PublicEvent(Phase.BID, seat, ACTION_REGISTRY.encode(Phase.BID, call))
              for seat, call in view.bid_history]
    if isinstance(view, TrainingPlayView):
        events.append(PublicEvent(Phase.TRUMP, view.bidder,
                                  ACTION_REGISTRY.encode(Phase.TRUMP, view.trump)))
        for trick in (*view.completed_tricks, view.trick):
            events.extend(PublicEvent(Phase.PLAY, seat, domino) for seat, domino in trick)
    elif not isinstance(view, TrainingBidView):
        # A trump view observes the resolved auction, but no trump decision yet.
        from .training import TrainingTrumpView
        if not isinstance(view, TrainingTrumpView):
            raise TypeError("Expected a phase-specific training view")
    return PublicHistory(tuple(events))
