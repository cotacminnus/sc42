"""Immutable policy inputs with stable integer domino IDs."""

from dataclasses import dataclass

from .models import Bid, Trump
from .players import BidView, PlayView, TrumpView


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Private actor snapshot; hand entries are permanent IDs, not positions."""

    seat: int
    hand: tuple[int, ...]
    match_marks: tuple[int, int]
    bid_history: tuple[tuple[int, Bid | None], ...]
    target_marks: int


@dataclass(frozen=True, slots=True)
class TrainingBidView(TrainingView):
    shaker: int
    current_bid: Bid | None
    current_bidder: int | None
    forced: bool


@dataclass(frozen=True, slots=True)
class TrainingTrumpView(TrainingView):
    bid: Bid


@dataclass(frozen=True, slots=True)
class TrainingPlayView(TrainingView):
    """Tricks retain chronological (seat, domino_id) pairs for policy recall."""

    legal_plays: tuple[int, ...]
    trick: tuple[tuple[int, int], ...]
    completed_tricks: tuple[tuple[tuple[int, int], ...], ...]
    bidder: int
    bid: Bid
    trump: Trump
    team_points: tuple[int, int]


def training_view(view: BidView | TrumpView | PlayView) -> TrainingView:
    """Encode an existing private view without exposing simulator hands."""
    common = dict(seat=view.seat, hand=tuple(domino.id for domino in view.hand),
                  match_marks=view.match_marks, bid_history=view.bid_history,
                  target_marks=view.target_marks)
    if isinstance(view, BidView):
        return TrainingBidView(**common, shaker=view.shaker, current_bid=view.current_bid,
                               current_bidder=view.current_bidder, forced=view.forced)
    if isinstance(view, TrumpView):
        return TrainingTrumpView(**common, bid=view.bid)
    return TrainingPlayView(
        **common, legal_plays=tuple(domino.id for domino in view.legal_plays),
        trick=tuple((seat, domino.id) for seat, domino in view.trick),
        completed_tricks=tuple(tuple((seat, domino.id) for seat, domino in trick)
                               for trick in view.completed_tricks),
        bidder=view.bidder, bid=view.bid, trump=view.trump, team_points=view.team_points,
    )
