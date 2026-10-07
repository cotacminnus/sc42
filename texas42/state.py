"""Shared hand transitions for the CLI and self-play environment."""

from __future__ import annotations

from dataclasses import dataclass
from .actions import ACTION_REGISTRY, BID_ACTIONS, PLAY_ACTIONS, TRUMP_ACTIONS, Phase
from .models import Bid, Domino, Trump
from .players import BidView, PlayView, TrumpView
from .rules import is_legal_bid, legal_plays, trick_points, trick_winner


# Re-export decoded tables for existing imports; permanent IDs live in actions.
DecisionView = BidView | TrumpView | PlayView


@dataclass(frozen=True)
class HandResult:
    """Seats are 0-3; team totals are ordered by seat parity."""

    bidder: int
    bid: Bid
    trump: Trump
    team_points: tuple[int, int]
    scoring_team: int
    marks_awarded: int


class HandState:
    """Internal mutable state. Only decision views belong in policy inputs."""

    def __init__(self, hands: list[list[Domino]], shaker: int,
                 marks: tuple[int, int], target_marks: int) -> None:
        self.hands = [list(hand) for hand in hands]
        self.shaker = shaker
        self.marks = marks
        self.target_marks = target_marks
        self.phase: Phase = "bid"
        self.seat = (shaker + 1) % 4
        self.bid_history: list[tuple[int, Bid | None]] = []
        self.bid: Bid | None = None
        self.bidder: int | None = None
        self.trump: Trump = None
        self.trick: list[tuple[int, Domino]] = []
        self.completed: list[tuple[tuple[int, Domino], ...]] = []
        self.points = [0, 0]
        self.result: HandResult | None = None

    @property
    def forced(self) -> bool:
        return self.phase == "bid" and len(self.bid_history) == 3 and self.bid is None

    def view(self) -> DecisionView:
        """Copy public history and the actor's hand into immutable snapshots."""
        if self.result is not None:
            raise RuntimeError("The hand is complete")
        common = dict(seat=self.seat, hand=tuple(self.hands[self.seat]),
                      match_marks=self.marks, bid_history=tuple(self.bid_history),
                      target_marks=self.target_marks)
        if self.phase == "bid":
            return BidView(**common, shaker=self.shaker, current_bid=self.bid,
                           current_bidder=self.bidder, forced=self.forced)
        assert self.bid is not None and self.bidder is not None
        if self.phase == "trump":
            return TrumpView(**common, bid=self.bid)
        return PlayView(**common, legal_plays=self.legal_dominoes(),
                        trick=tuple(self.trick), completed_tricks=tuple(self.completed),
                        bidder=self.bidder, bid=self.bid, trump=self.trump,
                        team_points=tuple(self.points))

    def legal_dominoes(self) -> tuple[Domino, ...]:
        return legal_plays(self.hands[self.seat],
                           self.trick[0][1] if self.trick else None, self.trump)

    def actions(self) -> tuple:
        """Decoded values in global-ID order; use the registry to encode values."""
        return tuple(entry.value for entry in ACTION_REGISTRY.entries)

    def legal_action_ids(self) -> tuple[int, ...]:
        """Return sorted global IDs, with future declarations masked out."""
        if self.result is not None:
            return ()
        if self.phase == "bid":
            return tuple(entry.action_id for entry in ACTION_REGISTRY.for_phase("bid")
                         if (not self.forced if entry.value is None
                             else is_legal_bid(entry.value, self.bid)))
        if self.phase == "trump":
            return tuple(entry.action_id for entry in ACTION_REGISTRY.for_phase("trump"))
        return tuple(sorted(domino.id for domino in self.legal_dominoes()))

    def apply(self, action_id: int) -> None:
        """Validate before mutating so rejected actions leave the state intact."""
        if self.result is not None:
            raise RuntimeError("The hand is complete")
        if type(action_id) is not int or action_id not in self.legal_action_ids():
            raise ValueError(f"Illegal {self.phase} action ID: {action_id!r}")
        action = ACTION_REGISTRY[action_id].value
        if self.phase == "bid":
            self.bid_history.append((self.seat, action))
            if action is not None:
                self.bid, self.bidder = action, self.seat
            if len(self.bid_history) == 4:
                # The shaker bids last and cannot pass when all others pass,
                # so every completed auction has a bidder and a contract.
                assert self.bidder is not None
                self.phase, self.seat = "trump", self.bidder
            else:
                self.seat = (self.seat + 1) % 4
        elif self.phase == "trump":
            self.trump = action
            self.phase = "play"
        else:
            self.hands[self.seat].remove(action)
            self.trick.append((self.seat, action))
            if len(self.trick) < 4:
                self.seat = (self.seat + 1) % 4
                return
            self.seat = trick_winner(self.trick, self.trump)
            # The winning seat leads next; seat parity also identifies its team.
            self.points[self.seat % 2] += trick_points(self.trick)
            self.completed.append(tuple(self.trick))
            self.trick = []
            if len(self.completed) == 7:
                assert self.bid is not None and self.bidder is not None
                team = self.bidder % 2
                scoring_team = team if self.points[team] >= self.bid.target_points else 1 - team
                self.result = HandResult(self.bidder, self.bid, self.trump,
                                         tuple(self.points), scoring_team,
                                         self.bid.value_in_marks)
