from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import Bid, Domino, double_six_set, trump_name
from .players import BidView, Player, TrumpView, format_hand
from .rules import trick_points, trick_winner
from .state import HandResult, HandState


@dataclass(frozen=True)
class GameConfig:
    target_marks: int = 7
    debug: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.target_marks < 1:
            raise ValueError("target_marks must be positive")


class Game:
    def __init__(
        self,
        players: Sequence[Player],
        config: GameConfig | None = None,
        write: Callable[[str], None] = print,
    ) -> None:
        if len(players) != 4:
            raise ValueError("Texas 42 requires exactly four players")
        self.players = tuple(players)
        self.config = config or GameConfig()
        self.write = write
        self.rng = random.Random(self.config.seed)
        self.shaker = self.rng.randrange(4)
        self.marks = [0, 0]

    @staticmethod
    def team_for(seat: int) -> int:
        # Opposite seats partner: (0, 2) are team 0 and (1, 3) are team 1.
        return seat % 2

    def play_match(self) -> int:
        """Play to the configured mark target, rotating the shaker after each hand."""
        self.write("Texas 42 - you and Player 3 versus Players 2 and 4")
        self.write(f"{self.players[self.shaker].name} shakes first.")
        while max(self.marks) < self.config.target_marks:
            self.play_hand()
            self.shaker = (self.shaker + 1) % 4
        winner = 0 if self.marks[0] >= self.config.target_marks else 1
        names = "Players 1 & 3" if winner == 0 else "Players 2 & 4"
        self.write(f"{names} win the match, {self.marks[0]}-{self.marks[1]}!")
        return winner

    def play_hand(self) -> HandResult:
        """Play seven tricks and update match marks without rotating the shaker."""
        hands = self._deal()
        self.write(f"\n--- New hand: {self.players[self.shaker].name} shakes ---")
        if self.config.debug:
            self.write("DEBUG - all hands:")
            for seat, hand in enumerate(hands):
                self.write(f"  P{seat + 1} {self.players[seat].name}: {format_hand(hand)}")

        state = HandState(hands, self.shaker, tuple(self.marks), self.config.target_marks)
        while state.result is None:
            view = state.view()
            player = self.players[view.seat]
            if isinstance(view, BidView):
                action = player.choose_bid(view)
                if action is None and view.forced:
                    action = Bid(points=30)
                    message = f"{player.name} is forced to bid 30."
                elif action is None:
                    message = f"{player.name} passes."
                else:
                    message = f"{player.name} bids {action}."
            elif isinstance(view, TrumpView):
                action = player.choose_trump(view)
                message = ""  # Format the trump name after validating the action.
            else:
                action = player.choose_domino(view)
                message = f"P{view.seat + 1} {player.name} plays {action}"
            # Match types as well as values: bool/float are not numbered trumps.
            action_id = next((i for i, candidate in enumerate(state.actions())
                              if type(action) is type(candidate) and action == candidate), None)
            if action_id is None:
                raise ValueError(f"{player.name} returned an invalid {state.phase} action: {action!r}")
            completed_before = len(state.completed)
            state.apply(action_id)
            if isinstance(view, TrumpView):
                message = (
                    f"{player.name} wins the bid at {view.bid} "
                    f"and calls {trump_name(action)}."
                )
            self.write(message)
            if len(state.completed) != completed_before:
                trick = state.completed[-1]
                winner = trick_winner(trick, state.trump)
                points = trick_points(trick)
                self.write(
                    f"Trick {len(state.completed)}: {self.players[winner].name} wins "
                    f"{points} point{'s' if points != 1 else ''}."
                )

        result = state.result
        self.marks[result.scoring_team] += result.marks_awarded
        outcome = "made" if result.scoring_team == result.bidder % 2 else "was set on"
        self.write(
            f"The bidding team {outcome} {result.bid}; hand points "
            f"{result.team_points[0]}-{result.team_points[1]}."
        )
        self.write(f"Match marks: {self.marks[0]}-{self.marks[1]}")
        return result

    def _deal(self) -> list[list[Domino]]:
        deck = list(double_six_set())
        self.rng.shuffle(deck)
        return [deck[seat * 7 : (seat + 1) * 7] for seat in range(4)]
