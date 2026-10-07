"""Interactive/console match runner: deals, asks Player objects, prints, and logs.

Uses the same HandState transitions as SelfPlayEnv, so CLI play and training
episodes cannot drift apart.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from .features import FEATURE_SCHEMA_VERSION
from .actions import ACTION_REGISTRY

from .models import Bid, Domino, double_six_set, trump_name
from .players import BidView, Player, TrumpView, format_hand
from .rules import trick_points, trick_winner
from .state import HandResult, HandState
from .selfplay import Observation
from .training import training_view
from .rulesets import RULESET_ID, validate_ruleset


@dataclass(frozen=True)
class GameConfig:
    target_marks: int = 7
    debug: bool = False
    seed: int | None = None
    ruleset_id: str = RULESET_ID

    def __post_init__(self) -> None:
        validate_ruleset(self.ruleset_id)
        if type(self.target_marks) is not int or self.target_marks < 1:
            raise ValueError("target_marks must be a positive integer")


class Game:
    def __init__(
        self,
        players: Sequence[Player],
        config: GameConfig | None = None,
        write: Callable[[str], None] = print,
        log_state: Callable[[dict], None] | None = None,
    ) -> None:
        if len(players) != 4:
            raise ValueError("Texas 42 requires exactly four players")
        self.players = tuple(players)
        self.config = config or GameConfig()
        self.write = write
        self.log_state = log_state
        self._hand_number = 0
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
        self._hand_number += 1
        self.write(f"\n--- New hand: {self.players[self.shaker].name} shakes ---")
        if self.config.debug:
            self.write("DEBUG - all hands:")
            for seat, hand in enumerate(hands):
                self.write(f"  P{seat + 1} {self.players[seat].name}: {format_hand(hand)}")

        state = HandState(hands, self.shaker, tuple(self.marks), self.config.target_marks,
                          self.config.ruleset_id)
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
            try:
                action_id = ACTION_REGISTRY.encode(state.phase, action)
            except ValueError as error:
                raise ValueError(f"{player.name} returned an invalid {state.phase} action: {action!r}") from error
            completed_before = len(state.completed)
            if self.log_state is not None:
                # Capture the actor's private snapshot BEFORE applying its move;
                # the rest of this record describes the resulting transition.
                legal = state.legal_action_ids()
                observation = Observation(state.seat, state.phase, training_view(view), legal,
                                          ACTION_REGISTRY.mask(legal), ruleset_id=self.config.ruleset_id)
            state.apply(action_id)
            if self.log_state is not None:
                # Marks are committed once below, after the hand finishes. Use a
                # projected copy so the terminal record includes that award.
                marks = list(self.marks)
                result = state.result
                if result is not None:
                    marks[result.scoring_team] += result.marks_awarded
                terminated = result is not None and max(marks) >= self.config.target_marks
                winner = result.scoring_team if terminated else None
                # Records hold only the actor's view. Belief-model labels would
                # need the full deal in a separate, never-observed field.
                self.log_state({
                    "schema_version": 5,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "ruleset_id": self.config.ruleset_id,
                    "hand": self._hand_number,
                    "shaker": self.shaker,
                    "observation": asdict(observation),
                    "action_id": action_id,
                    "action_key": ACTION_REGISTRY[action_id].key,
                    "action": action.id if isinstance(action, Domino) else
                              asdict(action) if is_dataclass(action) else action,
                    "hand_result": asdict(result) if result is not None else None,
                    "match_marks": marks,
                    "terminated": terminated,
                    "match_winner": winner,
                    "rewards": [1 if seat % 2 == winner else -1 for seat in range(4)]
                               if terminated else [0, 0, 0, 0],
                })
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
