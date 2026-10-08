from texas42.actions import ACTION_REGISTRY, PLAY_ACTIONS
from texas42.models import Bid, Domino
from texas42.phases import Phase
from texas42.rules import (
    is_trump,
    led_suit,
    suit_rank,
    trick_winner,
)


class HeuristicAgent:
    """Simple rule-based Texas 42 agent (V1)."""

    TRUMPS = (*range(7), "doubles")

    def __init__(self):
        self.seat = None

    def reset(self, seat: int) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError("Invalid seat")
        self.seat = seat

    def act(self, obs) -> int:
        if obs.seat != self.seat:
            raise RuntimeError("Agent assigned to another seat")

        if obs.phase == Phase.BID:
            action = self._bid(obs)
        elif obs.phase == Phase.TRUMP:
            action = self._trump(obs)
        elif obs.phase == Phase.PLAY:
            action = self._play(obs)
        else:
            raise ValueError("Unknown phase")

        if action not in obs.legal_action_ids:
            raise ValueError(f"Illegal heuristic action: {action}")

        return action

    # ---------- Helpers ----------

    def _hand(self, obs):
        return [PLAY_ACTIONS[i] for i in obs.view.hand]

    def _suit_tiles(self, hand, trump):
        return [d for d in hand if is_trump(d, trump)]

    def _best_suit(self, hand):
        def quality(trump):
            tiles = self._suit_tiles(hand, trump)
            ranks = [suit_rank(d, trump) for d in tiles]
            return (
                len(tiles),
                max(ranks, default=(0, 0)),
                sum(d.count_value for d in tiles),
            )

        return max(self.TRUMPS, key=quality)

    @staticmethod
    def _power(domino, trump):
        """Trumps are stronger; doubles rank highest in their suit."""
        if is_trump(domino, trump):
            return (1, *suit_rank(domino, trump))

        suit = led_suit(domino, trump)
        return (0, *suit_rank(domino, suit))

    def _discard_cost(self, action_id, trump):
        """Prefer losing low-value, low-strength dominoes."""
        domino = PLAY_ACTIONS[action_id]
        return (
            domino.count_value,
            self._power(domino, trump),
            domino.pip_total,
        )

    @staticmethod
    def _unbeatable(domino, trump):
        """Whether the currently winning domino is unbeatable."""
        if trump == "doubles":
            return domino == Domino(6, 6)
        if trump is None:
            return domino.is_double
        return domino == Domino(trump, trump)

    # ---------- Bidding ----------

    def _bid(self, obs):
        hand = self._hand(obs)
        best_suit = self._best_suit(hand)
        tiles = self._suit_tiles(hand, best_suit)

        length = len(tiles)
        counters = sum(d.count_value for d in tiles)

        # Deliberately simple, aggressive hand classification.
        if length == 7:
            desired = Bid(marks=5)
        elif length == 6 and counters >= 5:
            desired = Bid(marks=2)
        elif length == 5 and counters >= 5:
            desired = Bid(marks=1)
        elif length == 6:
            desired = Bid(points=39)
        elif length == 5:
            desired = Bid(points=35)
        elif length == 4:
            desired = Bid(points=30)
        else:
            desired = None

        legal_bids = [
            (action_id, ACTION_REGISTRY[action_id].value)
            for action_id in obs.legal_action_ids
            if isinstance(ACTION_REGISTRY[action_id].value, Bid)
        ]

        pass_id = ACTION_REGISTRY.encode(Phase.BID, None)

        # Weak hand: pass unless forced.
        if desired is None:
            if pass_id in obs.legal_action_ids:
                return pass_id
            return min(legal_bids, key=lambda x: x[1].strength)[0]

        affordable = [
            (action_id, bid)
            for action_id, bid in legal_bids
            if bid.strength <= desired.strength
        ]

        if affordable:
            # Open aggressively with the desired bid.
            # When raising, choose the smallest sufficient raise.
            select = (
                max if obs.view.current_bid is None else min
            )
            return select(
                affordable, key=lambda x: x[1].strength
            )[0]

        if pass_id in obs.legal_action_ids:
            return pass_id

        # Forced bidder must make a legal bid.
        return min(legal_bids, key=lambda x: x[1].strength)[0]

    # ---------- Trump ----------

    def _trump(self, obs):
        hand = self._hand(obs)

        doubles = sum(d.is_double for d in hand)
        numbered_max = max(
            len(self._suit_tiles(hand, suit))
            for suit in range(7)
        )

        # Your proposed Follow Me rule.
        if doubles >= 3 and doubles > numbered_max:
            chosen = None
        else:
            chosen = self._best_suit(hand)

        return ACTION_REGISTRY.encode(Phase.TRUMP, chosen)

    # ---------- Playing ----------

    def _play(self, obs):
        view = obs.view
        trump = view.trump
        legal = obs.legal_action_ids
        trick = [
            (seat, PLAY_ACTIONS[domino_id])
            for seat, domino_id in view.trick
        ]

        # 1. Leading: play the strongest available trump.
        # If no trump, lead the strongest available domino.
        if not trick:
            return max(
                legal,
                key=lambda i: self._power(PLAY_ACTIONS[i], trump),
            )

        winner = trick_winner(trick, trump)
        partner_winning = winner % 2 == self.seat % 2

        def winner_after(action_id):
            new_trick = trick + [
                (self.seat, PLAY_ACTIONS[action_id])
            ]
            return trick_winner(new_trick, trump)

        # 2. Partner is currently winning.
        if partner_winning:
            winning_domino = next(
                d for seat, d in trick if seat == winner
            )

            safe = (
                len(trick) == 3
                or self._unbeatable(winning_domino, trump)
            )

            if safe:
                # Give as many counter points as possible,
                # without unnecessarily wasting stronger dominoes.
                return min(
                    legal,
                    key=lambda i: (
                        -PLAY_ACTIONS[i].count_value,
                        self._power(PLAY_ACTIONS[i], trump),
                    ),
                )

            # Partner may still be overtaken.
            return min(
                legal,
                key=lambda i: self._discard_cost(i, trump),
            )

        # 3. Opponent is winning.
        winning_moves = [
            i for i in legal
            if winner_after(i) == self.seat
        ]

        counters_in_trick = sum(
            domino.count_value for _, domino in trick
        )

        if winning_moves and (
            len(trick) == 3 or counters_in_trick >= 5
        ):
            # Win using the least powerful sufficient domino.
            return min(
                winning_moves,
                key=lambda i: self._power(PLAY_ACTIONS[i], trump),
            )

        # 4. Otherwise discard the cheapest domino.
        return min(
            legal,
            key=lambda i: self._discard_cost(i, trump),
        )
