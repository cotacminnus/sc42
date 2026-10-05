from __future__ import annotations

from collections.abc import Sequence

from .models import DOUBLE_TRUMP, Bid, Domino, Suit, Trump


def is_legal_bid(candidate: Bid, current: Bid | None) -> bool:
    """Validate bid ordering and the common maximum mark jumps."""
    if current is not None and candidate.strength <= current.strength:
        return False

    if current is None or current.points is not None:
        # The first bid expressed in marks may be at most two marks.
        return candidate.marks is None or candidate.marks <= 2

    assert current.marks is not None
    return candidate.marks is not None and candidate.marks <= current.marks + 1


def is_trump(domino: Domino, trump: Trump) -> bool:
    if trump is None:
        return False
    if trump == DOUBLE_TRUMP:
        return domino.is_double
    return domino.contains(trump)


def led_suit(domino: Domino, trump: Trump) -> Suit:
    """A trump lead calls trump; any other lead calls its higher numbered end."""
    if is_trump(domino, trump):
        assert trump is not None
        return trump
    return domino.high


def follows_suit(domino: Domino, suit: Suit, trump: Trump) -> bool:
    """Trump dominoes belong exclusively to trump when deciding who must follow."""
    if suit == DOUBLE_TRUMP:
        return domino.is_double
    if trump == suit:
        return is_trump(domino, trump)
    return not is_trump(domino, trump) and domino.contains(suit)


def legal_plays(hand: Sequence[Domino], lead: Domino | None, trump: Trump) -> tuple[Domino, ...]:
    """Require following suit when possible; otherwise allow any domino."""
    if lead is None:
        return tuple(hand)
    suit = led_suit(lead, trump)
    following = tuple(domino for domino in hand if follows_suit(domino, suit, trump))
    return following or tuple(hand)


def suit_rank(domino: Domino, suit: Suit) -> tuple[int, int]:
    """Return an orderable rank within a suit; doubles are high."""
    if suit == DOUBLE_TRUMP:
        if not domino.is_double:
            raise ValueError("only doubles can be ranked in the doubles suit")
        return (1, domino.high)
    if not domino.contains(suit):
        raise ValueError(f"{domino} is not in suit {suit}")
    # Compare the double flag first, then the other end for non-doubles.
    return (1 if domino.is_double else 0, domino.other_end(suit))


def trick_winner(plays: Sequence[tuple[int, Domino]], trump: Trump) -> int:
    """Return the winning seat from plays in lead order, preferring any trump."""
    if not plays:
        raise ValueError("cannot score an empty trick")
    lead = plays[0][1]
    suit = led_suit(lead, trump)

    trump_plays = [(seat, domino) for seat, domino in plays if is_trump(domino, trump)]
    if trump_plays:
        assert trump is not None
        candidates = trump_plays
        winning_suit: Suit = trump
    else:
        candidates = [
            (seat, domino)
            for seat, domino in plays
            if follows_suit(domino, suit, trump)
        ]
        winning_suit = suit

    return max(candidates, key=lambda item: suit_rank(item[1], winning_suit))[0]


def trick_points(plays: Sequence[tuple[int, Domino]]) -> int:
    # Seven trick points plus 35 counter points make 42 points in a full hand.
    return 1 + sum(domino.count_value for _, domino in plays)

