from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


DOUBLE_TRUMP = "doubles"
Trump: TypeAlias = int | Literal["doubles"] | None
Suit: TypeAlias = int | Literal["doubles"]


@dataclass(frozen=True, order=True)
class Domino:
    """A normalized double-six domino, stored high end first."""

    high: int
    low: int

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 6:
            raise ValueError("domino ends must satisfy 0 <= low <= high <= 6")

    @classmethod
    def make(cls, first: int, second: int) -> Domino:
        return cls(max(first, second), min(first, second))

    @property
    def is_double(self) -> bool:
        return self.high == self.low

    @property
    def pip_total(self) -> int:
        return self.high + self.low

    @property
    def count_value(self) -> int:
        return self.pip_total if self.pip_total in (5, 10) else 0

    def contains(self, suit: int) -> bool:
        return self.high == suit or self.low == suit

    def other_end(self, suit: int) -> int:
        if not self.contains(suit):
            raise ValueError(f"{self} does not contain suit {suit}")
        return self.low if self.high == suit else self.high

    def __str__(self) -> str:
        return f"[{self.high}|{self.low}]"


def double_six_set() -> tuple[Domino, ...]:
    return tuple(Domino(high, low) for high in range(7) for low in range(high + 1))


@dataclass(frozen=True)
class Bid:
    """A point bid (30-41) or a mark bid (1-5)."""

    points: int | None = None
    marks: int | None = None

    def __post_init__(self) -> None:
        if (self.points is None) == (self.marks is None):
            raise ValueError("a bid must specify exactly one of points or marks")
        if self.points is not None and not 30 <= self.points <= 41:
            raise ValueError("point bids must be between 30 and 41")
        if self.marks is not None and not 1 <= self.marks <= 5:
            raise ValueError("mark bids must be between 1 and 5")

    @property
    def strength(self) -> int:
        if self.points is not None:
            return self.points
        assert self.marks is not None
        return 42 + self.marks - 1

    @property
    def target_points(self) -> int:
        return self.points if self.points is not None else 42

    @property
    def value_in_marks(self) -> int:
        return self.marks or 1

    def __str__(self) -> str:
        if self.points is not None:
            return str(self.points)
        assert self.marks is not None
        suffix = "mark" if self.marks == 1 else "marks"
        return f"{self.marks} {suffix}"

    @classmethod
    def parse(cls, text: str) -> Bid:
        value = text.strip().lower().replace(" ", "")
        if value == "42":
            return cls(marks=1)
        if value.endswith("marks"):
            value = value[:-5]
        elif value.endswith("mark"):
            value = value[:-4]
        elif value.endswith("m"):
            value = value[:-1]
        number = int(value)
        if 30 <= number <= 41:
            return cls(points=number)
        if 1 <= number <= 5:
            return cls(marks=number)
        raise ValueError("enter 30-41, 42, or 1m-5m")


def trump_name(trump: Trump) -> str:
    if trump is None:
        return "Follow Me (no trump)"
    if trump == DOUBLE_TRUMP:
        return "doubles"
    names = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes")
    return names[trump]

