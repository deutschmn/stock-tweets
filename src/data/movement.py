from ctypes import ArgumentError
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import pandas as pd


class Direction(IntEnum):
    UP = 1  # price goes up
    DOWN = 0  # price goes down
    SAME = -1  # price stays the same


@dataclass
class Tweet:
    text: str
    followers: int
    date: pd.Timestamp


@dataclass
class Movement:
    tweets: pd.DataFrame
    stock: str
    price: pd.DataFrame
    day: pd.Timestamp

    def __post_init__(self):
        self.price_movement: float = self.price["movement percent"]

        self.model_input: List[Tweet] = [
            Tweet(text, followers, date)
            for (text, followers, date) in zip(
                self.tweets["text"], self.tweets["user_followers"], self.tweets["date"]
            )
        ]


@dataclass
class ClassifiedMovement(Movement):
    direction: Direction

    def __post_init__(self):
        super().__post_init__()

        if self.direction == Direction.SAME:
            raise ArgumentError("Movement mustn't have direction SAME")

        self.model_output = ModelOutput(self.direction, self.price_movement)


@dataclass
class ModelOutput:
    """Stores the labels for a sample (direction for classification, price_movement for regression)"""

    direction: Direction
    price_movement: float
