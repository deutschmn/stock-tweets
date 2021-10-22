from dataclasses import dataclass
from typing import List

import pandas as pd


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
        self.model_input: List[Tweet] = [
            Tweet(text, followers, date)
            for (text, followers, date) in zip(
                self.tweets["text"], self.tweets["user_followers"], self.tweets["date"]
            )
        ]

        self.price_movement: pd.Series = self.price["movement percent"]
