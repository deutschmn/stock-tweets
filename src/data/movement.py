from dataclasses import dataclass

import pandas as pd


@dataclass
class Movement:
    tweets: pd.DataFrame
    stock: str
    price: float
    day: pd.Timestamp
