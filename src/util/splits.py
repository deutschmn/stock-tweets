from dataclasses import dataclass

import pandas as pd


class DateRange:
    def __init__(self, start: str, end: str) -> None:
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)


@dataclass
class DataSplit:
    train: DateRange
    val: DateRange
    test: DateRange
