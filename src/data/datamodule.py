from typing import List, Optional
import json
import os
import itertools
import pickle

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning.core.datamodule import LightningDataModule

from src.data.dataset import MovementDataset
from src.data.movement import Movement
from src.util.splits import DataSplit, DateRange


class MovementDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_split: DataSplit,
        classify_threshold_up: float,
        classify_threshold_down: float,
        tweet_path: str,
        price_path: str,
        min_followers: int,
        min_tweets_day: int,
        time_lag: int,
        num_workers: int = 0,
    ):
        """Inits the DataModule

        Args:
            batch_size (int): Batch size to use for train/val/test loading
            data_split (DataSplit): How the data is split
            classify_threshold_up (float): Threshold for when a course is classified as going UP
            classify_threshold_down (float): Threshold for when a course is classified as going DOWN
            min_followers (int): Minimum number of followers to consider a tweet from this author
            min_tweets_day (int): Minimum number of tweets about a stock on one day for a movement to be considered as a sample
            time_lag (int): Number of days between tweets and supposed market reaction
            tweet_path (str): Path to tweet data.
            price_path (str): Path to price data.
            num_workers (int): Number of workers for the DataLoaders
        """
        super().__init__()

        self.batch_size = batch_size
        self.data_split = data_split
        self.classify_threshold_up = classify_threshold_up
        self.classify_threshold_down = classify_threshold_down
        self.tweet_path = tweet_path
        self.price_path = price_path
        self.min_followers = min_followers
        self.min_tweets_day = min_tweets_day
        self.time_lag = time_lag
        self.num_workers = num_workers

        self.movements: List[Movement] = []
        self.train_ds: Optional[MovementDataset] = None
        self.val_ds: Optional[MovementDataset] = None
        self.test_ds: Optional[MovementDataset] = None

    def _load_tweets(self) -> pd.DataFrame:
        dfs = []

        for symbol in tqdm(os.listdir(self.tweet_path), desc="Tweets"):
            for day in os.listdir(os.path.join(self.tweet_path, symbol)):
                file_path = os.path.join(self.tweet_path, symbol, day)

                with open(file_path) as f:
                    content = f.readlines()
                    d = list(map(json.loads, content))
                    df = pd.DataFrame(d)

                    if self.min_followers > 0:
                        df = df[
                            df["user"].apply(lambda u: u["followers_count"])
                            > self.min_followers
                        ]

                    if len(df) > 0:
                        dfs.append(df)

        tweets = pd.concat(dfs)
        print(f"Found {len(tweets)} tweets")

        simple_tweets = pd.DataFrame()
        simple_tweets["date"] = pd.to_datetime(tweets["created_at"])
        simple_tweets["text"] = tweets["text"].apply(lambda t: t.replace("\n", " "))
        simple_tweets["user_name"] = tweets["user"].apply(lambda u: u["name"])
        simple_tweets["user_followers"] = tweets["user"].apply(
            lambda u: u["followers_count"]
        )
        simple_tweets["sym"] = tweets["entities"].apply(
            lambda entities: list(map(lambda s: s["text"], entities["symbols"]))
        )

        return simple_tweets

    def _load_prices(self) -> pd.DataFrame:
        dfs = []
        cols = [
            "date",
            "movement percent",
            "norm open price",
            "norm high price",
            "norm low price",
            "norm close price",
            "volume",
        ]

        for symbol in tqdm(os.listdir(self.price_path), desc="Prices"):
            file_path = os.path.join(self.price_path, symbol)

            df = pd.read_csv(file_path, sep="\t", names=cols)
            df["symbol"] = symbol.replace(".txt", "")
            dfs.append(df)

        prices = pd.concat(dfs, axis=0)
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.set_index(["symbol", "date"])
        return prices

    def _load_movements_from_files(self) -> List[Movement]:
        tweets = self._load_tweets()
        prices = self._load_prices()
        movements = []
        missing_prices = []

        for day in tqdm(prices.index.get_level_values("date").unique()):
            day = pd.to_datetime(day) - pd.to_timedelta(self.time_lag, unit="days")
            relevant_tweets = tweets[tweets["date"].dt.date == day]
            relevant_stocks = pd.unique(list(itertools.chain(*relevant_tweets["sym"])))
            for stock in relevant_stocks:
                rel_tweets = relevant_tweets[
                    relevant_tweets["sym"].apply(lambda x: stock in x)
                ]
                try:
                    movements.append(
                        Movement(rel_tweets, stock, prices.loc[stock, day], day)
                    )
                except Exception as e:
                    missing_prices.append(e)

        nr_movements = len(movements)

        if self.min_tweets_day > 0:
            movements = list(
                filter(lambda m: len(m.tweets) > self.min_tweets_day, movements)
            )

        print(
            f"Loaded {nr_movements} movements, returning {len(movements)}. Found no price for {len(missing_prices)}."
        )

        return movements

    def _load_movements(self) -> List[Movement]:
        cache_file = f"data/movements_{self.min_followers}_{self.min_tweets_day}_{self.time_lag}_.pickle"
        try:
            with open(cache_file, "rb") as f:
                movements = pickle.load(f)
        except:
            print(
                "Couldn't load cached movements. Loading movements from original files..."
            )
            movements = self._load_movements_from_files()
            with open(cache_file, "wb") as f:
                pickle.dump(movements, f)

        # print some direction stats
        directions = map(
            lambda m: -1
            if (m.price["movement percent"] < self.classify_threshold_down)
            else +1
            if (m.price["movement percent"] > self.classify_threshold_up)
            else 0,
            movements,
        )
        print("Movement distribution:")
        print(pd.Series(list(directions)).value_counts())

        # filter out movements that are too small
        return list(
            filter(
                lambda m: (m.price["movement percent"] < self.classify_threshold_down)
                or (m.price["movement percent"] > self.classify_threshold_up),
                movements,
            )
        )

    def prepare_data(self):
        self.movements = self._load_movements()

    def _get_movements(self, date_range: DateRange):
        return [
            m
            for m in self.movements
            if m.day >= date_range.start and m.day < date_range.end
        ]

    def setup(self, stage: Optional[str] = None):
        X_train = self._get_movements(self.data_split.train)
        X_val = self._get_movements(self.data_split.val)
        X_test = self._get_movements(self.data_split.test)

        self.train_ds = MovementDataset(X_train)
        self.val_ds = MovementDataset(X_val)
        self.test_ds = MovementDataset(X_test)

    def _coll_samples(self, batch):
        tweets = list(map(lambda x: x[0:2], batch))
        prices = torch.stack(list(map(lambda x: torch.tensor(x[-1]), batch))).float()
        return tweets, prices

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self._coll_samples,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._coll_samples,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self._coll_samples,
            shuffle=False,
            num_workers=self.num_workers,
        )
