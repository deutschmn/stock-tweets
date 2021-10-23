from ctypes import ArgumentError
from typing import List, Optional, Tuple
import json
import os
from operator import attrgetter
import itertools
import pickle
import re
import sys

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
from pytorch_lightning.core.datamodule import LightningDataModule
from loguru import logger

from src.data.dataset import MovementDataset
from src.data.movement import (
    Direction,
    ModelOutput,
    ClassifiedMovement,
    Movement,
    Tweet,
)
from src.util.splits import DataSplit, DateRange
from src.util.stats import movement_stats


class MovementDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_split: DataSplit,
        tweet_path: str,
        price_path: str,
        min_followers: int,
        min_tweets_day: int,
        more_recent_first: bool,
        classify_threshold_up: Optional[float] = None,
        classify_threshold_down: Optional[float] = None,
        classify_threshold_min_spread: Optional[float] = None,
        time_lag: Optional[int] = None,
        max_lag: Optional[int] = None,
        num_workers: int = 0,
    ):
        """Inits the DataModule

        Args:
            batch_size (int): Batch size to use for train/val/test loading
            data_split (DataSplit): How the data is split
            classify_threshold_up (Optional[float]): Threshold for when a course is classified as going UP
            classify_threshold_down (Optional[float]): Threshold for when a course is classified as going DOWN
            classify_threshold_min_spread: (Optional[float]): If this is set, thresholds are determined automatically so that the train data set is balanced. This is the minimum absolute value for either up or down threshold.
            min_followers (int): Minimum number of followers to consider a tweet from this author
            min_tweets_day (int): Minimum number of tweets about a stock on one day for a movement to be considered as a sample
            more_recent_first (bool): If true, more recent tweets come first, else the other way around
            time_lag (int, optional): Number of days between tweets and supposed market reaction. If None, all tweets before price date are returned.
            max_lag (int, optional): Maximum number of days between tweet and price for a tweet to be considered for the movement
            tweet_path (str): Path to tweet data.
            price_path (str): Path to price data.
            num_workers (int): Number of workers for the DataLoaders
        """
        super().__init__()

        self.batch_size = batch_size
        self.data_split = data_split
        self.classify_threshold_up = classify_threshold_up
        self.classify_threshold_down = classify_threshold_down
        self.classify_threshold_min_spread = classify_threshold_min_spread
        self.tweet_path = tweet_path
        self.price_path = price_path
        self.min_followers = min_followers
        self.min_tweets_day = min_tweets_day
        self.more_recent_first = more_recent_first
        self.time_lag = time_lag
        self.max_lag = max_lag
        self.num_workers = num_workers

        self.movements: List[ClassifiedMovement] = []
        self.train_ds: Optional[MovementDataset] = None
        self.val_ds: Optional[MovementDataset] = None
        self.test_ds: Optional[MovementDataset] = None

        if not (
            (
                self.classify_threshold_up is not None
                and self.classify_threshold_down is not None
                and self.classify_threshold_min_spread is None
            )
            or (
                self.classify_threshold_up is None
                and self.classify_threshold_down is None
                and self.classify_threshold_min_spread is not None
            )
        ):
            raise ArgumentError("Invalid classify_threshold config")

        if self.time_lag is not None and self.max_lag is not None:
            raise ArgumentError("Cannot specify both time_lag and max_lag")

    def _clean_tweet(self, tweet: str):
        clean = tweet

        clean = clean.replace("\n", " ")  # remove new lines
        clean = re.sub(r"http\S+", "", clean)  # remove urls

        return clean

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
        simple_tweets["text"] = tweets["text"].apply(lambda t: self._clean_tweet(t))
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
            day = pd.to_datetime(day)

            if self.time_lag is not None:
                # tweets from time lag day
                tweet_day = day - pd.to_timedelta(self.time_lag, unit="days")
                relevant_tweets = tweets[tweets["date"].dt.date == tweet_day]
            else:
                # all tweets before price day (within max_lag)
                oldest_tweet_day = day - pd.to_timedelta(self.max_lag, unit="days")
                relevant_tweets = tweets[
                    (tweets["date"].dt.date < day)
                    & (tweets["date"].dt.date > oldest_tweet_day)
                ]

            relevant_stocks = pd.unique(list(itertools.chain(*relevant_tweets["sym"])))
            for stock in relevant_stocks:
                rel_tweets = relevant_tweets[
                    relevant_tweets["sym"].apply(lambda x: stock in x)
                ]
                try:
                    price = prices.loc[stock, day]
                    # sort tweets
                    rel_tweets = rel_tweets.sort_values(
                        by="date", ascending=not self.more_recent_first
                    )

                    movements.append(Movement(rel_tweets, stock, price, day))
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
        # new cached file if any of these params changes
        id_fields = [
            self.min_followers,
            self.min_tweets_day,
            self.time_lag,
            self.more_recent_first,
            self.max_lag,
        ]
        cache_file = f"data/movements_{'_'.join(map(str, id_fields))}.pickle"
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

        return movements
    def _classify_movements(
        self, movements: List[Movement], threshold_up: float, threshold_down: float
    ) -> List[ClassifiedMovement]:
        classified_movements: List[ClassifiedMovement] = []
        for m in movements:
            direction = (
                Direction.UP
                if m.price_movement >= threshold_up
                else Direction.DOWN
                if m.price_movement <= threshold_down
                else Direction.SAME
            )
            # filter out movements that are too small (categorised as SAME)
            if direction != Direction.SAME:
                classified_movements.append(
                    ClassifiedMovement(
                        tweets=m.tweets,
                        stock=m.stock,
                        price=m.price,
                        day=m.day,
                        direction=direction,
                    )
                )
        return classified_movements

    def _find_thresholds(self, movements: List[Movement]) -> Tuple[float, float]:
        nr_ups = len(
            [
                m
                for m in movements
                if m.price_movement > self.classify_threshold_min_spread
            ]
        )
        nr_downs = len(
            [
                m
                for m in movements
                if m.price_movement < -self.classify_threshold_min_spread
            ]
        )

        if nr_ups == nr_downs:
            # already balanced
            thres_up = self.classify_threshold_min_spread
            thres_down = -self.classify_threshold_min_spread
        else:
            limit_upwards = nr_ups > nr_downs

            if limit_upwards:
                thres_up = None
                thres_down = -self.classify_threshold_min_spread
                down = 0
            else:
                thres_up = self.classify_threshold_min_spread
                thres_down = None
                up = 0

            movements.sort(key=attrgetter("price_movement"), reverse=not limit_upwards)

            total = len(movements)
            same = 0

            min_error = sys.maxsize
            for movement in movements:
                total_considered = total - same
                if limit_upwards:
                    if movement.price_movement < -self.classify_threshold_min_spread:
                        down += 1
                    elif movement.price_movement < self.classify_threshold_min_spread:
                        same += 1
                    else:
                        up = total - down - same
                        portion = up / total_considered
                        error = abs(0.5 - portion)  # ideally, portion is 50%
                        if error < min_error:
                            min_error = error
                            thres_up = movement.price_movement
                            same += 1  # if we don't take this thres, this mov will count as same (in next iteration)
                        else:
                            break
                else:
                    if movement.price_movement > self.classify_threshold_min_spread:
                        up += 1
                    elif movement.price_movement > -self.classify_threshold_min_spread:
                        same += 1
                    else:
                        down = total - up - same
                        portion = down / total_considered
                        error = abs(0.5 - portion)  # ideally, portion is 50%
                        if error < min_error:
                            min_error = error
                            thres_down = movement.price_movement
                            same += 1  # if we don't take this thres, this mov will count as same (in next iteration)
                        else:
                            break

            if thres_up is None or thres_down is None:
                raise RuntimeError(
                    f"Couldn't find threshold, min spread of {self.classify_threshold_min_spread} exceeded"
                )

        return thres_up, thres_down

    def prepare_data(self):
        self.movements: List[Movement] = self._load_movements()

    def _get_movements(self, date_range: DateRange) -> List[Movement]:
        return [
            m
            for m in self.movements
            if m.day >= date_range.start and m.day < date_range.end
        ]

    def setup(self, stage: Optional[str] = None):
        X_train_unclass = self._get_movements(self.data_split.train)
        X_val_unclass = self._get_movements(self.data_split.val)
        X_test_unclass = self._get_movements(self.data_split.test)

        if self.classify_threshold_min_spread is not None:
            thres_up, thres_down = self._find_thresholds(X_train_unclass)
        else:
            thres_up, thres_down = (
                self.classify_threshold_up,
                self.classify_threshold_down,
            )

        logger.info(f"Thresholds for classification: {thres_up = }, {thres_down = }")

        X_train = self._classify_movements(X_train_unclass, thres_up, thres_down)
        X_val = self._classify_movements(X_val_unclass, thres_up, thres_down)
        X_test = self._classify_movements(X_test_unclass, thres_up, thres_down)

        logger.info(f"{movement_stats(X_train) = }")
        logger.info(f"{movement_stats(X_val) = }")
        logger.info(f"{movement_stats(X_test) = }")

        self.train_ds = MovementDataset(X_train)
        self.val_ds = MovementDataset(X_val)
        self.test_ds = MovementDataset(X_test)

    def _coll_samples(
        self, batch: List[ClassifiedMovement]
    ) -> Tuple[List[List[Tweet]], List[ModelOutput]]:
        model_input = [x.model_input for x in batch]
        target = [x.model_output for x in batch]
        return model_input, target

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
