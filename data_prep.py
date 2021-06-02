# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import json
import os
import itertools


# %%
def load_tweets(min_followers=None, tweet_path="data/tweet/raw"):
    dfs = []

    for symbol in tqdm(os.listdir(tweet_path)):
        for day in os.listdir(os.path.join(tweet_path, symbol)):
            file_path = os.path.join(tweet_path, symbol, day)

            with open(file_path) as f:
                content = f.readlines()
                d = list(map(json.loads, content))
                df = pd.DataFrame(d)

                if min_followers is not None:
                    df = df[df["user"].apply(lambda u: u["followers_count"]) > min_followers]
                
                if len(df) > 0:
                    dfs.append(df)

    tweets = pd.concat(dfs)
    print(f"Found {len(tweets)} tweets")

    simple_tweets = pd.DataFrame()
    simple_tweets["date"] = pd.to_datetime(tweets["created_at"])
    simple_tweets["text"] = tweets["text"].apply(lambda t: t.replace("\n", " "))
    simple_tweets["user_name"] = tweets["user"].apply(lambda u: u["name"])
    simple_tweets["user_followers"] = tweets["user"].apply(lambda u: u["followers_count"])
    simple_tweets["sym"] = tweets["entities"].apply(lambda entities: list(map(lambda s: s["text"], entities["symbols"])))

    return simple_tweets

# load_tweets(100000)


# %%
def load_prices(price_path="data/price/preprocessed"):
    dfs = []
    cols = ["date", "movement percent", "norm open price", "norm high price", "norm low price", "norm close price", "volume"]

    for symbol in tqdm(os.listdir(price_path)):
        file_path = os.path.join(price_path, symbol)

        df = pd.read_csv(file_path, sep="\t", names=cols)
        df["symbol"] = symbol.replace(".txt", "")
        dfs.append(df)

    prices = pd.concat(dfs, axis=0)
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.set_index(['symbol', 'date'])
    return prices

# load_prices()


# %%
class Movement:
    def __init__(self, tweets, stock, price, day):
        self.tweets = tweets
        self.stock = stock
        self.price = price
        self.day = day

    def __repr__(self):
        return f"Movement of {self.stock} on {self.day.date()}: {len(self.tweets)} tweet(s)"


# %%
def load_movements(min_tweets_day=None):
    tweets = load_tweets()
    prices = load_prices()
    movements = []
    missing_prices = []

    for day in tqdm(prices.index.get_level_values('date').unique()):
        day = pd.to_datetime(day)
        todays_tweets = tweets[tweets['date'].dt.date == day]
        todays_stocks = pd.unique(list(itertools.chain(*todays_tweets["sym"])))
        for stock in todays_stocks:
            rel_tweets = todays_tweets[todays_tweets["sym"].apply(lambda x: stock in x)]
            try:
                movements.append(Movement(rel_tweets, stock, prices.loc[stock, day], day))
            except Exception as e:
                missing_prices.append(e)

    nr_movements = len(movements)

    if min_tweets_day is not None:
        movements = list(filter(lambda m: len(m.tweets) > min_tweets_day, movements))

    print(f"Loaded {nr_movements} movements, returning {len(movements)}. Found no price for {len(missing_prices)}.")

    return movements

# load_movements()


# %%
# import matplotlib.pyplot as plt
# plt.hist(list(map(lambda m: len(m.tweets), movements)), bins=100, range=(0,50))


