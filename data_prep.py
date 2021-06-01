# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import json
import os


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
    simple_tweets["sym"] = tweets["entities"].apply(lambda entities: ",".join(list(map(lambda s: s["text"], entities["symbols"]))))

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

    return pd.concat(dfs, axis=0)

# load_prices()


