import os

os.environ["WANDB_SILENT"] = "true"

import autogpu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_prep
import wandb
from data_loading import MovementDataset, classify_movement


def log_to_wandb(wandb_run, tweet_df, movement_df):
    x = plt.figure()
    plt.hist(tweet_df["sentiment"], bins=30)
    sent_hist = wandb.Image(x)

    x = plt.figure()
    plt.hist(tweet_df["weight"], bins=30)
    weight_hist = wandb.Image(x)

    wandb_run = wandb.init(
        id=wandb_run.id, entity="deutschmann", project="stock-tweets", resume="allow"
    )
    wandb_run.log(
        {
            "corr_table": wandb.Table(data=movement_df.corr()),
            "sentiment_hist": sent_hist,
            "weight_hist": weight_hist,
        }
    )
    wandb_run.finish()


def analyse(run, values):
    api = wandb.Api()
    runs = api.runs(f"deutschmann/stock-tweets")
    wandb_run = list(filter(lambda r: r.name == run, runs))[0]

    movements = data_prep.load_movements(
        classify_threshold_up=wandb_run.config["classify_threshold_up"],
        classify_threshold_down=wandb_run.config["classify_threshold_down"],
        min_followers=None,
        min_tweets_day=wandb_run.config["min_tweets_day"],
        time_lag=wandb_run.config["time_lag"],
    )
    device = autogpu.freest()

    model = torch.load(f"artifacts/model_{run}.pt", map_location=device)
    model.device = device

    # add hocks for tracking activations
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    model.transformer.register_forward_hook(get_activation("transformer"))
    model.attention.register_forward_hook(get_activation("attention"))

    ds = MovementDataset(movements, model.transformer_name)
    dl = DataLoader(
        ds, batch_size=1, collate_fn=MovementDataset.coll_samples, shuffle=False
    )

    weights = []
    sentiments = []

    for tweets, target in tqdm(dl):
        pred = model(tweets)
        weights.append(activation["attention"].cpu().detach().squeeze().numpy())
        sentiments.append(
            torch.softmax(activation["transformer"].logits, dim=-1).cpu().numpy()
        )

    weights = np.concatenate(weights)
    sentiments = np.concatenate(sentiments) @ values

    tweet_df = None
    for mov_id, tweets in enumerate(map(lambda m: m.tweets, movements)):
        tweets["mov_id"] = mov_id
        if tweet_df is None:
            tweet_df = tweets
        else:
            tweet_df = tweet_df.append(tweets)

    tweet_df["weight"] = weights
    tweet_df["sentiment"] = sentiments
    tweet_df["weighted_sentiment"] = tweet_df["sentiment"] * tweet_df["weight"]

    movement_df = pd.DataFrame(
        [{"stock": o.stock, "price": o.price["movement percent"]} for o in movements]
    )
    movement_df["sent_mean"] = tweet_df.groupby(by="mov_id").mean()["sentiment"]
    movement_df["sent_weighted"] = tweet_df.groupby(by="mov_id").sum()[
        "weighted_sentiment"
    ]

    log_to_wandb(wandb_run, tweet_df, movement_df)


def main():
    run = "stellar-wind-175"
    # labels = ["neg", "neutral", "pos"]
    # values = np.array([-1, 0, 1])[:, np.newaxis]
    values = np.array([1, 2, 3, 4, 5])[:, np.newaxis]

    analyse(run, values)


if __name__ == "__main__":
    main()
