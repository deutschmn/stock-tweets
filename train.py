from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb
import pandas as pd
import matplotlib.pyplot as plt

import autogpu

from model import MovementPredictor
import data_prep


class MovementDataset(Dataset):
    def __init__(self, movements, transformer_model, tweet_maxlen=100):
        super()
        self.movements = movements
        
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.tweet_maxlen = tweet_maxlen

    def __len__(self):
        return len(self.movements)

    def __getitem__(self, idx):
        m = self.movements[idx]
        encd_tweets = self.tokenizer(list(m.tweets["text"]), 
                            return_tensors="pt", 
                            padding="max_length", 
                            max_length=self.tweet_maxlen,
                            truncation=True)

        return encd_tweets, list(m.tweets["user_followers"]), m.price["movement percent"]

    def coll_samples(batch):
        tweets = list(map(lambda x: x[0:2], batch))
        prices = torch.stack(list(map(lambda x: torch.tensor(x[-1]), batch))).float()
        return tweets, prices


def load_movements():
    cache_file = "data/movements.pickle"
    try:
        with open(cache_file, "rb") as f:
            movements = pickle.load(f)
    except:
        print("Couldn't load cached movements. Loading movements from original files.")
        movements = data_prep.load_movements(min_tweets_day=wandb.config.min_tweets_day)
        with open(cache_file, "wb") as f:
            pickle.dump(movements, f)

    # print some direction stats
    directions = map(lambda m: -1 if (m.price["movement percent"] < wandb.config.classify_threshold_down) else 
                               +1 if (m.price["movement percent"] > wandb.config.classify_threshold_up) else 0, movements)
    print("Movement distribution:")
    print(pd.Series(list(directions)).value_counts())

    # filter out movements that are too small
    return list(filter(lambda m: (m.price["movement percent"] < wandb.config.classify_threshold_down) 
                                or (m.price["movement percent"] > wandb.config.classify_threshold_up), 
                        movements))


def classify_movement(m):
    classes = torch.zeros_like(m)
    classes[m > wandb.config.classify_threshold_up] = 1
    classes[m < wandb.config.classify_threshold_down] = -1
    return classes


def compute_metrics(pred, target):
    pred = pred.cpu().detach()
    target = target.cpu().detach()
    
    # scatter plot of true x pred
    plt.figure(figsize=(10,10))
    plt.scatter(target, pred, alpha=0.3, s=5)
    plt.xlabel('true')
    plt.ylabel('pred')

    pred_classes = classify_movement(pred)
    target_classes = classify_movement(target)

    return {
        "mse": mean_squared_error(target, pred, squared=True),
        "rmse": mean_squared_error(target, pred, squared=False),
        "r2": r2_score(target, pred),
        "acc": accuracy_score(target_classes, pred_classes),
        "f1": f1_score(target_classes, pred_classes, average="micro"),
        "scatter": wandb.Image(plt)
    }


def evaluate(model, loader, metric_prefix):
    pred_list = []
    target_list = []

    for tweets, target in tqdm(loader):
        pred_list.append(model(tweets))
        target_list.append(target)
        
    metrics = compute_metrics(torch.cat(pred_list), torch.cat(target_list))
    
    # add prefix to label
    metrics = {metric_prefix + '_' + k: v for k, v in metrics.items()}

    return metrics


def main():
    wandb.init(project='stock-tweets', entity='deutschmann', config='config.yaml')
    config = wandb.config

    movements = load_movements()
    X_train, rest = train_test_split(movements, train_size=config.train_size)
    val_of_rest = config.val_size / (1 - config.train_size)
    X_val, X_test = train_test_split(rest, train_size=val_of_rest)
    
    train_ds = MovementDataset(X_train, config.transformer_model)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, 
                                collate_fn=MovementDataset.coll_samples)
    val_ds = MovementDataset(X_val, config.transformer_model)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, 
                                collate_fn=MovementDataset.coll_samples)
    test_ds = MovementDataset(X_test, config.transformer_model)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, 
                                collate_fn=MovementDataset.coll_samples)

    device = torch.device(config.device) if config.device else autogpu.freest()
    
    model = MovementPredictor(config.transformer_model, config.transformer_out, device, 
                                hidden_dim=config.hidden_dim, 
                                freeze_transformer=config.freeze_transformer)
    model.to(device)
    wandb.watch(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss = nn.MSELoss()

    for epoch in tqdm(range(config.epochs)):
        model.train()
        for tweets, target in tqdm(train_loader, desc=f"epoch {epoch}"):
            pred = model(tweets)
            l = loss(pred, target.to(device))
            wandb.log({'train_loss': l})
            l.backward()
            optim.step()
        
        model.eval()
        with torch.no_grad():
            train_metrics = evaluate(model, train_loader, metric_prefix='train')
            val_metrics = evaluate(model, val_loader, metric_prefix='val')
            test_metrics = evaluate(model, test_loader, metric_prefix='test')

            wandb.log({'epoch': epoch} | train_metrics | val_metrics | test_metrics)


if __name__ == "__main__":
    main()
