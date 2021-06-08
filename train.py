import os
os.environ["CUDA_VISIBLE_DEVICES"]="10,11,12,13"

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb

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

        return encd_tweets['input_ids'], encd_tweets['token_type_ids'], \
            encd_tweets['attention_mask'], list(m.tweets["user_followers"]), \
            m.price["movement percent"]

    def coll_samples(batch):
        tweets = list(map(lambda x: x[0:4], batch))
        prices = torch.stack(list(map(lambda x: torch.tensor(x[4]), batch))).float()
        return tweets, prices


def load_movements():
    cache_file = "data/movements.pickle"
    try:
        with open(cache_file, "rb") as f:
            movements = pickle.load(f)
    except:
        print("Couldn't load cached movements. Loading movements from original files.")
        movements = data_prep.load_movements(min_tweets_day=5)
        with open(cache_file, "wb") as f:
            pickle.dump(movements, f)
    return movements


def classify_movement(m):
    classes = torch.zeros_like(m)
    classes[m > wandb.config.classify_threshold_up] = 1
    classes[m < wandb.config.classify_threshold_down] = -1
    return classes


def compute_metrics(pred, target):
    pred = pred.cpu().detach()
    target = target.cpu().detach()
    
    pred_classes = classify_movement(pred)
    target_classes = classify_movement(target)
    return {
        "mse": mean_squared_error(target, pred, squared=True),
        "rmse": mean_squared_error(target, pred, squared=False),
        "acc": accuracy_score(target_classes, pred_classes),
        "f1": f1_score(target_classes, pred_classes, average="micro")
    }


def evaluate(model, loader, metric_prefix):
    total_metrics = None
    for tweets, target in tqdm(loader):
        pred = model(tweets)
        metrics = compute_metrics(pred, target)
        
        if total_metrics is None:
            total_metrics = metrics
        else:
            for k, v in metrics.items():
                total_metrics[k] += v
    
    # compute average over batches and add prefix to label
    total_metrics = {metric_prefix + '_' + k: v / len(loader) for k, v in total_metrics.items()}

    return total_metrics


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

    device = torch.device(config.device)
    model = MovementPredictor(config.transformer_model, device, hidden_dim=config.hidden_dim, 
                                freeze_transformer=config.freeze_transformer)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
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
