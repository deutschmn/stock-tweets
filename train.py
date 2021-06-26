from ctypes import ArgumentError
from tqdm import tqdm
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score, confusion_matrix
from torch.utils.data import DataLoader
import seaborn as sn
import pandas as pd
import wandb
import matplotlib.pyplot as plt

import autogpu

from model import MovementPredictor
import data_prep
from data_loading import MovementDataset, classify_movement


def get_criterion(loss_type):
    if loss_type == 'regression':
        reg_crit = nn.MSELoss()
        def reg_criterion(input, target):
            return reg_crit(input, target)
        return reg_criterion
    elif loss_type == 'classification':
        class_crit = nn.BCEWithLogitsLoss()
        def class_criterion(input, target):
            return class_crit(input, classify_movement(target, 
                                                        wandb.config.classify_threshold_up,
                                                        wandb.config.classify_threshold_down))    
        return class_criterion
    else:
        raise ArgumentError(f"Unknown loss type {loss_type}")


def make_scatter(pred, target):
    # scatter plot of true x pred
    plt.figure(figsize=(10,10))

    x = 0.15
    plt.plot([-x,x],[-x,x], alpha=0.5, color="grey", linestyle="--")
    plt.xlim(-x, x)
    plt.ylim(-x, x)

    plt.scatter(target, pred, alpha=0.3, s=5)
    plt.xlabel('true')
    plt.ylabel('pred')
    return wandb.Image(plt)


def make_confusion(target_classes, pred_classes):
    df_cm = pd.DataFrame(confusion_matrix(target_classes, pred_classes))
    sn.heatmap(df_cm, annot=True, fmt='g')

    plt.xlabel("pred")
    plt.ylabel("true")
    return wandb.Image(plt)


def compute_metrics(pred, target):
    pred = pred.cpu().detach()
    target = target.cpu().detach()
    
    if wandb.config.loss_type == 'regression':
        pred_classes = classify_movement(pred, wandb.config.classify_threshold_up,
                                               wandb.config.classify_threshold_down)
    elif wandb.config.loss_type == 'classification':
        pred_classes = (torch.sigmoid(pred) > 0.5).int().float()
        pred_classes = (torch.sigmoid(pred) > 0.5).int().float()
    else:
        raise ArgumentError(f"Unknown loss type {wandb.config.loss_type}")
    
    target_classes = classify_movement(target, wandb.config.classify_threshold_up,
                                                wandb.config.classify_threshold_down)

    return {
        "mse": mean_squared_error(target, pred, squared=True),
        "rmse": mean_squared_error(target, pred, squared=False),
        "r2": r2_score(target, pred),
        "acc": accuracy_score(target_classes, pred_classes),
        "f1": f1_score(target_classes, pred_classes, average="micro"),
        "scatter": make_scatter(pred, target),
        "confusion": make_confusion(target_classes, pred_classes)
    }


def evaluate(model, loader, criterion, device, metric_prefix):
    pred_list = []
    target_list = []

    running_loss = 0.0
    for tweets, target in tqdm(loader):
        pred = model(tweets)
        running_loss += criterion(pred, target.to(device)).item()

        pred_list.append(pred)
        target_list.append(target)
    
    metrics = compute_metrics(torch.cat(pred_list), torch.cat(target_list)) | \
                                {'loss': running_loss / len(loader)}

    # add prefix to label
    metrics = {metric_prefix + '_' + k: v for k, v in metrics.items()}

    return metrics


def main():
    wandb.init(project='stock-tweets', entity='deutschmann', config='config.yaml')
    config = wandb.config

    movements = data_prep.load_movements(wandb.config.classify_threshold_up, 
                    wandb.config.classify_threshold_down, 
                    min_followers=None, 
                    min_tweets_day=wandb.config.min_tweets_day, 
                    time_lag=wandb.config.time_lag)
                    
    X_train, rest = train_test_split(movements, train_size=config.train_size)
    val_of_rest = config.val_size / (1 - config.train_size)
    X_val, X_test = train_test_split(rest, train_size=val_of_rest)
    
    train_ds = MovementDataset(X_train, config.transformer_model)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, 
                                collate_fn=MovementDataset.coll_samples, shuffle=True)
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
    
    optim = getattr(torch.optim, config.optim)(model.parameters(), lr=config.lr)
    criterion = get_criterion(config.loss_type)

    min_val_loss = float('inf')

    for epoch in tqdm(range(config.epochs)):
        model.train()

        running_loss = 0.0

        for tweets, target in tqdm(train_loader, desc=f"epoch {epoch}"):
            pred = model(tweets)
            loss = criterion(pred, target.to(device))
            loss.backward()
            optim.step()

            running_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate(model, val_loader, criterion, device, metric_prefix='val')
            test_metrics = evaluate(model, test_loader, criterion, device, metric_prefix='test')

            if epoch % 5 == 0:
                train_metrics = evaluate(model, train_loader, criterion, device, metric_prefix='train')
                wandb.log({'epoch': epoch} | train_metrics )

            wandb.log({'epoch': epoch, 'train_loss': running_loss / len(train_loader)} 
                        | val_metrics | test_metrics)

            if val_metrics['val_loss'] < min_val_loss:
                print("New min val loss!")
                min_val_loss = val_metrics['val_loss']

                best_val_metrics = {'best_' + k: v for k, v in val_metrics.items()}
                wandb.log({'epoch': epoch} | best_val_metrics)

                torch.save(model, f"artifacts/model_{wandb.run.name}.pt")


    train_metrics = evaluate(model, train_loader, criterion, device, metric_prefix='train')
    wandb.log({'epoch': epoch} | train_metrics)


if __name__ == "__main__":
    main()
