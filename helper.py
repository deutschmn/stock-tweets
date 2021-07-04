import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

import wandb


def make_scatter(pred, target):
    # scatter plot of true x pred
    plt.figure(figsize=(10, 10))

    x = 0.15
    plt.plot([-x, x], [-x, x], alpha=0.5, color="grey", linestyle="--")
    plt.xlim(-x, x)
    plt.ylim(-x, x)

    plt.scatter(target, pred, alpha=0.3, s=5)
    plt.xlabel("true")
    plt.ylabel("pred")
    return wandb.Image(plt)


def make_confusion(target_classes, pred_classes):
    df_cm = pd.DataFrame(confusion_matrix(target_classes, pred_classes))
    sn.heatmap(df_cm, annot=True, fmt="g")

    plt.xlabel("pred")
    plt.ylabel("true")
    return wandb.Image(plt)
