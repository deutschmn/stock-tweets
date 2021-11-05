from ctypes import ArgumentError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

def plot_confusion(confusion_matrix: np.ndarray, confusion_type: str, epoch: int):
    """Plots a confusion matrix

    Args:
        confusion_matrix (np.ndarray): confusion matrix
        confusion_type (str): type of matrix to create ("absolute", "precision" or "recall")

    Returns:
        matplotlib figure object (to log or call plt.show on)
    """
    plt.close("all")

    fig = plt.figure()

    classes = ["down", "up"]

    df_cm = pd.DataFrame(
        confusion_matrix,
        columns=classes,
        index=classes,
    )

    if confusion_type == "absolute":
        plt.title(f"Absolute confusion ({epoch = })")
        fmt = "g"
    elif confusion_type == "precision":
        plt.title(f"Relative confusion (diag = precision, {epoch = })")
        df_cm = df_cm.divide(df_cm.sum(axis=0), axis="columns") * 100
        fmt = ".1f"
    elif confusion_type == "recall":
        plt.title(f"Relative confusion (diag = recall, {epoch = })")
        df_cm = df_cm.divide(df_cm.sum(axis=1), axis="rows") * 100
        fmt = ".1f"
    else:
        raise ArgumentError(f"Unknown type '{confusion_type}'")

    sn.heatmap(df_cm, annot=True, fmt=fmt)

    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    return fig
