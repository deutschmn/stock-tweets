from torch import nn
import torch
from torchmetrics.collections import MetricCollection
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from src.model.base import MovementPredictor


class RegressionMovementPredictor(MovementPredictor):
    def setup_loss(self) -> nn.Module:
        return nn.MSELoss()

    def setup_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "r2": R2Score()
            }
        )

    def preprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def postprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        return preds