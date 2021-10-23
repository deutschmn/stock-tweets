from typing import List
from torch import nn
import torch
from torchmetrics.collections import MetricCollection
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from src.data.movement import ModelOutput
from src.model.base import MovementPredictor


class RegressionMovementPredictor(MovementPredictor):
    def setup_loss(self) -> nn.Module:
        return nn.MSELoss()

    def setup_metrics(self) -> MetricCollection:
        return MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError(), "r2": R2Score()}
        )

    def preprocess_targets(self, target: List[ModelOutput]) -> torch.Tensor:
        return torch.stack(
            [torch.tensor(x.price_movement, device=self.device) for x in target]
        ).float()

    def postprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        return preds
