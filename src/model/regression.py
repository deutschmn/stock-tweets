from typing import Dict, List
from torch import nn
import torch
from torchmetrics.metric import Metric
from src.model.base import MovementPredictor


class RegressionMovementPredictor(MovementPredictor):
    def setup_loss(self) -> nn.Module:
        return nn.MSELoss()

    def preprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        # for regression, targets stay the same
        return target

    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        # TODO implement (probably nothing to do?)
        raise NotImplementedError()

    def setup_metrics(self) -> Dict[str, List[Metric]]:
        # TODO implement regression metrics
        raise NotImplementedError()