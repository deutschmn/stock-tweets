from typing import List
from torch import nn
import torch
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1,
    ConfusionMatrix,
    MetricCollection,
    MatthewsCorrcoef,
)
from src.data.movement import ModelOutput
from src.model.base import MovementPredictor


class ClassificationMovementPredictor(MovementPredictor):
    def setup_loss(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def preprocess_targets(self, target: List[ModelOutput]) -> torch.Tensor:
        return torch.stack(
            [torch.tensor(x.direction, device=self.device) for x in target]
        ).float()

    def setup_metrics(self) -> MetricCollection:
        num_classes = 2  # UP and DOWN

        return MetricCollection(
            {
                "acc": Accuracy(),
                "precision": Precision(num_classes=num_classes, average="macro"),
                "recall": Recall(num_classes=num_classes, average="macro"),
                "f1_macro": F1(num_classes=num_classes, average="macro"),
                "f1_micro": F1(num_classes=num_classes, average="micro"),
                "confusion_matrix": ConfusionMatrix(num_classes=num_classes),
                "mcc": MatthewsCorrcoef(num_classes=num_classes),
            }
        )

    def postprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        return target.int()

    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(preds) > 0.5).int()
