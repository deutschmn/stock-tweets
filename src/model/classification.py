from torch import nn
import torch
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix, MetricCollection
from src.model.base import MovementPredictor


class ClassificationMovementPredictor(MovementPredictor):
    def _classify_movement(self, movements):
        classes = torch.full_like(
            movements, fill_value=float("nan"), device=self.device
        )

        classes[movements > self.classify_threshold_up] = 1
        classes[movements < self.classify_threshold_down] = 0

        if torch.isnan(classes).any():
            raise RuntimeError(
                "Found NaN: movement inside of exclusion region detected"
            )

        return classes

    def setup_loss(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def preprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        # for classification, movements need to be classified for the loss
        return self._classify_movement(target)

    def setup_metrics(self) -> MetricCollection:
        num_classes = 2  # UP and DOWN
        
        # TODO add back in scatter and confusion plotting, maybe using
        # > "scatter": helper.make_scatter(pred, target),
        # > "confusion": helper.make_confusion(target_classes, pred_classes),

        return MetricCollection(
            {
                "acc": Accuracy(),
                "precision": Precision(num_classes=num_classes, average="macro"),
                "recall": Recall(num_classes=num_classes, average="macro"),
                "f1_macro": F1(num_classes=num_classes, average="macro"),
                "f1_micro": F1(num_classes=num_classes, average="micro"),
                # "confusion_matrix": ConfusionMatrix(num_classes=num_classes),
            }
        )

    def postprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        return target.int()

    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(preds) > 0.5).int()
