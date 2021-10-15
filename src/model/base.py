from typing import Any
from pytorch_lightning.core.lightning import LightningModule
from abc import ABC, abstractmethod

from loguru import logger
import torch
from torch import nn
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from transformers import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
import wandb

from src.util.plotting import plot_confusion


class MovementPredictor(LightningModule, ABC):
    def __init__(
        self,
        optim: str,
        lr: float,
        classify_threshold_up: float,
        classify_threshold_down: float,
        transformer_model: str,
        transformer_out: int,
        hidden_dim: int,
        freeze_transformer: bool,
        tweet_max_len: int,
    ):
        """Inits the model

        Args:
            optim (str): Name of torch.optim.* optimizer to use
            lr (float): Learning rate to use for the optimiser
            classify_threshold_up (float): Threshold for when a course is classified as going UP
            classify_threshold_down (float): Threshold for when a course is classified as going DOWN
            transformer_model (str): Which Hugging Face transformer model to use
            transformer_out (int): The number of output values of the transformer
            hidden_dim (int): Hidden dimension to use for classification layer. If 0, no hidden layer is used
            freeze_transformer (bool): If true, freezes transformer weights, or fine-tune them otherwise
            tweet_max_len (int): Length to which tweets are truncated or padded.
        """
        super().__init__()

        self.transformer_name = transformer_model
        self.optim = optim
        self.lr = lr
        self.classify_threshold_up = classify_threshold_up
        self.classify_threshold_down = classify_threshold_down
        self.tweet_max_len = tweet_max_len

        self.loss = self.setup_loss()

        metrics = self.setup_metrics()
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_model
        )

        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if hidden_dim > 0:
            self.sentiment_classifier = nn.Sequential(
                nn.Linear(transformer_out, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.sentiment_classifier = nn.Linear(transformer_out, 1)

    @abstractmethod
    def setup_loss(self) -> nn.Module:
        pass

    @abstractmethod
    def preprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        """Processes targets before loss computation"""
        pass

    @abstractmethod
    def postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        """Processes preds after loss before metric computation"""
        pass

    @abstractmethod
    def postprocess_targets(self, target: torch.Tensor) -> torch.Tensor:
        """Processes targets after loss before metric computation"""
        pass

    @abstractmethod
    def setup_metrics(self) -> MetricCollection:
        pass

    @abstractmethod
    def forward(self, model_input):
        pass

    def _step(self, batch: Any, batch_idx: int, mode: str, metrics: MetricCollection):
        tweets, target = batch

        logits = self(tweets)
        target = self.preprocess_targets(target)

        loss = self.loss(logits, target)

        pred = self.postprocess_preds(logits)
        target = self.postprocess_targets(target)

        self.log(f"{mode}/loss", loss.item())

        if (mode != "train") or (mode == "train" and batch_idx % 10 == 0):
            logger.debug(f"{logits = }")
            logger.debug(f"{torch.sigmoid(logits) = }")
            logger.debug(f"{pred = }")
            logger.debug(f"{target = }")
            for metric_name, metric in metrics.items():
                value = metric(pred, target)
                # confusion only logged on epoch end
                if metric_name != "confusion_matrix":
                    self.log(name=f"{mode}/step/{metric_name}", value=value)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train", metrics=self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val", metrics=self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test", metrics=self.test_metrics)

    def _epoch_end(self, training_step_outputs, mode, metrics):
        for metric_name, metric in metrics.items():
            if metric_name == "confusion_matrix":
                self._log_confusion(mode, metric_name, metric, interval="full")
            else:
                self.log(name=f"{mode}/full/{metric_name}", value=metric.compute())

    def _log_confusion(
        self, mode: str, metric_name: str, metric: Metric, interval: str
    ):
        # workaround as PyTorch Lightning doesn't support logging imgs directly
        wandb.log(
            {
                f"{mode}/{interval}/{metric_name}": wandb.Image(
                    plot_confusion(
                        metric.compute().detach().cpu().numpy(),
                        confusion_type="absolute",
                    )
                )
            }
        )

    def training_epoch_end(self, training_step_outputs):
        return self._epoch_end(
            training_step_outputs, mode="train", metrics=self.train_metrics
        )

    def validation_epoch_end(self, training_step_outputs):
        return self._epoch_end(
            training_step_outputs, mode="val", metrics=self.val_metrics
        )

    def test_epoch_end(self, training_step_outputs):
        return self._epoch_end(
            training_step_outputs, mode="test", metrics=self.test_metrics
        )

    def configure_optimizers(self):
        return getattr(torch.optim, self.optim)(self.parameters(), lr=self.lr)
