from ctypes import ArgumentError
from typing import Any
from pytorch_lightning.core.lightning import LightningModule
from abc import ABC, abstractmethod

import torch
from torch import nn
from torchmetrics.collections import MetricCollection
from transformers import AutoModelForSequenceClassification


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
        attention_input: str,
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
            attention_input (str): which inputs to use for the attention ('followers', 'sentiment' or 'both')

        """
        super().__init__()

        self.transformer_name = transformer_model
        self.optim = optim
        self.lr = lr
        self.classify_threshold_up = classify_threshold_up
        self.classify_threshold_down = classify_threshold_down

        self.loss = self.setup_loss()

        metrics = self.setup_metrics()
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_model
        )

        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False

        self.attention_input = attention_input

        if attention_input == "sentiment":
            attention_in_dim = transformer_out
        elif attention_input == "followers":
            attention_in_dim = 1
        elif attention_input == "both":
            attention_in_dim = transformer_out + 1
        else:
            raise ArgumentError(f"Unknown attention input {attention_input}")

        # TODO more complex attention module?
        self.attention = nn.Sequential(
            nn.Linear(attention_in_dim, 1), nn.LeakyReLU(), nn.Softmax(dim=0)
        )

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

    def _forward_movement(self, tweets):
        tweets_encd = map(lambda x: x.to(self.device), tweets[0].values())
        tweets_followers = (
            torch.tensor(tweets[1], dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        )

        tweet_reps = self.transformer(*tweets_encd).logits

        if self.attention_input == "sentiment":
            attention_in = tweet_reps
        elif self.attention_input == "followers":
            attention_in = tweets_followers
        elif self.attention_input == "both":
            attention_in = torch.cat([tweet_reps, tweets_followers], dim=-1)
        else:
            raise ArgumentError(f"Unknown attention input {self.attention_input}")

        attention_weights = self.attention(attention_in)
        return torch.mm(tweet_reps.T, attention_weights).squeeze()

    def forward(self, tweets):
        tweet_sentiment = torch.stack(list(map(self._forward_movement, tweets)))
        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)

    def _step(self, batch: Any, batch_idx: int, mode: str, metrics: MetricCollection):
        tweets, target = batch

        pred = self(tweets)
        target = self.preprocess_targets(target)

        loss = self.loss(pred, target)

        pred = self.postprocess_preds(pred)
        target = self.postprocess_targets(target)

        self.log(f"{mode}/loss", loss.item())

        if (mode != "train") or (mode == "train" and batch_idx % 10 == 0):
            for metric_name, metric in metrics.items():
                value = metric(pred, target)
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
            self.log(name=f"{mode}/full/{metric_name}", value=metric.compute())

    def training_epoch_end(self, training_step_outputs):
        return self._epoch_end(training_step_outputs, mode="train", metrics=self.train_metrics)

    def validation_epoch_end(self, training_step_outputs):
        return self._epoch_end(training_step_outputs, mode="val", metrics=self.val_metrics)

    def test_epoch_end(self, training_step_outputs):
        return self._epoch_end(training_step_outputs, mode="test", metrics=self.test_metrics)

    def configure_optimizers(self):
        return getattr(torch.optim, self.optim)(self.parameters(), lr=self.lr)
