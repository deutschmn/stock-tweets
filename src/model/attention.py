from ctypes import ArgumentError
import torch
from torch import nn

from src.model.base import MovementPredictor
from src.model.transformer import TransformerConfig


class AttentionMovementPredictor(MovementPredictor):
    def __init__(
        self,
        optim: str,
        lr: float,
        classify_threshold_up: float,
        classify_threshold_down: float,
        transformer_config: TransformerConfig,
        hidden_dim: int,
        freeze_transformer: bool,
        tweet_max_len: int,
        attention_input: str,
    ):
        """
        Same args as MovementPredictor except:
            attention_input (str): which inputs to use for the attention ('followers', 'sentiment' or 'both')
        """
        super().__init__(
            optim,
            lr,
            classify_threshold_up,
            classify_threshold_down,
            transformer_config,
            hidden_dim,
            freeze_transformer,
            tweet_max_len,
        )

        self.attention_input = attention_input

        if attention_input == "sentiment":
            attention_in_dim = transformer_config.out_dim
        elif attention_input == "followers":
            attention_in_dim = 1
        elif attention_input == "both":
            attention_in_dim = transformer_config.out_dim + 1
        else:
            raise ArgumentError(f"Unknown attention input {attention_input}")

        # TODO more complex attention module?
        self.attention = nn.Sequential(
            nn.Linear(attention_in_dim, 1), nn.LeakyReLU(), nn.Softmax(dim=0)
        )

    def _forward_movement(self, model_inputs):
        tweets, followers = model_inputs

        tweets_encd = self.tokenizer(
            tweets,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tweet_max_len,
            truncation=True,
        ).to(self.device)

        tweets_followers = (
            torch.tensor(followers, dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        )

        tweet_reps = self.transformer(**tweets_encd).logits

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

    def forward(self, model_input):
        tweet_sentiment = torch.stack(list(map(self._forward_movement, model_input)))
        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)