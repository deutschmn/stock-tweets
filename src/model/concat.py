from typing import List

from loguru import logger
import torch

from src.data.movement import Tweet
from src.model.base import MovementPredictor
from src.model.transformer import TransformerConfig


class ConcatMovementPredictor(MovementPredictor):
    def __init__(
        self,
        optim: str,
        lr: float,
        transformer_config: TransformerConfig,
        hidden_dim: int,
        freeze_transformer: bool,
        tweet_max_len: int,
        test_as_second_val_loader: bool = False,
        join_token: str = " ",
    ):
        """
        Same args as MovementPredictor except:
            join_token (str): Token with which tweets are joined together
        """
        super().__init__(
            optim,
            lr,
            transformer_config,
            hidden_dim,
            freeze_transformer,
            tweet_max_len,
            test_as_second_val_loader,
        )
        self.join_token = join_token

    def forward(self, model_input: List[List[Tweet]]):
        # TODO do something smarter - take beginning of each tweet??

        # join tweet texts of a movement together
        tweets = [
            self.join_token.join([tweet.text for tweet in tweets])
            for tweets in model_input
        ]

        tweets_encd = self.tokenizer(
            tweets,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tweet_max_len,
            truncation=True,
            return_offsets_mapping=True,
        ).to(self.device)

        offset_mapping = tweets_encd.pop("offset_mapping")
        self._log_portion_used(offset_mapping, tweets)

        tweet_sentiment = self.transformer(**tweets_encd).logits

        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)

    def _log_portion_used(self, offset_mapping: torch.Tensor, tweets: List[str]):
        """Logs which portions of the tweets are used, i.e., not truncated"""

        used = offset_mapping.view(offset_mapping.size(0), -1).max(dim=1).values
        total = torch.tensor([len(t) for t in tweets], device=self.device)
        portion_used = used / total
        logger.debug(f"{portion_used = }")
