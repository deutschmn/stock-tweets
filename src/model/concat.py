from src.model.base import MovementPredictor


class ConcatMovementPredictor(MovementPredictor):
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
        join_token: str = " ",
    ):
        """
        Same args as MovementPredictor except:
            join_token (str): Token with which tweets are joined together
        """
        super().__init__(
            optim,
            lr,
            classify_threshold_up,
            classify_threshold_down,
            transformer_model,
            transformer_out,
            hidden_dim,
            freeze_transformer,
            tweet_max_len,
        )
        self.join_token = join_token

    def forward(self, model_input):
        # TODO do something smarter - take beginning of each tweet??

        # join tweets of a movement together, ignore followers
        tweets = [self.join_token.join(tweets) for tweets, _ in model_input]

        tweets_encd = self.tokenizer(
            tweets,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tweet_max_len,
            truncation=True,
        ).to(self.device)

        tweet_sentiment = self.transformer(**tweets_encd).logits

        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)
