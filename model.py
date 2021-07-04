from ctypes import ArgumentError

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class MovementPredictor(nn.Module):
    def __init__(
        self,
        transformer_model,
        transformer_out,
        device,
        hidden_dim,
        freeze_transformer,
        attention_input,
    ):
        super().__init__()
        self.transformer_name = transformer_model
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_model
        )

        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False

        self.device = device

        self.attention_input = attention_input

        if attention_input == 'sentiment':
            attention_in_dim = transformer_out
        elif attention_input == 'followers':
            attention_in_dim = 1
        elif attention_input == 'both':
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


    def forward_movement(self, tweets):
        tweets_encd = map(lambda x: x.to(self.device), tweets[0].values())
        tweets_followers = (
            torch.tensor(tweets[1], dtype=torch.float).unsqueeze(dim=-1).to(self.device)
        )

        tweet_reps = self.transformer(*tweets_encd).logits

        if self.attention_input == 'sentiment':
            attention_in = tweet_reps
        elif self.attention_input == 'followers':
            attention_in = tweets_followers
        elif self.attention_input == 'both':
            attention_in = torch.cat([tweet_reps, tweets_followers], dim=-1)
        else:
            raise ArgumentError(f"Unknown attention input {attention_input}")
        
        attention_weights = self.attention(
            attention_in
        )  
        return torch.mm(tweet_reps.T, attention_weights).squeeze()

    def forward(self, tweets):
        tweet_sentiment = torch.stack(list(map(self.forward_movement, tweets)))
        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)
