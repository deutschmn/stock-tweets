from transformers import AutoModelForSequenceClassification
import torch
from torch import nn

class MovementPredictor(nn.Module):
    def __init__(self, transformer_model, transformer_out, device, hidden_dim, freeze_transformer):
        super().__init__()
        self.transformer_name = transformer_model
        self.transformer = AutoModelForSequenceClassification.from_pretrained(transformer_model)
        
        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False
        
        self.device = device

        self.follower_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU(),
            nn.Softmax(dim=0)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(transformer_out, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_movement(self, tweets):
        tweets_encd = map(lambda x: x.to(self.device), tweets[0].values())
        tweets_followers = torch.tensor(tweets[1], dtype=torch.float).unsqueeze(dim=-1).to(self.device)

        tweet_reps = self.transformer(*tweets_encd).logits
        follower_reps = self.follower_layer(tweets_followers)
        return torch.mm(tweet_reps.T, follower_reps).squeeze()

    def forward(self, tweets):
        tweet_sentiment = torch.stack(list(map(self.forward_movement, tweets)))
        return self.sentiment_classifier(tweet_sentiment).squeeze(dim=-1)
