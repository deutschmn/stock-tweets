from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class MovementDataset(Dataset):
    def __init__(self, movements, transformer_model, tweet_maxlen=100):
        super()
        self.movements = movements
        
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.tweet_maxlen = tweet_maxlen

    def __len__(self):
        return len(self.movements)

    def __getitem__(self, idx):
        m = self.movements[idx]
        encd_tweets = self.tokenizer(list(m.tweets["text"]), 
                            return_tensors="pt", 
                            padding="max_length", 
                            max_length=self.tweet_maxlen,
                            truncation=True)

        return encd_tweets, list(m.tweets["user_followers"]), m.price["movement percent"]

    def coll_samples(batch):
        tweets = list(map(lambda x: x[0:2], batch))
        prices = torch.stack(list(map(lambda x: torch.tensor(x[-1]), batch))).float()
        return tweets, prices


def classify_movement(m, thres_up, thres_down):
    classes = torch.zeros_like(m)
    classes[m > thres_up] = 1
    classes[m < thres_down] = 0
    return classes
