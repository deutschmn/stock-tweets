class TransformerConfig:
    def __init__(self, hugging_face_name: str, out_dim: int):
        self.hugging_face_name = hugging_face_name
        self.out_dim = out_dim


class BertBaseMultilingualUncasedSentiment(TransformerConfig):
    # 12-layer, 768-hidden, 12-heads, 168M parameters
    def __init__(self):
        super().__init__(
            hugging_face_name="nlptown/bert-base-multilingual-uncased-sentiment",
            out_dim=5,
        )


class TwitterRobertaBaseSentiment(TransformerConfig):
    # 12-layer, 768-hidden, 12-heads, 125M parameters
    def __init__(self):
        super().__init__(
            hugging_face_name="cardiffnlp/twitter-roberta-base-sentiment",
            out_dim=3,
        )


class AlbertBase(TransformerConfig):
    # 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters
    def __init__(self):
        super().__init__(hugging_face_name="albert-base-v2", out_dim=2)
