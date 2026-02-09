import torch
from sentence_transformers import SentenceTransformer
import numpy as np

import config


class Embedder:
    def __init__(self, model_name=config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model.to(config.DEVICE)

    def encode(self, texts):
        with torch.no_grad():
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_single(self, text):
        return self.encode([text])[0]
