import faiss
import numpy as np

import config


class VectorStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.documents = []

    def add(self, embeddings, documents):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)

    def search(self, query_emb, top_k=config.TOP_K):
        q = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

    def clear(self):
        self.index.reset()
        self.documents = []
