# vector_store/faiss_store.py
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "data/faiss.index"
META_FILE = "data/meta.jsonl"

class FaissStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metas = [json.loads(l) for l in f]

    def search(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: 
                continue
            meta = self.metas[idx]
            results.append({"meta": meta, "score": float(score)})
        return results
