# ingest/embed_and_index.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

MODEL_NAME = "all-MiniLM-L6-v2"  # fast and good
INDEX_FILE = "data/faiss.index"
META_FILE = "data/meta.jsonl"

def build_index(jsonl_path="data/attributes.jsonl"):
    model = SentenceTransformer(MODEL_NAME)
    docs = []
    texts = []
    metas = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)    
            docs.append(d)
            # Compose a single searchable text for the attribute
            text = f"{d['attribute']} | type: {d['type']} | {d['description']}"
            texts.append(text)
            metas.append(d)

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # persist index
    faiss.write_index(index, INDEX_FILE)
    # save meta list in same order as vectors
    with open(META_FILE, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved index ({len(metas)} vectors) and metadata.")
    return INDEX_FILE, META_FILE

if __name__ == "__main__":
    build_index()
