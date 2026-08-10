import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import faiss
import numpy as np

EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "embeddings.npy"


class faissIndex:
    def __init__(self, embeddings):
        self.embeddings = np.load(embeddings).astype("float32")
        self.index = None

    def create_index(self):
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"FAISS index created with {self.index.ntotal} vectors.")

    def save_index(self, output_path):
        index_path = os.path.join(output_path, "faiss.index")
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to {index_path}")


if __name__ == "__main__":
    output_dir = os.path.join(PROJECT_ROOT, "index")
    faiss_index = faissIndex(EMBEDDINGS_PATH)
    faiss_index.create_index()
    faiss_index.save_index(output_dir)
