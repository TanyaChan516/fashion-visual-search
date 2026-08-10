import os
import pickle
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "index" / "faiss.index"
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "embeddings.npy"
IMAGE_PATHS_PATH = PROJECT_ROOT / "embeddings" / "image_paths.pkl"
METADATA_PATH = PROJECT_ROOT / "embeddings" / "metadata.pkl"


def main():
    with open(IMAGE_PATHS_PATH, "rb") as file:
        image_paths = pickle.load(file)

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(INDEX_PATH)

    print("Image paths:", len(image_paths))
    print("Metadata:", len(metadata))
    print("Embeddings:", embeddings.shape[0])
    print("FAISS:", index.ntotal)

    for idx in [1473, 1665, 4878]:
        print(f"\nIndex {idx}")
        print("Current path:", image_paths[idx])
        print("Metadata path:", metadata[idx].get("image_path"))
        print("Color:", metadata[idx].get("color"))
        print("Category:", metadata[idx].get("category"))

        same = os.path.abspath(image_paths[idx]) == os.path.abspath(
            metadata[idx].get("image_path", "")
        )
        print("Paths aligned:", same)


if __name__ == "__main__":
    main()
