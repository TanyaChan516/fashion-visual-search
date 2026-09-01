import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

from config import PROJECT_ROOT, get_engine_kwargs


class RetrievalEngine:
    def __init__(
        self,
        index_path,
        embeddings_path,
        image_paths_path,
        model_name,
        tokenizer_name,
        device,
        metadata_path=None,
    ):
        # ViT-gopt has a large temporary allocation peak while TIMM constructs
        # and initializes the model. Build it before keeping retrieval data in
        # memory so those allocations do not overlap.
        siglip_threads = int(os.environ.get("SIGLIP_TORCH_THREADS", "1"))
        torch.set_num_threads(siglip_threads)
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.device = device
        self.model.to(self.device)
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(embeddings_path).astype("float32", copy=False)
        with open(image_paths_path, "rb") as f:
            self.image_paths = pickle.load(f)

        self.metadata = None
        if metadata_path is not None:
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def get_stored_embedding(self, index):
        return self.embeddings[index : index + 1]

    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(img.to(self.device))
            norm_features = F.normalize(image_features, dim=-1)
            return norm_features.cpu().numpy().astype("float32")
        
    def extract_text_features(self, text_query):
        tokens = self.tokenizer(text_query)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens.to(self.device))
            text_features = F.normalize(text_features, dim=-1)
        return text_features.cpu().numpy().astype("float32")
    
    def search(self, query_image_path, k=5):
        query_feature = self.extract_image_features(query_image_path)
        score, index = self.index.search(query_feature, k)
        results = [self.image_paths[i] for i in index[0]]
        return index[0], results, score[0], query_feature

    def sigmoid_scale(self, scores, center, sharpness):
        return 1 / (1 + np.exp(-sharpness * (scores - center)))

    def text_search(self, text_query, k=5):
        text_embedding = self.extract_text_features(text_query)
        scores = (self.embeddings @ text_embedding.T).squeeze()
        indices = np.argsort(scores)[::-1][:k]
        text_scores = self.sigmoid_scale(scores[indices], center=0.06, sharpness=40)
        results = [self.image_paths[i] for i in indices]
        return indices, results, text_scores, text_embedding

    def multimodal_search(self, query_image_path, text_query, k=5):
        indices, results, image_scores, query_image_embedding = self.search(
            query_image_path, k=k
        )
        text_embedding = self.extract_text_features(text_query)
        candidate_embeddings = self.embeddings[indices]
        text_scores = self.sigmoid_scale(
            (candidate_embeddings @ text_embedding.T).squeeze(),
            center=0.06,
            sharpness=40,
        )
        return (
            indices,
            results,
            image_scores,
            text_scores,
            query_image_embedding,
            text_embedding,
        )

    def enrich_search_result(self, indices, image_scores=None, text_scores=None):
        if self.metadata is None:
            raise RuntimeError(
                "Metadata is not loaded. Cannot enrich search results without metadata."
            )
        results_info = []
        for i, idx in enumerate(indices):
            data = self.metadata[idx]
            info = data.copy()
            if image_scores is not None:
                info["image_score"] = float(image_scores[i])
            if text_scores is not None:
                info["text_score"] = float(text_scores[i])
            results_info.append(info)
        return results_info

    def visualize_result(self, query, results, scores):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(Image.open(query).convert("RGB"))
        plt.title("Query Image")
        plt.axis("off")
        for i, result in enumerate(results):
            img = Image.open(result).convert("RGB")
            plt.subplot(1, len(results) + 1, i + 2)
            plt.imshow(img)
            plt.title(f"Result {i + 1} (Score: {scores[i]:.2f})")
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    query_image_path = os.path.join(
        PROJECT_ROOT, "dataset", "evaluation_images", "image_11000.png"
    )
    text_query = "white dress"
    engine = RetrievalEngine(**get_engine_kwargs())
    # _, results, scores, _ = engine.search(query_image_path)
    _, results, scores, _ = engine.text_search(text_query)
    # _, results, image_scores, text_scores, _, _ = engine.multimodal_search(
    #     query_image_path, text_query, k=5
    # )
    print("Top 5 results:")
    for result, score in zip(results, scores):
        print(f"Image: {result}, Similarity: {score}")
    engine.visualize_result(query_image_path, results, scores)
