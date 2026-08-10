import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.retrievalEngine import RetrievalEngine
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import PROJECT_ROOT, get_classifier_kwargs, get_engine_kwargs_with_metadata
from embeddings.buildMetadata import MetadataBuilder


class SearchReranker:
    def __init__(
        self, image_weight=0.5, text_weight=0.2, category_weight=0.2, color_weight=0.1
    ):
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.category_weight = category_weight
        self.color_weight = color_weight

    def category_score(self, result, query_category, query_group):
        if result["category"] == query_category:
            return 1.0
        if result["group"] == query_group:
            return 0.5
        return 0.0
    
    def color_score(self, result, query_metadata):
        query_distribution = query_metadata.get("color_distribution", {})
        result_distribution = result.get("color_distribution", {})
        if query_distribution and result_distribution:
            colors = set(query_distribution) | set(result_distribution)
            overlap = sum(
                min(
                    float(query_distribution.get(color, 0.0)),
                    float(result_distribution.get(color, 0.0)),
                )
                for color in colors
            )
            return float(np.clip(overlap, 0.0, 1.0))
        query_primary = query_metadata.get("color")
        if result.get("color") == query_primary:
            return 1.0
        if result.get("secondary_color") == query_primary:
            return 0.6
        return 0.0

    def rerank(self, results, query_category=None, query_group=None, query_color=None):
        reranked = []
        for result in results:
            image_score = result.get("image_score", 0.0)
            text_score = result.get("text_score", 0.0)
            use_image = image_score > 0.0
            use_text = text_score > 0.0
            category_score = (
                self.category_score(result, query_category, query_group)
                if query_category is not None and query_group is not None
                else 0.0
            )
            color_score = (
                self.color_score(result, query_color)
                if query_color is not None
                else 0.0
            )
            if use_image and use_text:
                final_score = (
                    self.image_weight * image_score
                    + self.text_weight * text_score
                    + self.category_weight * category_score
                    + self.color_weight * color_score
                )
            elif use_image:
                w = self.image_weight + self.category_weight + self.color_weight
                final_score = (
                    self.image_weight / w * image_score
                    + self.category_weight / w * category_score
                    + self.color_weight / w * color_score
                )
            elif use_text:
                final_score = text_score
            else:
                final_score = 0.0
            enriched_result = result.copy()
            enriched_result["category_score"] = category_score
            enriched_result["color_score"] = color_score
            enriched_result["final_score"] = final_score
            reranked.append(enriched_result)
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked


if __name__ == "__main__":
    query_image_path = os.path.join(
        PROJECT_ROOT, "dataset", "evaluation_images", "image_10001.png"
    )
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    indices, _, scores, query_embedding = engine.search(query_image_path, k=5)
    results_info = engine.enrich_search_result(indices, scores)
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    query_category, query_group = classifier.extract_category_and_group(query_embedding)
    builder = MetadataBuilder(
        classifier, metadata_path=os.path.join(PROJECT_ROOT, "embeddings", "metadata.pkl")
    )
    query_color = builder.extract_color(query_image_path)
    reranker = SearchReranker()
    reranked_results = reranker.rerank(
        results_info, query_category, query_group, query_color
    )
    for res in reranked_results:
        print(
            f"Image: {res['image_path']}, Final Score: {res['final_score']:.4f}, "
            f"Image Score: {res['image_score']:.4f}, "
            f"Category Score: {res['category_score']:.4f}, "
            f"Color Score: {res['color_score']:.4f}"
        )
