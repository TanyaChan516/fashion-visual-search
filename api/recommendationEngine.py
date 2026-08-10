import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.retrievalEngine import RetrievalEngine
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import get_classifier_kwargs, get_engine_kwargs, get_recommender_kwargs
from embeddings.buildMetadata import MetadataBuilder


class RecommendationEngine:
    def __init__(
        self,
        metadata_path,
        group_recommendations,
        color_recommendations,
        style_recommendations,
    ):
        with open(metadata_path, "rb") as file:
            self.metadata = pickle.load(file)
        self.group_recommendations = group_recommendations
        self.color_recommendations = color_recommendations
        self.style_recommendations = style_recommendations

    def score_pattern(self, query_pattern, item_pattern):
        if query_pattern != "solid":
            return 1.0 if item_pattern == "solid" else 0.0
        return 0.5

    def recommend(self, query_metadata, k=8):
        recommendations = []
        for item in self.metadata:
            group_score = (
                1.0
                if item["group"]
                in self.group_recommendations[query_metadata["group"]]
                else 0.0
            )
            if group_score > 0:
                color_score = (
                    1.0
                    if item["color"]
                    in self.color_recommendations[query_metadata["color"]]
                    else 0.0
                )
                style_score = (
                    1.0
                    if item["style"]
                    in self.style_recommendations[query_metadata["style"]]
                    else 0.0
                )
                occasion_score = (
                    1.0 if item["occasion"] == query_metadata["occasion"] else 0.0
                )
                pattern_score = self.score_pattern(query_metadata["pattern"], item["pattern"])
                final_score = (
                    0.35 * group_score
                    + 0.25 * style_score
                    + 0.2 * occasion_score
                    + 0.1 * color_score
                    + 0.1 * pattern_score
                )

                item = item.copy()
                item["recommendation_score"] = final_score
                recommendations.append(item)
        return sorted(
            recommendations,
            key=lambda x: x["recommendation_score"],
            reverse=True,
        )[:k]


if __name__ == "__main__":
    query_image_path = os.path.join(
        PROJECT_ROOT, "dataset", "evaluation_images", "image_11000.png"
    )
    engine = RetrievalEngine(**get_engine_kwargs())
    query_embedding = engine.extract_image_features(query_image_path)
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(
        classifier, metadata_path=os.path.join(PROJECT_ROOT, "embeddings", "metadata.pkl")
    )
    query_metadata = builder.extract_all_metadata(query_embedding, query_image_path)
    print(f"Query Metadata: {query_metadata}")
    recommender = RecommendationEngine(**get_recommender_kwargs())
    recommendations = recommender.recommend(query_metadata=query_metadata)
    print(f"Recommendations: {recommendations}")
