import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from api.retrievalEngine import RetrievalEngine
from config import ATTRIBUTE_CONFIG, CATEGORY_MAPPING, PROMPT_TEMPLATE, get_engine_kwargs


class ZeroShotCategoryClassifier:
    def __init__(self, engine, category_mapping=None, attribute_mapping=None):
        self.engine = engine
        self.category_mapping = category_mapping
        self.attribute_mapping = attribute_mapping
        self.category_embeddings = {}
        self.attribute_embeddings = {}
        self.result_info = {"query": None, "results": []}

    def extract_text_embedding(self, label, prompt_templates):
        prompts = [template.format(label) for template in prompt_templates]
        tokens = self.engine.tokenizer(prompts).to(self.engine.device)
        with torch.no_grad():
            text_embedding = self.engine.model.encode_text(tokens)
            text_embedding = F.normalize(text_embedding, dim=-1).mean(axis=0)
            text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding.cpu().numpy().astype("float32")

    def create_embeddings(self, mode="category"):
        if mode == "category":
            print("Building category embeddings...")
            for label in self.category_mapping.keys():
                self.category_embeddings[label] = self.extract_text_embedding(label, PROMPT_TEMPLATE)
            print("Finished building category embeddings.")
        if mode == "attribute":
            print("Building attribute embeddings...")
            for attr_name, config in self.attribute_mapping.items():
                self.attribute_embeddings[attr_name] = {}
                for label in config["labels"]:
                    self.attribute_embeddings[attr_name][label] = self.extract_text_embedding(label, config["prompts"])
            print("Finished building attribute embeddings.")

    def classify(self, image_embedding, label_embeddings):
        similarities = {}
        for label, text_embedding in label_embeddings.items():
            similarities[label] = float(image_embedding @ text_embedding)
        return max(similarities, key=similarities.get)

    def extract_category_and_group(self, image_embedding):
        category = self.classify(image_embedding, self.category_embeddings)
        group = self.category_mapping.get(category)
        return category, group

    def extract_attribute(self, image_embedding, attr_name):
        return self.classify(image_embedding,  self.attribute_embeddings[attr_name])

    def category_score(self, query_category, result_category):
        if query_category == result_category:
            return 1.0
        if self.category_mapping.get(query_category) == self.category_mapping.get(result_category):
            return 0.5
        return 0.0

    def attribute_score(self, query_attr, result_attr):
        return 1.0 if query_attr == result_attr else 0.0

    def score(self, query_embedding, result_index, attr_name=None):
        scores = []
        self.result_info = {"query": None, "results": []}
        if attr_name is None:
            query_category, query_group = self.extract_category_and_group(query_embedding)
            for idx in result_index:
                result_embedding = self.engine.get_stored_embedding(idx)
                result_category, result_group = self.extract_category_and_group(result_embedding)
                self.result_info["query"] = (query_category, query_group)
                self.result_info["results"].append((result_category, result_group))
                score = self.category_score(query_category, result_category)
                scores.append(score)
        else:
            query_attr = self.extract_attribute(query_embedding, attr_name)
            for idx in result_index:
                result_embedding = self.engine.get_stored_embedding(idx)
                result_attr = self.extract_attribute(result_embedding, attr_name)
                self.result_info["query"] = query_attr
                self.result_info["results"].append(result_attr)
                score = self.attribute_score(query_attr, result_attr)
                scores.append(score)
        return scores


if __name__ == "__main__":
    engine = RetrievalEngine(**get_engine_kwargs())

    classifier = ZeroShotCategoryClassifier(
        engine,
        category_mapping=CATEGORY_MAPPING,
        attribute_mapping=ATTRIBUTE_CONFIG,
    )
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")

    query_image_path = "dataset/evaluation_images/image_11000.png"
    indices, _, _, query_embedding = engine.search(query_image_path)
    # match_score = classifier.score(query_embedding, indices)
    match_score = classifier.score(query_embedding, indices, attr_name="sleeve")
    print(f"Mean Category Match Score: {np.mean(match_score):.4f}")
    print(f"Category Result: {classifier.result_info}")
