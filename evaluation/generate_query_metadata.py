import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json

from api.retrievalEngine import RetrievalEngine
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import METADATA_PATH, get_classifier_kwargs, get_engine_kwargs_with_metadata
from embeddings.buildMetadata import MetadataBuilder

QUERY_DIR = PROJECT_ROOT / "dataset/evaluation_images"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/query_metadata.csv"

fields = [
    "query_filename",
    "predicted_category",
    "predicted_group",
    "predicted_primary_color",
    "predicted_secondary_color",
    "predicted_is_multicolor",
    "predicted_color_distribution",
    "predicted_pattern",
    "predicted_sleeve",
    "predicted_sleeve",
    "predicted_neckline",
    "predicted_style",
    "status",
    "error",
]


def main():
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    image_paths = sorted(
        path
        for path in QUERY_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] {image_path.name}")
            try:
                _, _, _, query_embedding = engine.search(str(image_path), k=1)
                metadata = builder.extract_all_metadata(query_embedding, str(image_path))
                writer.writerow(
                    {
                        "query_filename": image_path.name,
                        "predicted_category": metadata.get("category"),
                        "predicted_group": metadata.get("group"),
                        "predicted_primary_color": metadata.get("color"),
                        "predicted_secondary_color": metadata.get("secondary_color", ""),
                        "predicted_is_multicolor": metadata.get("is_multicolor"),
                        "predicted_color_distribution": json.dumps(
                            metadata.get("color_distribution")
                        ),
                        "predicted_pattern": metadata.get("pattern", ""),
                        "predicted_sleeve": metadata.get("sleeve", ""),
                        "predicted_neckline": metadata.get("neckline", ""),
                        "predicted_style": metadata.get("style", ""),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as error:
                print(f"Error: {error}")
                writer.writerow(
                    {
                        "query_filename": image_path.name,
                        "status": "error",
                        "error": str(error),
                    }
                )
            file.flush()
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
