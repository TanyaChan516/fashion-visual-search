import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.retrievalEngine import RetrievalEngine
from api.searchReranker import SearchReranker
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import METADATA_PATH, get_classifier_kwargs, get_engine_kwargs_with_metadata
from embeddings.buildMetadata import MetadataBuilder


QUERIES_CSV = PROJECT_ROOT / "evaluation/selected_queries.csv"
OUTPUT_CSV = PROJECT_ROOT / "evaluation/judgments.csv"
QUERY_DIR = PROJECT_ROOT / "dataset/evaluation_images"

CANDIDATE_K = 50
TOP_K = 20

FIELDS = [
    "query_id",
    "query_filename",
    "product_filename",
    "product_category",
    "product_primary_color",
    "faiss_rank",
    "rerank_rank",
    "result_source",
    "relevance",
    "notes",
]


def load_existing_labels():
    if not OUTPUT_CSV.exists():
        return {}
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as file:
        return {
            (row["query_id"], row["product_filename"]): (
                row.get("relevance", ""),
                row.get("notes", ""),
            )
            for row in csv.DictReader(file)
        }


def main():
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    reranker = SearchReranker()

    with QUERIES_CSV.open("r", newline="", encoding="utf-8") as file:
        queries = list(csv.DictReader(file))
    existing_labels = load_existing_labels()
    output_rows = []

    for number, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        query_filename = query["query_filename"]
        query_path = QUERY_DIR / query_filename
        if not query_path.exists():
            print(f"Skipped missing file: {query_path}")
            continue
        print(f"[{number}/{len(queries)}] {query_id}: {query_filename}")
        indices, _, image_scores, query_embedding = engine.search(str(query_path), k=CANDIDATE_K)
        query_metadata = builder.extract_all_metadata(query_embedding, str(query_path))
        faiss_results = engine.enrich_search_result(indices, image_scores=image_scores)
        rerank_results = reranker.rerank(
            [dict(result) for result in faiss_results],
            query_category=query_metadata.get("category"),
            query_group=query_metadata.get("group"),
            query_color=query_metadata,
        )

        union = {}
        for rank, result in enumerate(faiss_results[:TOP_K], start=1):
            filename = Path(result["image_path"]).name
            union.setdefault(
                filename,
                {
                    "query_id": query_id,
                    "query_filename": query_filename,
                    "product_filename": filename,
                    "product_category": result.get("category", ""),
                    "product_primary_color": result.get("color", ""),
                    "faiss_rank": "",
                    "rerank_rank": "",
                    "result_source": "",
                    "relevance": "",
                    "notes": "",
                },
            )
            union[filename]["faiss_rank"] = rank

        for rank, result in enumerate(rerank_results[:TOP_K], start=1):
            filename = Path(result["image_path"]).name
            union.setdefault(
                filename,
                {
                    "query_id": query_id,
                    "query_filename": query_filename,
                    "product_filename": filename,
                    "product_category": result.get("category", ""),
                    "product_primary_color": result.get("color", ""),
                    "faiss_rank": "",
                    "rerank_rank": "",
                    "result_source": "",
                    "relevance": "",
                    "notes": "",
                },
            )
            union[filename]["rerank_rank"] = rank

        for row in union.values():
            if row["faiss_rank"] and row["rerank_rank"]:
                row["result_source"] = "both"
            elif row["faiss_rank"]:
                row["result_source"] = "faiss_only"
            else:
                row["result_source"] = "rerank_only"
            saved = existing_labels.get((row["query_id"], row["product_filename"]))
            if saved:
                row["relevance"], row["notes"] = saved
            output_rows.append(row)

        print(f"  {len(union)} unique products to judge")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nSaved {len(output_rows)} rows to {OUTPUT_CSV}")
    print("Fill only the relevance and notes columns.")


if __name__ == "__main__":
    main()
