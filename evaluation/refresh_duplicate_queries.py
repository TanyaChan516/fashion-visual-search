import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.retrievalEngine import RetrievalEngine
from api.searchReranker import SearchReranker
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import METADATA_PATH, get_classifier_kwargs, get_engine_kwargs_with_metadata
from embeddings.buildMetadata import MetadataBuilder

EVAL = PROJECT_ROOT / "evaluation"
JUDGMENTS = EVAL / "judgments.csv"
QUERY_DIR = PROJECT_ROOT / "dataset/evaluation_images"
AFFECTED = {"q001", "q008", "q009", "q014", "q016"}
TOP_K, CANDIDATE_K = 20, 50


def main():
    old = pd.read_csv(JUDGMENTS)
    queries = pd.read_csv(EVAL / "selected_queries.csv")
    duplicates = pd.read_csv(EVAL / "duplicates.csv")
    duplicate_map = dict(
        zip(duplicates.duplicate_filename, duplicates.canonical_filename)
    )

    def canonical_name(name):
        return duplicate_map.get(name, name)

    grades = {}
    for _, row in old.dropna(subset=["relevance"]).iterrows():
        grades[(row.query_id, canonical_name(row.product_filename))] = row.relevance

    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    reranker = SearchReranker()

    def unique_top(results):
        output = []
        seen = set()
        for result in results:
            name = canonical_name(Path(result["image_path"]).name)
            if name in seen:
                continue
            seen.add(name)
            output.append((name, result))
            if len(output) == TOP_K:
                break
        return output

    new_rows = []
    for _, query in queries[queries.query_id.isin(AFFECTED)].iterrows():
        query_path = QUERY_DIR / query.query_filename
        indices, _, image_scores, query_embedding = engine.search(
            str(query_path), k=CANDIDATE_K
        )
        query_metadata = builder.extract_all_metadata(
            query_embedding, str(query_path)
        )
        faiss_results = engine.enrich_search_result(
            indices, image_scores=image_scores
        )
        rerank_results = reranker.rerank(
            [dict(result) for result in faiss_results],
            query_category=query_metadata.get("category"),
            query_group=query_metadata.get("group"),
            query_color=query_metadata,
        )

        rows = {}
        ranked_results = [
            (unique_top(faiss_results), "faiss_rank"),
            (unique_top(rerank_results), "rerank_rank"),
        ]
        for results, rank_col in ranked_results:
            for rank, (name, result) in enumerate(results, 1):
                rows.setdefault(
                    name,
                    {
                        "query_id": query.query_id,
                        "query_filename": query.query_filename,
                        "product_filename": name,
                        "product_category": result.get("category", ""),
                        "product_primary_color": result.get("color", ""),
                        "faiss_rank": "",
                        "rerank_rank": "",
                        "result_source": "",
                        "relevance": grades.get((query.query_id, name), ""),
                    },
                )[rank_col] = rank

        for row in rows.values():
            if row["faiss_rank"] and row["rerank_rank"]:
                row["result_source"] = "both"
            elif row["faiss_rank"]:
                row["result_source"] = "faiss_only"
            else:
                row["result_source"] = "rerank_only"
            new_rows.append(row)

    shutil.copy2(JUDGMENTS, EVAL / "judgments_before_dedup.csv")
    kept = old[~old.query_id.isin(AFFECTED)]
    updated = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(JUDGMENTS, index=False)

    new_items = updated[
        updated.query_id.isin(AFFECTED)
        & (
            updated.relevance.isna()
            | (updated.relevance.astype(str).str.strip() == "")
        )
    ]
    new_items.to_csv(EVAL / "new_items_to_grade.csv", index=False)

    print("Refreshed:", ", ".join(sorted(AFFECTED)))
    print("New replacement items to grade:", len(new_items))


if __name__ == "__main__":
    main()
