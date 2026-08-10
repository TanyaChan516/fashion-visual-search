import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import pickle
import time

import numpy as np

from api.retrievalEngine import RetrievalEngine
from api.searchReranker import SearchReranker
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import METADATA_PATH, get_classifier_kwargs, get_engine_kwargs_with_metadata
from embeddings.buildMetadata import MetadataBuilder

IMAGE_PATHS_FILE = PROJECT_ROOT / "embeddings/image_paths.pkl"
DETAIL_CSV = PROJECT_ROOT / "evaluation/latency_results.csv"
SUMMARY_CSV = PROJECT_ROOT / "evaluation/latency_summary.csv"

NUM_QUERIES = 1000
WARMUP_QUERIES = 10
CANDIDATE_K = 50
TOP_K = 20
SEED = 42


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1_000_000


def run_query(image_path, engine, builder, reranker):
    # 1. Image embedding + FAISS retrieval
    search_out, search_ms = timed(lambda: engine.search(str(image_path), k=CANDIDATE_K))
    indices, _, image_scores, query_embedding = search_out

    # 2. Query metadata used by the reranker
    query_metadata, metadata_ms = timed(
        lambda: builder.extract_all_metadata(query_embedding, str(image_path))
    )

    # 3. Attach stored metadata/scores to FAISS candidates
    faiss_result, enrich_ms = timed(
        lambda: engine.enrich_search_result(indices, image_scores=image_scores)
    )

    # 4. Rerank all 50 candidates and keep final top 20
    _, rerank_ms = timed(
        lambda: reranker.rerank(
            [dict(result) for result in faiss_result],
            query_category=query_metadata.get("category"),
            query_group=query_metadata.get("group"),
            query_color=query_metadata,
        )[:TOP_K]
    )

    raw_pipeline_ms = search_ms + enrich_ms
    reranked_pipeline_ms = raw_pipeline_ms + metadata_ms + rerank_ms

    return {
        "search_ms": search_ms,
        "enrich_ms": enrich_ms,
        "raw_pipeline_ms": raw_pipeline_ms,
        "metadata_ms": metadata_ms,
        "rerank_ms": rerank_ms,
        "reranked_pipeline_ms": reranked_pipeline_ms,
        "rerank_overhead_ms": metadata_ms + rerank_ms,
    }


def main():
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    reranker = SearchReranker()

    with IMAGE_PATHS_FILE.open("rb") as f:
        stored_paths = pickle.load(f)

    image_paths = []
    seen = set()
    for stored_path in stored_paths:
        resolved = (
            Path(stored_path)
            if Path(stored_path).is_absolute()
            else PROJECT_ROOT / stored_path
        )
        key = str(resolved)
        if key not in seen and resolved.exists():
            seen.add(key)
            image_paths.append(resolved)
    n = min(NUM_QUERIES, len(image_paths))
    benchmark_images = image_paths[:n]
    warmup_images = benchmark_images[: min(WARMUP_QUERIES, n)]

    print("\nWarming up...")
    for image_path in warmup_images:
        run_query(image_path, engine, builder, reranker)

    rows = []
    print("\nBenchmarking...")
    for i, image_path in enumerate(benchmark_images, start=1):
        row = {
            "query_number": i,
            "query_filename": image_path.name,
            "status": "ok",
            "error": "",
        }
        try:
            row.update(run_query(image_path, engine, builder, reranker))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i % 50 == 0 or i == n:
            print(f"{i}/{n}")

    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_number",
        "query_filename",
        "status",
        "error",
        "search_ms",
        "enrich_ms",
        "raw_pipeline_ms",
        "metadata_ms",
        "rerank_ms",
        "reranked_pipeline_ms",
        "rerank_overhead_ms",
    ]
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("All benchmark queries failed.")

    metrics = [
        "search_ms",
        "enrich_ms",
        "raw_pipeline_ms",
        "metadata_ms",
        "rerank_ms",
        "reranked_pipeline_ms",
        "rerank_overhead_ms",
    ]
    summary_rows = []
    for metric in metrics:
        values = np.asarray([row[metric] for row in successful], dtype=float)
        stats = {
            "n": len(values),
            "mean_ms": values.mean(),
            "median_ms": np.median(values),
            "p95_ms": np.percentile(values, 95),
            "min_ms": values.min(),
            "max_ms": values.max(),
            "std_ms": values.std(ddof=1) if len(values) > 1 else 0.0
        }
        summary_rows.append({"metric": metric, **stats})

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "n",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "std_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Latency Summary ===")
    print(f"{'Metric':24} {'Mean':>10} {'Median':>10} {'P95':>10} {'Max':>10}")
    for row in summary_rows:
        print(
            f"{row['metric']:24} {row['mean_ms']:10.2f} "
            f"{row['median_ms']:10.2f} {row['p95_ms']:10.2f} "
            f"{row['max_ms']:10.2f}"
        )
    print(f"\nSuccessful: {len(successful)}/{len(rows)}")
    print(f"Failure rate: {(len(rows) - len(successful)) / len(rows) * 100:.2f}%")
    print(f"\nSaved:\n  {DETAIL_CSV}\n  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
