import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "evaluation/judgments.csv"
PER_QUERY_CSV = PROJECT_ROOT / "evaluation/metrics_per_query.csv"
SUMMARY_CSV = PROJECT_ROOT / "evaluation/metrics_summary.csv"

KS_PRECISION_HIT = [1, 5, 10, 20]
KS_NDCG = [5, 10, 20]
RELEVANT_THRESHOLD = 2
MAX_RANK = 20


def dcg(grades):
    return sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(grades, start=1)
    )


def ndcg_at_k(ranked_grades, all_query_grades, k):
    actual = ranked_grades[:k]
    ideal = sorted(all_query_grades, reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score > 0 else 0.0


def get_rank_columns(df):
    faiss = "faiss_rank" if "faiss_rank" in df.columns else "raw_rank"
    rerank = "rerank_rank" if "rerank_rank" in df.columns else "reranked_rank"
    for col in [faiss, rerank, "query_id", "relevance"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return faiss, rerank


def ranking_grades(query_df, rank_col):
    ranked = query_df.dropna(subset=[rank_col]).copy()
    ranked[rank_col] = pd.to_numeric(ranked[rank_col], errors="raise")
    ranked = ranked.sort_values(rank_col)
    if len(ranked) < MAX_RANK:
        raise ValueError(
            f"{query_df['query_id'].iloc[0]} has only {len(ranked)} items in "
            f"{rank_col}; expected {MAX_RANK}."
        )
    return ranked["relevance"].astype(int).tolist()[:MAX_RANK]


def calculate_metrics(ranked_grades, all_query_grades):
    binary = [1 if grade >= RELEVANT_THRESHOLD else 0 for grade in ranked_grades]
    metrics = {}
    for k in KS_PRECISION_HIT:
        top = binary[:k]
        metrics[f"Precision@{k}"] = sum(top) / k
        metrics[f"Hit@{k}"] = 1.0 if any(top) else 0.0
    first_relevant = next((rank for rank, value in enumerate(binary, start=1) if value), None)
    metrics["MRR"] = 1.0 / first_relevant if first_relevant else 0.0
    for k in KS_NDCG:
        metrics[f"nDCG@{k}"] = ndcg_at_k(ranked_grades, all_query_grades, k)
    return metrics


def main():
    df = pd.read_csv(INPUT_CSV)
    faiss_col, rerank_col = get_rank_columns(df)
    used = df[df[faiss_col].notna() | df[rerank_col].notna()].copy()
    used["relevance"] = pd.to_numeric(used["relevance"], errors="coerce")
    missing = used[used["relevance"].isna()]
    if not missing.empty:
        print("\nMissing relevance grades:")
        print(
            missing[["query_id", "query_filename", "product_filename"]].to_string(
                index=False
            )
        )
        raise ValueError(f"\nFill the {len(missing)} missing relevance grade(s) first.")

    if not used["relevance"].between(0, 3).all():
        raise ValueError("Relevance grades must be 0, 1, 2, or 3.")

    per_query_rows = []
    for query_id, query_df in used.groupby("query_id", sort=True):
        all_grades = (
            query_df.drop_duplicates("product_filename")["relevance"]
            .astype(int)
            .tolist()
        )
        for method, rank_col in [("FAISS", faiss_col), ("Reranked", rerank_col)]:
            grades = ranking_grades(query_df, rank_col)
            metrics = calculate_metrics(grades, all_grades)
            per_query_rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    **metrics,
                }
            )

    per_query = pd.DataFrame(per_query_rows)
    per_query.to_csv(PER_QUERY_CSV, index=False)

    metric_cols = [
        "Precision@1",
        "Precision@5",
        "Precision@10",
        "Precision@20",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@20",
        "MRR",
        "nDCG@5",
        "nDCG@10",
        "nDCG@20",
    ]

    means = per_query.groupby("method")[metric_cols].mean()

    summary_rows = []
    for metric in metric_cols:
        faiss = means.loc["FAISS", metric]
        reranked = means.loc["Reranked", metric]
        summary_rows.append(
            {
                "metric": metric,
                "FAISS": faiss,
                "Reranked": reranked,
                "delta": reranked - faiss,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\nQueries evaluated: {per_query['query_id'].nunique()}")
    print(f"Relevant threshold: relevance >= {RELEVANT_THRESHOLD}")
    print(f"MRR evaluated over judged top-{MAX_RANK}")
    print("\n=== Average metrics across queries ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved:\n  {PER_QUERY_CSV}\n  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
