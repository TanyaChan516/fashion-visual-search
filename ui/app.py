import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image

from api.recommendationEngine import RecommendationEngine
from api.retrievalEngine import RetrievalEngine
from api.searchReranker import SearchReranker
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import (
    ATTRIBUTE_CONFIG,
    CATEGORY_MAPPING,
    METADATA_PATH,
    get_classifier_kwargs,
    get_engine_kwargs_with_metadata,
    get_recommender_kwargs,
)
from embeddings.buildMetadata import MetadataBuilder


st.set_page_config(layout="wide")
st.title("Fashion Visual Search")


@st.cache_resource
def load_engine():
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    reranker = SearchReranker()
    recommender = RecommendationEngine(**get_recommender_kwargs())
    return engine, classifier, builder, reranker, recommender

engine, classifier, builder, reranker, recommender = load_engine()


def get_attribute_caption(item):
    parts = []
    material = item.get("material")
    if material:
        parts.append(material)
    if item.get("group") == "Tops":
        neckline = item.get("neckline")
        sleeve = item.get("sleeve")
        if neckline and sleeve:
            parts.append(neckline)
            parts.append(sleeve)
    return " • ".join(parts)


groups = ["All"] + sorted(set(CATEGORY_MAPPING.values()))
selected_group = st.sidebar.selectbox("Group", groups)
if selected_group == "All":
    categories = ["All"] + sorted(CATEGORY_MAPPING.keys())
else:
    categories = ["All"] + sorted(
        [cat for cat, grp in CATEGORY_MAPPING.items() if grp == selected_group]
    )
selected_category = st.sidebar.selectbox("Category", categories)
selected_color = st.sidebar.selectbox(
    "Color",
    [
        "All",
        "black",
        "white",
        "grey",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
    ],
)
selected_style = st.sidebar.selectbox(
    "Style", ["All"] + sorted(set(ATTRIBUTE_CONFIG["style"]["labels"]))
)
selected_occasion = st.sidebar.selectbox(
    "Occasion", ["All"] + sorted(set(ATTRIBUTE_CONFIG["occasion"]["labels"]))
)
selected_material = st.sidebar.selectbox(
    "Material", ["All"] + sorted(set(ATTRIBUTE_CONFIG["material"]["labels"]))
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
text_query = st.text_input("Optional text query", "")
faiss_results = []
rerank_results = []
final_results = []
recommendations = []

if not uploaded_file and not text_query:
    st.info("Upload an image, enter a text query, or both.")
elif text_query and not uploaded_file:
    indices, _, text_scores, _ = engine.text_search(text_query, k=12)
    results_info = engine.enrich_search_result(indices, text_scores=text_scores)
    final_results = reranker.rerank(results_info)
elif uploaded_file and not text_query:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Query Image")
    st.image(query_image, width=300)
    tmp_path = os.path.join(PROJECT_ROOT, "tmp_query.png")
    query_image.save(tmp_path)
    
    indices, _, image_scores, query_embedding = engine.search(tmp_path, k=50)
    query_metadata = builder.extract_all_metadata(query_embedding, tmp_path)
    recommendations = recommender.recommend(query_metadata=query_metadata, k=4)
    st.markdown(
        f"### {query_metadata['category'].title()} ({query_metadata['color']})"
    )
    st.caption(
        f"{query_metadata['material']} • {query_metadata['neckline']} • "
        f"{query_metadata['sleeve']}"
    )
    st.caption(
        f"{query_metadata['pattern']} • {query_metadata['structure']} • "
        f"{query_metadata['style']}"
    )
    st.caption(f"Best for: {query_metadata['occasion']}")

    faiss_results = engine.enrich_search_result(indices, image_scores=image_scores)
    rerank_results = reranker.rerank(
        [dict(result) for result in faiss_results],
        query_category=query_metadata["category"],
        query_group=query_metadata["group"],
        query_color=query_metadata,
    )
elif uploaded_file and text_query:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Query Image")
    st.image(query_image, width=300)
    tmp_path = os.path.join(PROJECT_ROOT, "tmp_query.png")
    query_image.save(tmp_path)
    
    indices, _, image_scores, text_scores, query_image_embedding, _ = (
        engine.multimodal_search(tmp_path, text_query, k=50)
    )
    query_metadata = builder.extract_all_metadata(query_image_embedding, tmp_path)
    recommendations = recommender.recommend(query_metadata=query_metadata, k=4)
    st.markdown(
        f"### {query_metadata['category'].title()} ({query_metadata['color']})"
    )
    st.caption(
        f"{query_metadata['material']} • {query_metadata['neckline']} • "
        f"{query_metadata['sleeve']}"
    )
    st.caption(
        f"{query_metadata['pattern']} • {query_metadata['structure']} • "
        f"{query_metadata['style']}"
    )
    st.caption(f"Best for: {query_metadata['occasion']}")
    st.markdown(f"**Text Query:** {text_query}")
    faiss_results = engine.enrich_search_result(indices, image_scores=image_scores)
    rerank_results = reranker.rerank(
        [dict(result) for result in faiss_results],
        query_category=query_metadata["category"],
        query_group=query_metadata["group"],
        query_color=query_metadata,
    )

rank_mode = st.sidebar.radio("Ranking", ["FAISS", "Reranked"])
if rank_mode == "FAISS":
    final_results = faiss_results
elif rank_mode == "Reranked":
    final_results = rerank_results

if final_results:
    if selected_group != "All":
        final_results = [res for res in final_results if res["group"] == selected_group]
    if selected_category != "All":
        final_results = [res for res in final_results if res["category"] == selected_category]
    if selected_color != "All":
        final_results = [res for res in final_results if res["color"] == selected_color]
    if selected_style != "All":
        final_results = [res for res in final_results if res["style"] == selected_style]
    if selected_occasion != "All":
        final_results = [res for res in final_results if res["occasion"] == selected_occasion]
    if selected_material != "All":
        final_results = [res for res in final_results if res["material"] == selected_material]

st.markdown("---")
st.subheader("Top Matches")

if not final_results:
    st.warning("No results found matching the selected filters.")
else:
    display_results = final_results[:8]
    num_cols = 4

    for row_start in range(0, len(display_results), num_cols):
        row_items = display_results[row_start : row_start + num_cols]
        cols = st.columns(num_cols)

        for col, result in zip(cols, row_items):
            with col:
                img = Image.open(result["image_path"]).convert("RGB")
                st.image(img, use_container_width=True)
                title = f"{result['color'].title()} {result['category'].title()}"
                st.markdown(f"**{title}**")
                st.caption(f"File: {Path(result['image_path']).name}")
                st.caption(f"Match Score: {result.get('final_score', 0) * 100:.2f}%")
                attribute_caption = get_attribute_caption(result)
                if attribute_caption:
                    st.caption(attribute_caption)
                st.caption(f"{result['pattern']} • {result['style']}")

    if recommendations and uploaded_file:
        st.markdown("---")
        st.subheader("Recommended Matches")

        rec_cols = st.columns(4)

        for i, rec in enumerate(recommendations):
            with rec_cols[i % 4]:
                img = Image.open(rec["image_path"]).convert("RGB")
                st.image(img, use_container_width=True)
                title = f"{rec['color'].title()} {rec['category'].title()}"
                st.markdown(f"**{title}**")

                st.caption(f"File: {Path(rec['image_path']).name}")

                st.caption(
                    f"Recommendation Score: {rec['recommendation_score'] * 100:.2f}%"
                )
                attribute_caption = get_attribute_caption(rec)
                if attribute_caption:
                    st.caption(attribute_caption)
                st.caption(f"{rec['pattern']} • {rec['style']}")
