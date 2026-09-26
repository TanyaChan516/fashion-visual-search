import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import io
import json
import base64
import tempfile
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, ValidationError
from api.searchReranker import SearchReranker
from api.retrievalEngine import RetrievalEngine
from api.recommendationEngine import RecommendationEngine
from embeddings.buildMetadata import MetadataBuilder
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from api.segment import GroundedSAM, Detection, isolate_detection
from config import (
    METADATA_PATH,
    get_classifier_kwargs,
    get_engine_kwargs_with_metadata,
    get_recommender_kwargs,
    get_grounded_sam_kwargs,
)

app = FastAPI(title="Fashion Visual Search API", version="2.0")


def load_engine():
    engine = RetrievalEngine(**get_engine_kwargs_with_metadata())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    reranker = SearchReranker()
    recommender = RecommendationEngine(**get_recommender_kwargs())
    segmenter = GroundedSAM(**get_grounded_sam_kwargs())
    return engine, classifier, builder, reranker, recommender, segmenter

engine, _, builder, reranker, recommender, segmenter = load_engine()


class TextSearchRequest(BaseModel):
    query: str
    k: int = 12


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search/text")
def search_text(request: TextSearchRequest):
    indices, _, text_scores, _ = engine.text_search(request.query, k=request.k)
    faiss_results = engine.enrich_search_result(indices, text_scores=text_scores)
    rerank_results = reranker.rerank(faiss_results)
    return {
        "query": request.query,
        "faiss_results": faiss_results,
        "rerank_results": rerank_results,
    }


@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        query_path = tmp.name
    try:
        indices, _, image_scores, query_embedding = engine.search(query_path, k=50)
        query_metadata = builder.extract_all_metadata(query_embedding, query_path)
        recommendations = recommender.recommend(query_metadata=query_metadata, k=4)
        faiss_results = engine.enrich_search_result(indices, image_scores=image_scores)
        rerank_results = reranker.rerank(
            [dict(result) for result in faiss_results],
            query_category=query_metadata["category"],
            query_group=query_metadata["group"],
            query_color=query_metadata,
        )
        return {
            "query_metadata": query_metadata,
            "recommendations": recommendations,
            "faiss_results": faiss_results,
            "rerank_results": rerank_results,
        }
    finally:
        os.unlink(query_path)


@app.post("/search/multimodal")
async def search_multimodal(file: UploadFile = File(...), query: str = Form(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        query_path = tmp.name
    try:
        indices, _, image_scores, text_scores, query_embedding, _ = (
            engine.multimodal_search(query_path, query, k=50)
        )
        query_metadata = builder.extract_all_metadata(query_embedding, query_path)
        recommendations = recommender.recommend(query_metadata=query_metadata, k=4)
        faiss_results = engine.enrich_search_result(
            indices,
            image_scores=image_scores,
            text_scores=text_scores,
        )
        rerank_results = reranker.rerank(
            [dict(result) for result in faiss_results],
            query_category=query_metadata["category"],
            query_group=query_metadata["group"],
            query_color=query_metadata,
        )
        return {
            "query": query,
            "query_metadata": query_metadata,
            "recommendations": recommendations,
            "faiss_results": faiss_results,
            "rerank_results": rerank_results,
        }
    finally:
        os.unlink(query_path)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        image_path = tmp.name
    try:
        image = Image.open(image_path).convert("RGB")
        detections = segmenter.detect(image)
        detections = segmenter.category_nms(detections)
        return {
            "detections": [
                {
                    "id": det.id,
                    "label": det.label,
                    "category": det.category,
                    "score": float(det.score),
                    "box": [float(x) for x in det.box],
                    "manual": det.manual,
                }
                for det in detections
            ]
        }
    finally:
        os.unlink(image_path)


class DetectionInput(BaseModel):
    id: str
    label: str
    category: str
    score: float = 0.0
    box: list[float] = Field(min_length=4, max_length=4)
    manual: bool = False


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/segment")
async def segment(file: UploadFile = File(...), detections: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    try:
        raw_detections = json.loads(detections)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="detections must be valid JSON")
    if not isinstance(raw_detections, list) or not raw_detections:
        raise HTTPException(
            status_code=400,
            detail="detections must be a non-empty JSON array",
        )
    try:
        validated_detections = [DetectionInput(**item) for item in raw_detections]
    except (TypeError, ValidationError) as exc:
        raise HTTPException(
            status_code=400,
            detail="detections must match the expected detection schema",
        ) from exc
    detection_objects = [
        Detection(
            id=item.id,
            label=item.label,
            category=item.category,
            score=item.score,
            box=item.box,
            manual=item.manual,
        )
        for item in validated_detections
    ]
    detections_with_masks = segmenter.segment(image, detection_objects)
    outputs = []
    for det in detections_with_masks:
        garment = isolate_detection(image, det)
        outputs.append(
            {
                "id": det.id,
                "label": det.label,
                "category": det.category,
                "score": float(det.score),
                "box": [float(x) for x in det.box],
                "manual": det.manual,
                "image": pil_to_base64(garment),
            }
        )
    return {"garments": outputs}
