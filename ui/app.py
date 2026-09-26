import os
import sys
import hashlib
import importlib
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
import base64
import io
import json

API_URL = "http://127.0.0.1:8000"

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.elements.lib.image_utils import image_to_url as streamlit_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig


def _label_kit_image_to_url(image, width, clamp, channels, output_format, image_id):
    return streamlit_image_to_url(
        image,
        LayoutConfig(width=width),
        clamp,
        channels,
        output_format,
        image_id,
    )


label_kit_detection = importlib.import_module(
    "streamlit_label_kit.LabelToolKit.detection"
)
label_kit_detection.image_to_url = _label_kit_image_to_url
detection = label_kit_detection.detection

from config import CATEGORY_MAPPING, ATTRIBUTE_CONFIG, DETECTION_CATEGORIES
from api.segment import Detection, _clip_box

st.set_page_config(layout="wide")
st.title("Fashion Visual Search")


for key, default in {
    "outfit_image_key": None,
    "detections": None,
    "edit_boxes": False,
    "bbox_editor_result": None,
    "bbox_editor_epoch": 0,
    "segmented_detections": None,
    "outfit_search": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def base64_to_pil(encoded):
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def clear_outfit_results():
    st.session_state.segmented_detections = None
    st.session_state.outfit_search = None

def reset_outfit_state(image_key):
    for key in list(st.session_state):
        if key.startswith("selected_"):
            del st.session_state[key]
    st.session_state.outfit_image_key = image_key
    st.session_state.detections = None
    st.session_state.edit_boxes = False
    st.session_state.bbox_editor_result = None
    st.session_state.bbox_editor_epoch += 1
    clear_outfit_results()

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

def show_query_image(query_image: Image.Image):
    st.subheader("Query Image")
    st.image(query_image, width=300)

def show_metadata(query_metadata):
    st.markdown(f"### {query_metadata['category'].title()} ({query_metadata['color']})")
    attribute_caption = get_attribute_caption(query_metadata)
    if attribute_caption:
        st.caption(attribute_caption)
    secondary_parts = []
    for key in ["pattern", "structure", "style"]:
        value = query_metadata.get(key)
        if value:
            secondary_parts.append(value)
    if secondary_parts:
        st.caption(" • ".join(secondary_parts))
    occasion = query_metadata.get("occasion")
    if occasion:
        st.caption(f"Best for: {occasion}")

def show_detection(image: Image.Image, detections):
    img = np.array(image.convert("RGB")).copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = (
            det.label
            if getattr(det, "manual", False)
            else f"{det.label} {det.score * 100:.1f}%"
        )
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - text_height - baseline), (int(x1) + text_width + 12, int(y1)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1) + 6, int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(img)

def resize_canvas(image: Image.Image, max_height=720, max_width=720):
    width, height = image.size
    x_scale = max_width / width
    y_scale = max_height / height
    scale = min(x_scale, y_scale, 1.0)
    if scale == 1.0:
        return image, scale
    resized_image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return resized_image, scale

def show_bbox_editor(query_image, detections):
    st.markdown("### Edit / Add Bounding Boxes")
    st.caption("Drag existing boxes to adjust them, or draw a new box for a missing item.")
    label_list = list(DETECTION_CATEGORIES.keys())
    bboxes = [[float(v) for v in det.box] for det in detections]
    bbox_ids = [det.id for det in detections]
    labels = [label_list.index(det.label) for det in detections]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as editor_file:
        editor_path = editor_file.name
    try:
        query_image.save(editor_path)
        result = detection(
            image_path=editor_path,
            label_list=label_list,
            bboxes=bboxes,
            bbox_ids=bbox_ids,
            labels=labels,
            bbox_format="XYXY",
            read_only=False,
            bbox_show_label=True,
            item_editor=True,
            item_selector=True,
            class_select_type="select",
            ui_size="medium",
            image_width=720,
            image_height=720,
            key=f"bbox_editor_{st.session_state.bbox_editor_epoch}",
        )
    finally:
        os.unlink(editor_path)
    if result and result.get("key"):
        st.session_state.bbox_editor_result = result
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save boxes"):
            editor_result = st.session_state.get("bbox_editor_result")
            if editor_result is None:
                st.warning("No bounding-box changes found.")
                return
            updated_detections = []
            for item in editor_result.get("bbox", []):
                box = _clip_box(item["bboxes"], query_image.size)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                label = item["label_names"]
                bbox_id = item.get("bbox_ids")
                existing = next((det for det in detections if det.id == bbox_id), None)
                if existing is not None:
                    changed = existing.label != label or not np.allclose(existing.box, box)
                    existing.box = box
                    existing.label = label
                    existing.category = DETECTION_CATEGORIES[label]
                    existing.mask = None
                    existing.manual = getattr(existing, "manual", False) or changed
                    updated_detections.append(existing)
                else:
                    updated_detections.append(
                        Detection(
                            label=label,
                            category=DETECTION_CATEGORIES[label],
                            score=0.0,
                            box=box,
                            manual=True,
                        )
                    )
            updated_ids = {det.id for det in updated_detections}
            for det in detections:
                if det.id not in updated_ids:
                    st.session_state.pop(f"selected_{det.id}", None)
            st.session_state.detections = updated_detections
            st.session_state.bbox_editor_result = None
            st.session_state.bbox_editor_epoch += 1
            clear_outfit_results()
            st.session_state.edit_boxes = False
            st.rerun()
    with cancel_col:
        if st.button("Cancel"):
            st.session_state.bbox_editor_result = None
            st.session_state.bbox_editor_epoch += 1
            st.session_state.edit_boxes = False
            st.rerun()

def detect_outfit(uploaded_file):
    response = requests.post(
        f"{API_URL}/detect",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=180,
    )
    response.raise_for_status()
    return [
        Detection(
            id=item["id"],
            label=item["label"],
            category=item["category"],
            score=item["score"],
            box=item["box"],
            manual=item["manual"],
        )
        for item in response.json()["detections"]
    ]

def segment_outfit(uploaded_file, selected_detections):
    detections_payload = [
        {
            "id": det.id,
            "label": det.label,
            "category": det.category,
            "score": float(det.score),
            "box": [float(x) for x in det.box],
            "manual": det.manual,
        }
        for det in selected_detections
    ]
    response = requests.post(
        f"{API_URL}/segment",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        data={"detections": json.dumps(detections_payload)},
        timeout=300,
    )
    response.raise_for_status()
    garments = response.json()["garments"]
    return [
        {
            "id": item["id"],
            "label": item["label"],
            "category": item["category"],
            "image": base64_to_pil(item["image"]),
        }
        for item in garments
    ]

def search_segmented_item(item_image, detection_id):
    buffer = io.BytesIO()
    item_image.save(buffer, format="PNG")
    response = requests.post(
        f"{API_URL}/search/image",
        files={"file": ("garment.png", buffer.getvalue(), "image/png")},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "detection_id": detection_id,
        "query_image": item_image,
        "metadata": data["query_metadata"],
        "faiss_results": data["faiss_results"],
        "rerank_results": data["rerank_results"],
        "recommendations": data["recommendations"],
    }


def show_segmented_items(segmented_items):
    st.markdown("### Segmented Garments")
    for item in segmented_items:
        garment_image = item["image"]
        preview_image, _ = resize_canvas(garment_image, max_height=220, max_width=220)
        garment_col, action_col = st.columns([2.3, 1])
        with garment_col:
            st.image(preview_image, width=preview_image.width)
        with action_col:
            if st.button("Search similar", key=f"search_segment_{item['id']}"):
                st.session_state.outfit_search = search_segmented_item(garment_image, item['id'])
                st.rerun()


def show_search_results(faiss_results, rerank_results, rank_mode,
                        selected_group, selected_category, selected_color, selected_style, selected_occasion, selected_material,
                        recommendations=None, uploaded_file=None):
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
                    match_score = result.get(
                        "final_score",
                        result.get("image_score", result.get("text_score", 0)),
                    )
                    st.caption(f"Match Score: {match_score * 100:.2f}%")
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

with st.sidebar:
    st.subheader("Search")
    image_type = st.radio("Search from", ["Product photo", "Outfit / model photo"])
    rank_mode = st.radio("Ranking", ["FAISS", "Reranked"])

    st.divider()

    groups = ["All"] + sorted(set(CATEGORY_MAPPING.values()))
    selected_group = st.selectbox("Group", groups)

    if selected_group == "All":
        categories = ["All"] + sorted(CATEGORY_MAPPING.keys())
    else:
        categories = ["All"] + sorted(
            [cat for cat, grp in CATEGORY_MAPPING.items() if grp == selected_group]
        )
    selected_category = st.selectbox("Category", categories)
    selected_color = st.selectbox(
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
    selected_style = st.selectbox(
        "Style", ["All"] + sorted(set(ATTRIBUTE_CONFIG["style"]["labels"]))
    )
    selected_occasion = st.selectbox(
        "Occasion", ["All"] + sorted(set(ATTRIBUTE_CONFIG["occasion"]["labels"]))
    )
    selected_material = st.selectbox(
        "Material", ["All"] + sorted(set(ATTRIBUTE_CONFIG["material"]["labels"]))
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
text_query = None if image_type == "Outfit / model photo" else st.text_input("Optional text query", "")

query_image = None
faiss_results = []
rerank_results = []
final_results = []
recommendations = []
if not uploaded_file and not text_query:
    st.info("Upload an image, enter a text query, or both.")
elif text_query and not uploaded_file:
    response = requests.post(
        f"{API_URL}/search/text",
        json={"query": text_query, "k": 12},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    faiss_results = data["faiss_results"]
    rerank_results = data["rerank_results"]
    show_search_results(faiss_results, rerank_results, rank_mode,
                        selected_group, selected_category, selected_color, selected_style, selected_occasion, selected_material)
else:
    query_image = Image.open(uploaded_file).convert("RGB")
    if image_type == "Product photo":
        show_query_image(query_image)
        if not text_query:
            response = requests.post(
                f"{API_URL}/search/image",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=120,
            )
        else:
            response = requests.post(
                f"{API_URL}/search/multimodal",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                data={"query": text_query},
                timeout=120,
            )
            st.markdown(f"**Text Query:** {text_query}")
        response.raise_for_status()
        data = response.json()
        query_metadata = data["query_metadata"]
        recommendations = data["recommendations"]
        faiss_results = data["faiss_results"]
        rerank_results = data["rerank_results"]
        show_metadata(query_metadata)
        show_search_results(faiss_results, rerank_results, rank_mode,
                            selected_group, selected_category, selected_color, selected_style, selected_occasion, selected_material,
                            recommendations=recommendations, uploaded_file=uploaded_file)
    else:
        image_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.outfit_image_key != image_key:
            reset_outfit_state(image_key)

        if st.session_state.detections is None:
            st.session_state.detections = detect_outfit(uploaded_file)
        detections = st.session_state.detections

        if not st.session_state.edit_boxes:
            has_segments = bool(st.session_state.segmented_detections)
            left, center, right = st.columns([0.25, 3.5, 0.25])
            with center:
                if has_segments:
                    image_col, item_col, spacer_col, segment_col = st.columns([1.4, 0.9, 0.25, 1.2])
                else:
                    image_col, item_col = st.columns([1.4, 0.8])
                selected_detections = []
                with item_col:
                    st.markdown("### Detected Items")
                    for det in detections:
                        row = st.columns([3, 1])
                        with row[0]:
                            detection_name = det.label.title()
                            if getattr(det, "manual", False):
                                detection_name += " (Manual)"
                            else:
                                detection_name += f" ({det.score:.0%})"
                            selected = st.checkbox(detection_name, value=True, key=f"selected_{det.id}")
                            if selected:
                                selected_detections.append(det)
                        with row[1]:
                            if st.button("Delete", key=f"delete_{det.id}"):
                                st.session_state.detections = [d for d in st.session_state.detections if d.id != det.id]
                                st.session_state.pop(f"selected_{det.id}", None)
                                clear_outfit_results()
                                st.rerun()
                    if st.button("Edit / Add boxes"):
                        st.session_state.bbox_editor_result = None
                        st.session_state.bbox_editor_epoch += 1
                        st.session_state.edit_boxes = True
                        st.rerun()
                    if st.button("Reset detections"):
                        for det in st.session_state.detections:
                            st.session_state.pop(f"selected_{det.id}", None)
                        st.session_state.detections = None
                        st.session_state.bbox_editor_result = None
                        st.session_state.bbox_editor_epoch += 1
                        clear_outfit_results()
                        st.rerun()
                    if st.button("Segment selected items", type="primary"):
                        if selected_detections:
                            st.session_state.segmented_detections = segment_outfit(uploaded_file, selected_detections)
                            st.session_state.outfit_search = None
                            st.rerun()
                        else:
                            st.warning("Select at least one detection to segment.")
                annotated_image = show_detection(query_image, selected_detections)
                resized_image, _ = resize_canvas(annotated_image)
                with image_col:
                    st.image(resized_image, caption="Detected fashion items")
                if has_segments:
                    with segment_col:
                        show_segmented_items(st.session_state.segmented_detections)

            if st.session_state.outfit_search:
                search = st.session_state.outfit_search
                st.markdown("---")
                st.subheader("Selected Garment Search")
                show_query_image(search["query_image"])
                show_metadata(search["metadata"])
                show_search_results(
                    search["faiss_results"],
                    search["rerank_results"],
                    rank_mode,
                    selected_group,
                    selected_category,
                    selected_color,
                    selected_style,
                    selected_occasion,
                    selected_material,
                    recommendations=search["recommendations"],
                    uploaded_file=True,
                )
        else:
            show_bbox_editor(query_image, detections)
