import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, field
import uuid
from typing import Any
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model

@dataclass
class Detection:
    label: str
    category: str
    score: float
    box: list
    mask: Any = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    manual: bool = False

def _clip_box(box, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    return [x1, y1, x2, y2]

def isolate_detection(image, det):
    image_array = np.asarray(image.convert("RGB"))
    mask = det.mask.squeeze().cpu().numpy() if hasattr(det.mask, "cpu") else np.squeeze(det.mask)
    mask = mask.astype(bool)
    if mask.shape != image_array.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    isolated = np.full_like(image_array, 245)
    isolated[mask] = image_array[mask]
    x1, y1, x2, y2 = _clip_box(det.box, image.size)
    left, top = int(np.floor(x1)), int(np.floor(y1))
    right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
    return Image.fromarray(isolated[top:bottom, left:right])

class GroundedSAM:
    def __init__(self, gd_model_name, sam_model_name, device, detection_categories):
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_name)
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_name)
        self.sam_model = Sam2Model.from_pretrained(sam_model_name)
        self.sam_processor = Sam2Processor.from_pretrained(sam_model_name)

        requested_device = str(device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            selected_device = "cpu"
        elif requested_device == "mps" and not torch.backends.mps.is_available():
            selected_device = "cpu"
        else:
            selected_device = requested_device
        self.device = torch.device(selected_device)
        self.gd_model.to(self.device)
        self.sam_model.to(self.device)
        self.detection_categories = detection_categories

    def load_image(self, image):
        if isinstance(image,str):
            image = Image.open(image)
        return image

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
        return intersection / union if union > 0 else 0

    def intersection_over_smaller(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        smaller_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return intersection / smaller_area if smaller_area > 0 else 0

    def suppress(self, current, other, iou_threshold=0.7, containment_threshold=0.85, confidence_ratio=0.9):
        if current.category == other.category:
            return self.calculate_iou(current.box, other.box) >= iou_threshold
        if {current.category, other.category} == {"One-Pieces", "Bottoms"}:
            one_piece = current if current.category == "One-Pieces" else other
            bottom = current if current.category == "Bottoms" else other
            containment = self.intersection_over_smaller(one_piece.box, bottom.box)
            if (current.category == "One-Pieces" and other.category == "Bottoms"
                and containment >= containment_threshold
                and one_piece.score >= bottom.score * confidence_ratio
                ):
                return True
        return False

    def category_nms(self, detections: list[Detection], iou_threshold=0.7):
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            remaining = []
            for det in detections:
                if self.suppress(current, det, iou_threshold=iou_threshold):
                    continue
                remaining.append(det)
            detections = remaining
        return keep

    def valid_size(self, image: Image, bbox, min_ratio=0.03):
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = image.width * image.height
        return bbox_area / image_area >= min_ratio

    def normalize_detection_label(self, raw_label):
        raw_label = raw_label.strip().lower()
        if raw_label in self.detection_categories:
            return raw_label
        matches = []
        for label in self.detection_categories:
            match = re.search(rf"(?<!\\w){re.escape(label)}(?!\\w)", raw_label)
            if match:
                matches.append((match.start(), -len(label), label))
        return min(matches)[2] if matches else None

    def detect(self, image):
        labels = list(self.detection_categories.keys())
        text_labels = [labels]
        inputs = self.gd_processor(image, text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(outputs, threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]], text_labels=text_labels)[0]
        detections = []
        for box, score, raw_label in zip(results["boxes"], results["scores"], results["text_labels"]):
            label = self.normalize_detection_label(raw_label)
            if label is None:
                continue
            if not self.valid_size(image, box.tolist()):
                continue
            detections.append(Detection(label=label, category=self.detection_categories[label], score=float(score), box=box.tolist()))
        return detections

    def segment(self, image: Image, detections: list[Detection]):
        input_boxes = [[det.box for det in detections]]
        inputs = self.sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        for det, mask in zip(detections, masks):
            det.mask = mask
        return detections
