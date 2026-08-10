import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from api.retrievalEngine import RetrievalEngine
from api.zeroShotClassifier import ZeroShotCategoryClassifier
from config import (
    EMBEDDINGS_PATH,
    IMAGE_PATHS_PATH,
    METADATA_PATH,
    get_classifier_kwargs,
    get_engine_kwargs,
    resolve_project_path,
)


class MetadataBuilder:
    def __init__(self, classifier: ZeroShotCategoryClassifier, metadata_path: str):
        self.classifier = classifier
        self.metadata_path = metadata_path
        self.metadata = []

    def extract_all_attributes(self, image_embedding, group):
        attr = {}
        for attr_name in self.classifier.attribute_embeddings.keys():
            if attr_name in {"neckline", "sleeve"} and group not in {
                "Tops",
                "One-Pieces",
                "Outerwear",
            }:
                continue
            attr[attr_name] = self.classifier.extract_attribute(image_embedding, attr_name)
        return attr

    def create_foreground_mask(self, bgr_image, alpha=None):
        height, width = bgr_image.shape[:2]
        min_pixel = max(50, int(height * width * 0.005))
        if alpha is not None and np.any(alpha < 250):
            mask = np.where(alpha > 20, 255, 0).astype(np.uint8)
        else:
            lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            border_width = max(2, int(min(height, width) * 0.05))
            border_pixels = np.concatenate(
                [
                    lab_image[:border_width, :, :].reshape(-1, 3),
                    lab_image[-border_width:, :, :].reshape(-1, 3),
                    lab_image[:, :border_width, :].reshape(-1, 3),
                    lab_image[:, -border_width:, :].reshape(-1, 3),
                ],
                axis=0,
            )
            background_clr = np.median(border_pixels, axis=0)
            lab_dist = np.linalg.norm(lab_image - background_clr, axis=2)
            border_dist = np.linalg.norm(border_pixels - background_clr, axis=1)
            threshold = float(
                np.clip(np.percentile(border_dist, 95) + 4.0, 10.0, 24.0)
            )
            mask = np.where(lab_dist > threshold, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if components <= 1:
            return mask
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
        min_area = max(50, int(height * width * 0.002), int(largest_component * 0.04))
        cleaned_mask = np.zeros_like(mask)
        for component in range(1, components):
            area = stats[component, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == component] = 255
        if cv2.countNonZero(cleaned_mask) < min_pixel:
            return mask
        return cleaned_mask

    def create_grabcut_mask(self, bgr_image):
        height, width = bgr_image.shape[: 2]
        mask = np.zeros((height, width), dtype=np.uint8)
        margin = max(2, int(min(height, width) * 0.03))
        rectangle = (margin, margin, width - 2 * margin, height - 2 * margin)
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            bgr_image,
            mask,
            rectangle,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
        foreground_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return foreground_mask

    def get_final_foreground_mask(self, bgr_image, alpha=None, image_path=""):
        height, width = bgr_image.shape[:2]
        min_pixel = max(50, int(height * width * 0.005))
        mask = self.create_foreground_mask(bgr_image, alpha)
        method = "alpha" if alpha is not None and np.any(alpha < 250) else "lab"
        if cv2.countNonZero(mask) < min_pixel:
            print(f"LAB mask failed; using GrabCut: {image_path}")
            mask = self.create_grabcut_mask(bgr_image)
            method = "grabcut"
        success = cv2.countNonZero(mask) >= min_pixel
        return mask, method, success

    def classify_color_pixels(self, hsv_pixels):
        h = hsv_pixels[:, 0]
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]
        labels = np.full(len(hsv_pixels), "unknown", dtype="<U10")
        black_mask = (v < 30) | ((v < 60) & (s < 40))
        white_mask = (s < 25) & (v > 220) & ~black_mask
        beige_mask = (
            (h >= 8)
            & (h < 35)
            & (s >= 15)
            & (s < 90)
            & (v >= 140)
            & (v <= 245)
            & ~black_mask
            & ~white_mask
        )
        grey_mask = (s < 30) & ~black_mask & ~white_mask & ~beige_mask
        brown_mask = (
            (h >= 5)
            & (h < 30)
            & (s >= 45)
            & (v >= 55)
            & (v < 170)
            & ~black_mask
            & ~white_mask
            & ~beige_mask
            & ~grey_mask
        )
        labels[black_mask] = "black"
        labels[white_mask] = "white"
        labels[beige_mask] = "beige"
        labels[grey_mask] = "grey"
        labels[brown_mask] = "brown"
        chromatic_mask = ~(black_mask | white_mask | beige_mask | grey_mask | brown_mask)
        h_chromatic = h[chromatic_mask]
        chromatic_labels = np.full(len(h_chromatic), "unknown", dtype="<U10")
        red = (h_chromatic < 10) | (h_chromatic >= 170)
        orange = (h_chromatic >= 10) & (h_chromatic < 25)
        yellow = (h_chromatic >= 25) & (h_chromatic < 35)
        green = (h_chromatic >= 35) & (h_chromatic < 85)
        blue = (h_chromatic >= 85) & (h_chromatic < 130)
        purple = (h_chromatic >= 130) & (h_chromatic < 160)
        pink = (h_chromatic >= 160) & (h_chromatic < 170)
        chromatic_labels[red] = "red"
        chromatic_labels[orange] = "orange"
        chromatic_labels[yellow] = "yellow"
        chromatic_labels[green] = "green"
        chromatic_labels[blue] = "blue"
        chromatic_labels[purple] = "purple"
        chromatic_labels[pink] = "pink"
        labels[chromatic_mask] = chromatic_labels
        return labels

    def extract_color(self, image_path):
        resolved_path = resolve_project_path(image_path)
        image = cv2.imread(resolved_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {resolved_path}")
        alpha = None
        if image.ndim == 2:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr_image = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            bgr_image = image[:, :, :3]
        height, width = bgr_image.shape[:2]
        min_pixel = max(50, int(height * width * 0.005))
        foreground_mask, mask_method, mask_success = self.get_final_foreground_mask(bgr_image, alpha, image_path)
        if not mask_success:
            raise ValueError(f"Foreground detection failed after GrabCut: {resolved_path}")
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        foreground_pixels = hsv_image[foreground_mask > 0]
        if len(foreground_pixels) < min_pixel:
            raise ValueError(f"Foreground detection failed after fallback: {resolved_path}")
        labels = self.classify_color_pixels(foreground_pixels)
        color, count = np.unique(labels, return_counts=True)
        total = count.sum()
        rank_clr = sorted(zip(color, count), key=lambda x: x[1], reverse=True)
        clr_dist = {
            str(color): round(float(count) / float(total), 4)
            for color, count in rank_clr
        }
        main_clr = str(rank_clr[0][0])
        main_share = float(rank_clr[0][1]) / float(total)
        sec_clr = None
        sec_share = 0.0
        if len(rank_clr) > 1:
            sec_clr = str(rank_clr[1][0])
            sec_share = float(rank_clr[1][1]) / float(total)
        is_multi_clr = sec_clr is not None and sec_share >= 0.18 and main_share < 0.75
        return {
            "primary": main_clr,
            "secondary": sec_clr if is_multi_clr else None,
            "is_multicolor": is_multi_clr,
            "distribution": clr_dist,
            "segmentation_method": mask_method,
        }

    def extract_all_metadata(self, image_embedding, image_path):
        category, group = self.classifier.extract_category_and_group(image_embedding)
        color_info = self.extract_color(image_path)
        attributes = self.extract_all_attributes(image_embedding, group)
        return {
            "category": category,
            "group": group,
            "color": color_info["primary"],
            "secondary_color": color_info["secondary"],
            "is_multicolor": color_info["is_multicolor"],
            "color_distribution": color_info["distribution"],
            **attributes,
        }

    def build_metadata(self, embeddings: str, image_paths: str):
        embeddings = np.load(embeddings)
        with open(image_paths, "rb") as f:
            image_paths = pickle.load(f)
        if embeddings.shape[0] != len(image_paths):
            raise ValueError(
                f"Embedding/ path mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(image_paths)} paths"
            )
        self.metadata = []
        for idx, image_path in enumerate(image_paths):
            image_embedding = embeddings[idx : idx + 1]
            item = {
                "index": idx,
                "image_path": image_path,
                **self.extract_all_metadata(image_embedding, image_path),
            }
            self.metadata.append(item)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Metadata built and saved successfully as {self.metadata_path}")


if __name__ == "__main__":
    engine = RetrievalEngine(**get_engine_kwargs())
    classifier = ZeroShotCategoryClassifier(engine, **get_classifier_kwargs())
    classifier.create_embeddings("category")
    classifier.create_embeddings("attribute")
    builder = MetadataBuilder(classifier, metadata_path=METADATA_PATH)
    builder.build_metadata(EMBEDDINGS_PATH, IMAGE_PATHS_PATH)
