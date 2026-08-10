import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from natsort import natsorted
from PIL import Image

from config import get_model_kwargs
from embeddings.extractEmbedding import extractEmbedding


def normalize_stored_path(path: str) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(f"Old image path is outside the project: {path}") from exc
    return path.as_posix()


def encode_image(extractor: extractEmbedding, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")
        image = extractor.preprocess(image).unsqueeze(0)
    embedding = extractor.extract_features(image)
    return embedding.cpu().numpy().squeeze().astype("float32")


def update_embeddings(
    image_dir: Path,
    old_embedding_path: Path,
    old_image_path: Path,
    output_embeddings_path: Path,
    output_images_path: Path,
) -> None:
    old_embeddings = np.load(old_embedding_path)
    with open(old_image_path, "rb") as file:
        old_paths = pickle.load(file)
    if old_embeddings.shape[0] != len(old_paths):
        raise ValueError(
            f"Existing artifacts mismatch: {old_embeddings.shape[0]} embeddings but "
            f"{len(old_paths)} paths."
        )
    old_embedding_by_path = {}
    for stored_path, embedding in zip(old_paths, old_embeddings):
        normalized_path = normalize_stored_path(stored_path)
        if normalized_path not in old_embedding_by_path:
            old_embedding_by_path[normalized_path] = embedding.astype("float32")

    abs_filenames = [
        path.name
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    ]
    current_abs_paths = [image_dir / filename for filename in natsorted(abs_filenames)]
    current_rel_paths = [
        path.resolve().relative_to(PROJECT_ROOT).as_posix()
        for path in current_abs_paths
    ]
    current_path_set = set(current_rel_paths)
    old_path_set = set(old_embedding_by_path)
    print(f"Reusing embeddings for {len(current_path_set & old_path_set)} images")
    print(f"Generating embeddings for {len(current_path_set - old_path_set)} new images")
    print(f"Discarding embeddings for {len(old_path_set - current_path_set)} removed images")
    extractor = extractEmbedding(**get_model_kwargs())
    updated_embeddings = []
    for abs_path, rel_path in zip(current_abs_paths, current_rel_paths):
        if rel_path in old_embedding_by_path:
            embedding = old_embedding_by_path[rel_path]
        else:
            embedding = encode_image(extractor, abs_path)
            print(f"Encoded new image: {rel_path}")
        updated_embeddings.append(embedding)
    if not updated_embeddings:
        raise ValueError("No supported images found.")
    updated_matrix = np.vstack(updated_embeddings).astype("float32")
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_images_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings_path, updated_matrix)
    with open(output_images_path, "wb") as file:
        pickle.dump(current_rel_paths, file)
    print("\nUpdate complete")
    print(f"{len(updated_matrix)} Updated embeddings saved to: {output_embeddings_path}")
    print(f"{len(current_rel_paths)} Updated image paths saved to: {output_images_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update embeddings and image paths by reusing existing embeddings for "
            "unchanged images and generating new embeddings for new images."
        )
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "extracted_images",
    )
    parser.add_argument(
        "--old_embeddings",
        type=Path,
        default=PROJECT_ROOT / "embeddings" / "embeddings.npy",
    )
    parser.add_argument(
        "--old_image_paths",
        type=Path,
        default=PROJECT_ROOT / "embeddings" / "image_paths.pkl",
    )
    parser.add_argument(
        "--output_embeddings",
        type=Path,
        default=PROJECT_ROOT / "embeddings" / "embeddings_updated.npy",
    )
    parser.add_argument(
        "--output_image_paths",
        type=Path,
        default=PROJECT_ROOT / "embeddings" / "image_paths_updated.pkl",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    update_embeddings(
        image_dir=args.image_dir.resolve(),
        old_embedding_path=args.old_embeddings.resolve(),
        old_image_path=args.old_image_paths.resolve(),
        output_embeddings_path=args.output_embeddings.resolve(),
        output_images_path=args.output_image_paths.resolve(),
    )
