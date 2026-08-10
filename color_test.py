import json
from pathlib import Path

import cv2

from config import PROJECT_ROOT, resolve_project_path
from embeddings.buildMetadata import MetadataBuilder


TEST_IMAGES = [
    "dataset/extracted_images/image_776.png",
    "dataset/extracted_images/image_1245.png",
    "dataset/extracted_images/image_2311.png",
    "dataset/extracted_images/image_2609.png",
    "dataset/extracted_images/image_5503.png",
    "dataset/extracted_images/image_5507.png",
    "dataset/extracted_images/image_5599.png",
    "dataset/extracted_images/image_7764.png",
    "dataset/extracted_images/image_8367.png",
    "dataset/extracted_images/image_8732.png",
]


def load_bgr_and_alpha(image_path: str):
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

    return bgr_image, alpha


def main():
    # Color extraction does not use the classifier,
    # so None is sufficient for this test.
    builder = MetadataBuilder(classifier=None, metadata_path="unused.pkl")

    output_dir = Path(PROJECT_ROOT) / "debug_color_masks"
    output_dir.mkdir(exist_ok=True)

    for stored_path in TEST_IMAGES:
        try:
            color_info = builder.extract_color(stored_path)

            bgr_image, alpha = load_bgr_and_alpha(stored_path)

            # mask = builder.create_foreground_mask(
            #     bgr_image,
            #     alpha,
            # )

            mask, method, success = builder.get_final_foreground_mask(
                bgr_image, alpha, stored_path
            )
            print(f"Segmentation method for {stored_path}: {method}, success: {success}")
            print(f"Foreground pixels: {cv2.countNonZero(mask)}")

            mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            foreground_preview = cv2.bitwise_and(
                bgr_image,
                bgr_image,
                mask=mask,
            )

            comparison = cv2.hconcat(
                [
                    bgr_image,
                    mask_preview,
                    foreground_preview,
                ]
            )

            output_name = Path(stored_path).stem + "_mask_comparison.jpg"

            cv2.imwrite(
                str(output_dir / output_name),
                comparison,
            )

            print(f"\nImage: {stored_path}")
            print(json.dumps(color_info, indent=2))

        except Exception as error:
            print(f"\nFAILED: {stored_path}")
            print(error)


if __name__ == "__main__":
    main()
