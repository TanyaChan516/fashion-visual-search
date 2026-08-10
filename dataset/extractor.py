import argparse
import io
import os
from pathlib import Path

import pandas as pd
from natsort import natsorted
from PIL import Image


class parquetExtractor:
    def __init__(self, dataset_dir, image_output_dir, caption_output_dir, max_images):
        self.dataset_dir = dataset_dir
        self.image_output_dir = image_output_dir
        self.caption_output_dir = caption_output_dir
        self.max_images = max_images
        Path(self.image_output_dir).mkdir(exist_ok=True)
        Path(self.caption_output_dir).mkdir(exist_ok=True)

    def extract(self, mode="both"):
        """
        mode: 'image'   -> extract images only
              'caption' -> extract captions only
              'both'    -> extract images and captions together
        """
        count = 0
        captions = []

        for file in natsorted(os.listdir(self.dataset_dir)):
            if file.endswith(".parquet"):
                print(f"Processing file: {file}")
                df = pd.read_parquet(os.path.join(self.dataset_dir, file))
                for _, rec in df.iterrows():
                    if count >= self.max_images:
                        break
                    try:
                        count += 1
                        if mode in ("image", "both"):
                            img_data = rec["cloth"]
                            img_bytes = (
                                img_data["bytes"]
                                if isinstance(img_data, dict)
                                else img_data
                            )
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            img.save(os.path.join(self.image_output_dir, f"image_{count}.png"))
                        if mode in ("caption", "both"):
                            captions.append(rec.get("caption", ""))
                        print(f"Processed record {count} from file {file}")
                    except Exception as e:
                        print(f"Error processing record {count} from file {file}: {e}")
                        if mode in ("caption", "both"):
                            captions.append("")
                        continue

        if mode in ("caption", "both") and captions:
            caption_path = os.path.join(self.caption_output_dir, "captions.txt")
            with open(caption_path, "w") as f:
                for caption in captions:
                    f.write(caption + "\n")
            print(f"Saved {len(captions)} captions to {caption_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images and/or captions from parquet dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["image", "caption", "both"],
        default="both",
        help=(
            "'image' = images only, 'caption' = captions only, "
            "'both' = images + captions (default: both)"
        ),
    )
    parser.add_argument(
        "--dataset_dir", default="dataset/data", help="Path to parquet files"
    )
    parser.add_argument(
        "--image_output_dir",
        default="dataset/extracted_images",
        help="Path to save images",
    )
    parser.add_argument(
        "--caption_output_dir",
        default="dataset/extracted_captions",
        help="Path to save captions.txt",
    )
    parser.add_argument(
        "--max_images", type=int, default=11000, help="Max number of records to process"
    )
    args = parser.parse_args()

    extractor = parquetExtractor(
        args.dataset_dir,
        args.image_output_dir,
        args.caption_output_dir,
        args.max_images,
    )
    extractor.extract(mode=args.mode)
