import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from natsort import natsorted
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import torch
import torch.nn.functional as F

from config import get_model_kwargs


class extractEmbedding:
    def __init__(self, model_name, tokenizer_name, device):
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_features(self, image):
        with torch.no_grad():
            image_features = self.model.encode_image(image.to(self.device))
            norm_features = F.normalize(image_features, dim=-1)
            return norm_features

    def create_embeddings(self, image_dir):
        embeddings = []
        image_paths = []
        for filename in natsorted(os.listdir(image_dir)):
            if not filename.lower().endswith(("png", "jpg", "jpeg", "bmp")):
                continue
            abs_path = Path(image_dir) / filename
            with Image.open(abs_path) as pil_image:
                img = pil_image.convert("RGB")
                img = self.preprocess(img).unsqueeze(0)
            features = self.extract_features(img)
            features = features.cpu().numpy().squeeze().astype("float32")
            relative_path = abs_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            embeddings.append(features)
            image_paths.append(relative_path)
            print(f"Processed {filename}")
        embeddings = np.vstack(embeddings)
        return embeddings, image_paths
    
    def save_embeddings(self, embeddings, image_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(output_dir, "image_paths.pkl"), "wb") as f:
            pickle.dump(image_paths, f)
        print("Embeddings and image paths saved successfully.")


if __name__ == "__main__":
    image_dir = PROJECT_ROOT / "dataset" / "extracted_images"
    output_dir = PROJECT_ROOT / "embeddings"

    extractor = extractEmbedding(**get_model_kwargs())
    embeddings, image_paths = extractor.create_embeddings(image_dir)
    extractor.save_embeddings(embeddings, image_paths, output_dir)
