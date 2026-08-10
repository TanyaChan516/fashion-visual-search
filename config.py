import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(PROJECT_ROOT, "index/faiss.index")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "embeddings/embeddings.npy")
IMAGE_PATHS_PATH = os.path.join(PROJECT_ROOT, "embeddings/image_paths.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "embeddings/metadata.pkl")

MODEL_NAME = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
TOKENIZER_NAME = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_MAPPING = {
    "t-shirt": "Tops",
    "blouse": "Tops",
    "tank top": "Tops",
    "crop top": "Tops",
    "sweater": "Tops",
    "hoodie": "Tops",
    "sweatshirt": "Tops",
    "polo": "Tops",
    "trousers": "Bottoms",
    "jeans": "Bottoms",
    "shorts": "Bottoms",
    "skirt": "Bottoms",
    "leggings": "Bottoms",
    "dress": "One-Pieces",
    "jumpsuit": "One-Pieces",
    "bodysuit": "One-Pieces",
    "jacket": "Outerwear",
    "coat": "Outerwear",
    "blazer": "Outerwear",
}

PROMPT_TEMPLATE = [
    "a product photo of a {}",
    "a catalog photo of a {}",
    "a fashion product image of a {}",
    "an isolated {} on a plain background",
]

MATERIAL_LABELS = [
    "cotton", "denim", "chiffon", "satin", "linen", "wool", "tweed", "leather"
]

MATERIAL_PROMPTS = [
    "a fashion product made of {}",
    "a clothing item with {} fabric",
    "this garment looks like {} material"
]

NECKLINE_LABELS = [
    "crew neck", "v-neck", "square neck", "boat neck", "high neck", "halter neck", "off-shoulder"
]

NECKLINE_PROMPTS = [
    "a top with a {} neckline",
    "this garment has a {} neckline",
    "a clothing item with a {} neckline"
]

OCCASION_LABELS = [
    "casual wear", "work wear", "formal wear", "vacation wear", "party wear", "evening wear"
]

OCCASION_PROMPTS = [
    "an outfit suitable for {}",
    "this garment is appropriate for {}",
    "a clothing item for {}"
]

PATTERN_LABELS = [
    "solid",
    "striped",
    "floral",
    "plaid",
    "polka dot",
    "graphic",  # "animal print"
]

PATTERN_PROMPTS = [
    "a clothing item with {} pattern",
    "this garment has a {} pattern",
    "a {} fashion print"
]

SLEEVE_LABELS = [
    "sleeveless", "short sleeve", "long sleeve", "three-quarter sleeve", "puff sleeve"
]

SLEEVE_PROMPTS = [
    "a {} top",
    "this garment has {}",
    "a clothing item with {} sleeves"
]

STRUCTURE_LABELS = [
    "fitted", "regular", "loose", "flowy", "tailored", "boxy"
]

STRUCTURE_PROMPTS = [
    "a {} clothing silhouette",
    "this garment has a {} fit",
    "a {} structured fashion item"
]

STYLE_LABELS = [
    "minimalist", "classic", "feminine", "romantic", "elegant", "sporty", "edgy", "streetwear"
]

STYLE_PROMPTS = [
    "a {} fashion style",
    "this garment has a {} vibe",
    "a clothing item with {} aesthetic"
]

DETAIL_LABELS = [
    "buttons", "lace", "ruffles", "pleats", "embroidery", "bows", "pockets", "zippers", "sequins", "beading", "applique"
]

DETAILS_PROMPTS = [
    "a clothing item with {} detail",
    "this garment features {}",
    "a fashion product with {} design"
]

ATTRIBUTE_CONFIG = {
    "material": {"labels": MATERIAL_LABELS, "prompts": MATERIAL_PROMPTS},
    "neckline": {"labels": NECKLINE_LABELS, "prompts": NECKLINE_PROMPTS},
    "occasion": {"labels": OCCASION_LABELS, "prompts": OCCASION_PROMPTS},
    "pattern": {"labels": PATTERN_LABELS, "prompts": PATTERN_PROMPTS},
    "sleeve": {"labels": SLEEVE_LABELS, "prompts": SLEEVE_PROMPTS},
    "structure": {"labels": STRUCTURE_LABELS, "prompts": STRUCTURE_PROMPTS},
    "style": {"labels": STYLE_LABELS, "prompts": STYLE_PROMPTS},
    "detail": {"labels": DETAIL_LABELS, "prompts": DETAILS_PROMPTS},
}

GROUP_RECOMMENDATIONS = {
    "Tops": ["Bottoms"],
    "Bottoms": ["Tops"],
    "One-Pieces": ["Outerwear"],
    "Outerwear": ["Tops", "Bottoms", "One-Pieces"],
}

COLOR_RECOMMENDATIONS = {
    "black": ["black", "white", "grey", "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink"],
    "white": ["white", "black", "grey", "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink"],
    "grey": ["grey", "black", "white", "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink"],
    "beige": ["beige", "white", "black", "grey", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink"],
    "brown": ["brown", "beige", "white", "black", "grey", "red", "orange", "yellow", "green", "blue", "pink"],
    "red": ["red", "black", "white", "grey", "beige", "brown", "blue", "pink"],
    "orange": ["orange", "black", "white", "grey", "beige", "brown", "blue", "green"],
    "yellow": ["yellow", "black", "white", "grey", "beige", "brown", "green", "blue", "purple"],
    "green": ["green", "black", "white", "grey", "beige", "brown", "orange", "yellow", "blue", "purple", "pink"],
    "blue": ["blue", "black", "white", "grey", "beige", "brown", "red", "orange", "yellow", "green", "purple", "pink"],
    "purple": ["purple", "black", "white", "grey", "beige", "yellow", "green", "blue", "pink"],
    "pink": ["pink", "black", "white", "grey", "beige", "brown", "red", "green", "blue", "purple"]
}

def validate_color_recommendations():
    asymmetric_pairs = []
    for source_color, recommended_colors in COLOR_RECOMMENDATIONS.items():
        for target_color in recommended_colors:
            reverse_colors = COLOR_RECOMMENDATIONS.get(target_color, [])
            if source_color not in reverse_colors:
                asymmetric_pairs.append((source_color, target_color))
    return asymmetric_pairs


STYLE_RECOMMENDATIONS = {
    "minimalist": ["minimalist", "classic", "elegant"],
    "classic": ["minimalist", "classic", "elegant"],
    "feminine": ["feminine", "romantic", "elegant"],
    "romantic": ["feminine", "romantic", "elegant"],
    "elegant": ["minimalist", "classic", "feminine", "romantic", "elegant"],
    "sporty": ["sporty", "streetwear", "minimalist"],
    "edgy": ["edgy", "streetwear"],
    "streetwear": ["sporty", "edgy", "streetwear"],
}

def resolve_project_path(path: str) -> str:
    """Resolve a stored project-relative path into an absolute path."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_engine_kwargs():
    return {
        "index_path": INDEX_PATH,
        "embeddings_path": EMBEDDINGS_PATH,
        "image_paths_path": IMAGE_PATHS_PATH,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "device": DEVICE,
    }

def get_engine_kwargs_with_metadata():
    return {
        "index_path": INDEX_PATH,
        "embeddings_path": EMBEDDINGS_PATH,
        "image_paths_path": IMAGE_PATHS_PATH,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "device": DEVICE,
        "metadata_path": METADATA_PATH,
    }

def get_model_kwargs():
    return {
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "device": DEVICE,
    }

def get_classifier_kwargs():
    return {
        "category_mapping": CATEGORY_MAPPING,
        "attribute_mapping": ATTRIBUTE_CONFIG,
    }

def get_recommender_kwargs():
    return {
        "metadata_path": METADATA_PATH,
        "group_recommendations": GROUP_RECOMMENDATIONS,
        "color_recommendations": COLOR_RECOMMENDATIONS,
        "style_recommendations": STYLE_RECOMMENDATIONS,
    }


if __name__ == "__main__":
    asymmetric_pairs = validate_color_recommendations()

    if asymmetric_pairs:
        for source, target in asymmetric_pairs:
            print(f"Asymmetric pair: {source} -> {target}")
    else:
        print("All color recommendations are symmetric.")
