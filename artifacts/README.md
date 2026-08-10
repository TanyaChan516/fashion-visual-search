# Aligned retrieval artifact bundle

The runnable search state is one aligned, versioned bundle consisting of exactly these four files:

- `embeddings/embeddings.npy`
- `embeddings/image_paths.pkl`
- `embeddings/metadata.pkl`
- `index/faiss.index`

Alignment is positional: FAISS vector ID `i`, embedding row `i`, image-path entry `i`, and metadata entry `i` must describe the same catalog item. All four files must be generated from the same input ordering and released, downloaded, validated, and replaced together. **Never mix files from different generation runs**, even when their shapes or counts appear compatible; doing so can silently return the wrong paths or metadata for retrieved vectors.

## Current bundle manifest

The repository does not record a separate semantic version for this bundle. For release/versioning purposes, treat this complete hash manifest as the identity of the current bundle and assign a new bundle version whenever any member changes.

| File | SHA-256 |
|---|---|
| `embeddings/embeddings.npy` | `30acf26f2a5d12ed30a45e342ba5f5d782d2857f87f43c423e3763084248c267` |
| `embeddings/image_paths.pkl` | `96288d2a584ecfc3dd73c17c8707989afb2622690a09db5d75632db6ac4b2a6b` |
| `embeddings/metadata.pkl` | `53db976b94f047ac92f118ec380a9ced0a84188cc36db7e90319565dfe4935a9` |
| `index/faiss.index` | `55b3185d099f31546aca3724123b75ff0797799ba39ba95747a55f33c2997c87` |

Verified properties of this bundle:

- Vector/item count: `10424`
- Embedding dimension: `1536`
- Embedding dtype: `float32`
- FAISS index type: `IndexFlatIP`
- Model identifier from `config.py`: `hf-hub:timm/ViT-gopt-16-SigLIP2-384`
- `image_paths.pkl` entries: `10424`
- `metadata.pkl` entries: `10424`

No downloadable release location is recorded in the repository. If the bundle is distributed separately, the release should publish all four files together, a bundle version, this manifest, and download instructions.

## Integrity verification

From the repository root, verify the current files with:

```sh
shasum -a 256 \
  embeddings/embeddings.npy \
  embeddings/image_paths.pkl \
  embeddings/metadata.pkl \
  index/faiss.index
```

A matching digest verifies file bytes, but it does not replace the requirement to use one published four-file manifest. If any file is regenerated, regenerate or validate every bundle member and publish a new manifest/version.

## Pickle safety

`image_paths.pkl` and `metadata.pkl` are Python pickle files. Loading a pickle can execute code embedded in it. Load these files only from this trusted project workspace or a trusted project release whose hashes have been verified. Do not load pickle files obtained from untrusted or unverified sources.

## Regeneration

The repository's generation order is:

```sh
python dataset/extractor.py --mode both
python embeddings/extractEmbedding.py
python embeddings/buildMetadata.py
python index/build_index.py
```

The resulting files must pass, at minimum, these alignment checks before release:

- `embeddings.npy.shape[0]` equals the length of `image_paths.pkl`.
- The length of `metadata.pkl` equals the length of `image_paths.pkl`.
- `faiss.index.ntotal` equals the number of embedding rows.
- `faiss.index.d` equals `embeddings.npy.shape[1]`.
- Each metadata entry's index/path corresponds to the same embedding row and image-path entry.

Regeneration requires the exact source images, model weights/preprocessing behavior, input ordering, and compatible dependencies. Partial replacement of the current bundle is unsupported.
