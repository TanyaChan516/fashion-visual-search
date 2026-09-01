# GroundedSAM v2 evaluation summary

- Evaluation set: 32 outfit/model images containing 70 visible garments
- Detection recall: 88.6% (62/70)
- Detection precision: 82.7% (62/75)
- Good/Usable mask rate: 96.9% (62/64 rated masks)

Mask ratings use the following definitions:

- **Good:** The garment is cleanly isolated and ready for retrieval without correction.
- **Usable:** The garment is sufficiently isolated for retrieval, with only minor mask imperfections.
- **Failed:** Major missing or extra regions make the isolated garment unsuitable for retrieval.

The figures above are aggregate results. The raw working evaluation sheet and per-image remarks are intentionally not included in the repository.
