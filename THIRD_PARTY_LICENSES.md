# Third-party components

This repository is released under the MIT License (see `LICENSE`). It builds on
external work that keeps its own terms. Nothing in this list is redistributed
here: models, services and datasets are downloaded or checked out separately,
as described in the README.

## Basis of this work

**OSCAR** — Pulli et al., TU Wien (ACIN). This project extends OSCAR; the
retrieval cascade, the gallery-rendering idea and the two-container service
pattern originate there. The upstream repository
(https://github.com/pullover00/OSCAR) carries no license file; the author has
agreed to the publication of this derived work. Cite OSCAR when citing this
repository — see README.

## Services (run in their own containers, code not included here)

| Component | License | Note |
|---|---|---|
| **FoundationPose** (NVlabs) — 6D pose service | NVIDIA Source Code License, **non-commercial** | Own modifications ship as `services/foundationpose/foundationpose.patch`. That patch is a derivative of NVIDIA's code and stays under NVIDIA's terms, not MIT. |
| **dGeDi / GeDi** — geometric descriptor service | Code: **CC BY-NC 4.0** (non-commercial) | The upstream repository is used **unmodified** — there is no patch. `services/dgedi/server.py` (our HTTP wrapper) and `compute_diameters.py` are ours and MIT. |

Both services are non-commercial. A commercial user of this repository must
replace them or obtain separate permission from their authors.

## Shape encoders

| Component | License (verified in the checkout) |
|---|---|
| **ULIP-2** (Salesforce) | BSD 3-Clause, © 2022 Salesforce.com, Inc. |
| **Uni3D** (BAAI-Vision) | MIT, © 2023 BAAI-Vision |

Checkpoints come from the respective releases; their terms follow the upstream
model cards.

## Pretrained models pulled at runtime (Hugging Face / OpenAI)

CLIP ViT-B/32 (OpenAI), DINOv2-base and SAM 2.1 (Meta), SigLIP-base (Google),
Grounding DINO (IDEA Research), LLaVA-1.5-7B (llava-hf). Each is downloaded on
first use and governed by its own model card. Note in particular that the
LLaVA-1.5 weights derive from a Llama-family base model and therefore carry the
corresponding community license — check the model card before commercial use.

## Evaluation code

| Component | License |
|---|---|
| **bop_toolkit** (Hodan et al.) — BOP pose metrics | MIT, © 2019 Tomas Hodan |
| **MI3DOR evaluation code** (`Retrieval/cross_performance.m`, repository `tianbao-li/MI3DOR`, commit `4325c24c`) | Upstream terms; ported to Python in `evaluation/mi3dor_official_metrics.py` for the Stage-2 numbers |
| Official SHREC'18 track metric script (Pham et al., 3DOR 2018) | Obtained from the track organizers; used unmodified for the track comparison |

## Tools

**Blender 3.4.1** (GPL) renders the gallery views; it runs as an external
binary on the host and is not part of this repository. **PyBullet** (zlib)
drives the Stage-5 grasp simulation.

## Datasets

None of the datasets are redistributed here. Sources, expected layout and
licenses:

- **SHREC'18 RGB-D-to-CAD track** (Pham et al., 3DOR 2018) — from the track organizers.
- **MI3DOR** — from the benchmark authors.
- **YCB-V, T-LESS, LM-O, ITODD** — BOP benchmark (bop.felk.cvut.cz); per-dataset terms apply, ITODD in particular is restricted to non-commercial research.
- **Google Scanned Objects** — CC BY 4.0.
- **HouseCat6D** — from the dataset release, under its own terms.

Users must accept each dataset's license at the source before downloading.
