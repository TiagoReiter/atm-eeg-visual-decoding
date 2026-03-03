# ATM-S Encoder: Reproducibility, Generalisability, and Robustness of EEG-based Visual Decoding

> **Research Internship (Forschungspraxis)** · Technical University of Munich · 2024
> Full report: [`docs/EEG_Visual_Decoding_Research_Report.pdf`](docs/EEG_Visual_Decoding_Research_Report.pdf)

---

## Overview

This project evaluates the **Adaptive Thinking Mapper (ATM)** framework — a zero-shot EEG-to-image decoding pipeline introduced by [Li et al. (2024)](https://arxiv.org/abs/2403.07721) — across three research dimensions:

| Dimension | Question | Dataset |
|---|---|---|
| **Reproducibility** | Do the original results hold under identical conditions? | THINGS-EEG2 |
| **Generalisability** | Does the framework transfer to a structurally different dataset? | Alljoined1 (NSD) |
| **Hyperparameter Robustness** | How sensitive is performance to learning rate, dropout, and batch size? | THINGS-EEG2 |

The two datasets differ fundamentally: **THINGS-EEG2** contains isolated single-object images (1,654 categories), while **Alljoined1** uses naturalistic scene photographs from the NSD dataset mapped to 80 COCO super-categories — making the generalisability transfer a non-trivial test.

---

## Pipeline Architecture

The pipeline has two stages: a **retrieval stage** that aligns EEG signals with CLIP embeddings, and a **generation stage** that reconstructs images from those embeddings.

```
                         ┌─────────────────────────────────────────┐
                         │            STAGE 1 — RETRIEVAL          │
                         │                                         │
  EEG signal             │   ┌──────────────┐    ┌─────────────┐  │
  (250 Hz, 63ch)  ──────►│   │  EEG Encoder │───►│  Projection │  │
                         │   │  ATM-S or    │    │  Head       │  │──► EEG embedding
  Stimulus image  ──────►│   │  ATM-E       │    └─────────────┘  │
                         │   └──────────────┘                     │
                         │        CLIP (ViT-H-14)──► Image embed  │
                         │                                         │
                         │   Loss: L = α·L_CLIP + (1-α)·L_MSE     │
                         │         (α = 0.99)                      │
                         └─────────────────────────────────────────┘
                                          │
                                          │  Trained EEG embedding
                                          ▼
                         ┌─────────────────────────────────────────┐
                         │           STAGE 2 — GENERATION          │
                         │                                         │
  EEG embedding  ───────►│  Diffusion Prior ──► CLIP image embed  │
                         │                           │             │
                         │                           ▼             │
                         │               SDXL + IP-Adapter        │──► Reconstructed image
                         └─────────────────────────────────────────┘
```

### EEG Encoder variants

| Encoder | Backbone | Design intent |
|---|---|---|
| **ATM-S** | ShallowFBCSPNet + iTransformer | Shallow spatial filtering before transformer |
| **ATM-E** | EEGNetv4 + iTransformer | Deeper temporal–spectral feature extraction |

---

## Key Findings

**Reproducibility** — The original ATM results replicate closely on THINGS-EEG2. Top-1 accuracy at k=200-way retrieval is consistent with the reported values across 10 subjects, confirming the framework is reproducible. See the [research report](docs/EEG_Visual_Decoding_Research_Report.pdf) for per-subject tables.

**Generalisability** — Performance drops substantially on Alljoined1. The primary driver is the structural mismatch: the original framework was designed for isolated objects with THINGS category labels; naturalistic NSD scenes with COCO super-categories require fundamentally different representations. The generation pipeline is incompatible with Alljoined1 out of the box.

**Hyperparameter Robustness** — The contrastive stage is sensitive to learning rate and dropout, but performance plateaus in a stable region (LR ≈ 3×10⁻⁴, dropout ≈ 0.25). Batch size has relatively little effect once above a threshold. Full sweep results are in the report.

---

## Repository Structure

```
.
├── preprocessing/              # Raw EEG preprocessing — THINGS-EEG2 and Alljoined1
│   ├── preprocessing.py            # THINGS-EEG2: epoching, MVNN, channel selection
│   └── preprocessing_alljoined.py  # Alljoined1: same pipeline, NSD-specific adaptations
│
├── Retrieval/                  # Stage 1 — EEG → CLIP embedding (contrastive training)
│   ├── ATMS_retrieval.py           # ATM-S, THINGS-EEG2   [reproducibility]
│   ├── ATMS_retrieval_alljoined.py # ATM-S, Alljoined1    [generalisability]
│   ├── ATME_retrieval.py           # ATM-E, THINGS-EEG2   [reproducibility]
│   ├── ATME_retrieval_alljoined.py # ATM-E, Alljoined1    [generalisability]
│   ├── ATMS_retrieval_hyperparam.py# ATM-S with full CLI hyperparameter control
│   ├── ATMS_retrieval_joint_train.py # Multi-subject joint training variant
│   ├── eegdatasets_leaveone.py     # Dataset: THINGS-EEG2 + ViT-H-14 CLIP features
│   ├── eegdatasets_leaveone_alljoined.py # Dataset: Alljoined1 + NSD expdesign + COCO
│   ├── sweep_phase1_one_subject.py # Automated LR × dropout hyperparameter search
│   └── find_best_phase1.py         # Aggregates W&B sweep results
│
├── Generation/                 # Stage 2 — CLIP embedding → image reconstruction
│   ├── train_vae_latent_512_low_level_no_average.py  # VAE latent training, THINGS-EEG2
│   ├── train_vae_latent_512_low_level_no_average_alljoined.py # VAE latent, Alljoined1
│   ├── ATMS_reconstruction.py      # Full end-to-end reconstruction pipeline
│   ├── diffusion_prior.py          # Prior diffusion: EEG embed → CLIP image embed
│   └── custom_pipeline.py          # SDXL + IP-Adapter inference
│
├── shared/                     # Shared modules (used by both Retrieval and Generation)
│   ├── layers/                     # ATM transformer components (iTransformer, attention, embeddings)
│   ├── utils/                      # Masking, metrics, time-feature utilities
│   ├── loss.py                     # Symmetric CLIP contrastive loss
│   └── util.py                     # W&B logger, NativeScaler, LR scheduler
│
├── configs/                    # Dataset path configuration
│   ├── thingseeg.json              # THINGS-EEG2 data paths (relative to project root)
│   └── alljoined.json              # Alljoined1 data paths + NSD experiment design
│
├── Data/                       # Datasets — not in repo, see Setup below
├── docs/                       # Full research report (PDF)
└── .env.example                # Environment variable template
```

---

## Setup

### 1. Environment

```bash
conda env create -f preprocessing/bci_env.yml
conda activate bci_env
```

### 2. Weights & Biases

All experiments log to [Weights & Biases](https://wandb.ai). Set your API key before running:

```bash
cp .env.example .env
# Edit .env and set WANDB_API_KEY=your_key
# Or simply: wandb login
```

### 3. Data

Datasets are not included due to size. Expected layout after download:

```
Data/
├── Preprocessed_data_250Hz/
│   ├── ThingsEEG/          # sub-01/ … sub-10/
│   └── Alljoined1/         # sub-01/ … sub-08/ + nsd_expdesign.mat
└── images_set/
    ├── ThingsEEG/
    └── Alljoined1/
```

**THINGS-EEG2** (reproducibility): Download preprocessed data from [OSF](https://osf.io/3jk45/) or run the preprocessing script on raw data:
```bash
python preprocessing/preprocessing.py --project_dir ./Data/Raw/ThingsEEG
```

**Alljoined1** (generalisability): Download from [OpenNeuro](https://openneuro.org/) and preprocess:
```bash
python preprocessing/preprocessing_alljoined.py --project_dir ./Data/Raw/Alljoined1
```
Also copy `nsd_expdesign.mat` into `Data/Preprocessed_data_250Hz/Alljoined1/`.

---

## Running Experiments

All scripts are run from the **project root**. Data paths are resolved automatically via `configs/`.

### Reproducibility — THINGS-EEG2

```bash
# ATM-S encoder, all 10 subjects, in-subject evaluation
python Retrieval/ATMS_retrieval.py --subjects sub-01 sub-02 sub-03 sub-04 sub-05 \
    sub-06 sub-07 sub-08 sub-09 sub-10 --insubject True --logger

# ATM-E encoder (EEGNetv4 backbone)
python Retrieval/ATME_retrieval.py --subjects sub-08 --insubject --logger

# Full image reconstruction pipeline (requires trained encoder + diffusion prior)
python Generation/ATMS_reconstruction.py --subject sub-08
```

### Generalisability — Alljoined1

```bash
# ATM-S on naturalistic scenes (80 COCO categories, NSD stimuli)
python Retrieval/ATMS_retrieval_alljoined.py --subjects sub-01 sub-02 sub-03 \
    sub-04 sub-05 sub-06 sub-07 sub-08 --insubject True --logger

# ATM-E on Alljoined1
python Retrieval/ATME_retrieval_alljoined.py --subjects sub-01 --insubject --logger
```

### Hyperparameter Robustness

```bash
# Automated sweep: learning rate × dropout grid, logged to W&B
python Retrieval/sweep_phase1_one_subject.py

# Pull best-run results from W&B
python Retrieval/find_best_phase1.py

# Manual run with explicit hyperparameters
python Retrieval/ATMS_retrieval_hyperparam.py --subject sub-01 --lr 3e-4 \
    --dropout 0.25 --batch_size 256
```

### Low-Level Image Reconstruction (VAE latents)

```bash
# Train EEG → VAE latent mapping (THINGS-EEG2)
python Generation/train_vae_latent_512_low_level_no_average.py --subject sub-08

# Alljoined1 variant
python Generation/train_vae_latent_512_low_level_no_average_alljoined.py --subject sub-01
```

---

## Technical Stack

| Component | Technology |
|---|---|
| EEG encoders | [braindecode](https://braindecode.org/) — ShallowFBCSPNet, EEGNetv4 |
| Transformer backbone | iTransformer (custom, in `shared/layers/`) |
| Image embeddings | [OpenCLIP](https://github.com/mlfoundations/open_clip) — ViT-H-14 (laion2B) |
| Image generation | [Diffusers](https://huggingface.co/docs/diffusers) — SDXL + IP-Adapter |
| Experiment tracking | [Weights & Biases](https://wandb.ai) |
| Framework | PyTorch 2.x |

---

## Acknowledgments

The ATM framework and base code are from [Li et al. (2024)](https://arxiv.org/abs/2403.07721) — *"Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion"* ([GitHub](https://github.com/dongyangli-del/EEG_Image_decode)).

This project extends the original work with:
- Generalisation experiments on the Alljoined1 / NSD dataset
- Systematic hyperparameter sensitivity analysis
- Refactored codebase with shared modules and portable configuration

Conducted as a research internship at the **Technical University of Munich**, 2024.

---

## License

Academic and research use only.
