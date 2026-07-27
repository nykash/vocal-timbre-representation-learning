# Vocal Timbre Representation Learning

A prototype for learning compact vocal representations from audio with the intent to classify/detect different quality of timbre in human voices to create an interactive vocal coach.

The core model is a variational autoencoder trained without artist labels to reconstruct log-mel spectrogram chunks. Labels enter later, through a lightweight supervised Gaussian probe used to evaluate and interpret the latent space.

## Why I built it

Handcrafted timbre measurements are easy to explain but discard much of a voice's spectral structure. End-to-end artist classifiers can separate a closed label set, but do not necessarily learn a reusable representation. I wanted an intermediate approach:

1. learn from the audio itself with a reconstruction objective;
2. aggregate local chunks into a recording-level representation;
3. inspect whether the latent geometry tracks singer or style information; and
4. connect the result to a real interface rather than stop at an offline experiment.

## Investigations

- **Leakage-aware evaluation.** Splits happen by track, never by spectrogram chunk. Normalization and artist prototypes are fit on training tracks only.
- **Scalable feature caching.** Each track's chunks are cached independently, invalidated by the audio metadata and feature configuration, and loaded through a bounded in-memory cache.
- **Variable-length inference.** Six-second chunk embeddings are averaged into one recording representation.
- **Honest representation probing.** A diagonal Gaussian per artist measures separability without putting artist labels into the VAE objective.
- **Negative-result analysis.** A tag-IoU auxiliary loss was tested and disabled after it compressed the geometry; the retained latent space also has weak global agreement with subjective style tags.
- **Usability.** The same representation is exposed through an audio API and used by a React/LLM vocal-coaching flow.

## Model

```text
mono audio (22.05 kHz)
        ↓
6 s log-mel chunks (128 × 258)
        ↓
MLP encoder: 1024 → 512 → 256 → 128
        ↓
64-D Gaussian latent
        ↓
mean over chunks
        ↓
recording representation
        ↓
diagonal-Gaussian artist probe
```

The VAE minimizes mean-squared reconstruction error plus KL divergence to a unit Gaussian. Layer normalization, GELU activations, orthogonal initialization, gradient clipping, learning-rate reduction, and early stopping made training more stable.

## Repository layout

```text
vocal-timbre-representation-learning/
├── train.py                    # end-to-end training and held-out evaluation
├── predict.py                  # embed one clip and rank reference prototypes
├── vocal_features/
│   ├── config.py               # reproducible experiment configuration
│   ├── data.py                 # metadata, splitting, extraction, caching
│   ├── model.py                # VAE and objective
│   └── embedding.py            # aggregation, Gaussian probe, metrics
├── examples/
│   └── metadata.example.csv
└── docs/
    ├── experiments.md          # baselines, evidence, failures, next steps
    └── product-integration.md  # API, frontend, and LLM architecture
```

No audio, model weights, or private dataset paths are included.

## Prepare your own data

Use audio you have permission to process. WAV or FLAC is the safest choice, though any format supported by `librosa` should work.

Create this structure:

```text
my-dataset/
├── metadata.csv
└── audio/
    ├── singer_a_01.wav
    ├── singer_a_02.wav
    └── ...
```

`metadata.csv` must contain:

```csv
track_id,path,artist
singer_a_01,audio/singer_a_01.wav,Singer A
singer_a_02,audio/singer_a_02.wav,Singer A
singer_b_01,audio/singer_b_01.wav,Singer B
singer_b_02,audio/singer_b_02.wav,Singer B
```

Requirements:

- one row per recording;
- a unique `track_id`;
- paths relative to the metadata file;
- at least two tracks per artist so each artist can appear in train and validation;
- clips at least six seconds long;
- preferably isolated, dry vocals to reduce accompaniment and room leakage.

For meaningful evaluation, use more than two tracks per artist and vary songs, microphones, and recording sessions.

## Train

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py \
  --metadata /path/to/my-dataset/metadata.csv \
  --output artifacts
```

Training writes:

- `model.pt` — best validation checkpoint;
- `config.json` — feature, architecture, and optimization choices;
- `standardizer.npz` — training-only normalization statistics;
- `artist_prototypes.npz` — downstream Gaussian probe;
- `metrics.json` — loss history and held-out top-1/top-3/MRR.

The `.feature-cache` directory is reusable and safe to delete. It contains derived features, not source audio.

## Run inference

```bash
python predict.py /path/to/vocal_clip.wav --artifacts artifacts
```

The output ranks references in the trained probe. These scores mean “similar relative to this reference set,” not verified identity or calibrated certainty.

## What the experiments showed

The progression was:

1. note-conditioned spectral centroid and harmonic-ratio analysis;
2. a KNN probe over handcrafted note-level vectors;
3. a log-mel VAE with recording-level latent aggregation; and
4. an attempted tag-guided auxiliary objective.

The VAE was the most useful engineering direction because it retained a richer spectrum and handled arbitrary recording lengths. The tag-guided loss was not a success: attraction without negatives encouraged compressed geometry. On the retained 31-artist matrix, latent cosine similarity had only `r=0.064` and `R²=0.004` against human tag overlap. That is reported as a limitation, not hidden as a failed metric.

See [docs/experiments.md](docs/experiments.md) for the full account and [docs/product-integration.md](docs/product-integration.md) for how the model reached the LLM-assisted frontend.

## Scope

This is a compact research artifact extracted from a larger vocal-coaching application. It is not a production biometric system, a claim of singer identity, or a polished paper result. It shows the modeling decisions, infrastructure, failure analysis, and product integration behind the work.
