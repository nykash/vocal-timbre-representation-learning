# Experiment log: what worked, what did not

This is a retrospective of the experiments in the original VocalCoachAI repository. It is intentionally conservative: if a run did not leave a reproducible metric artifact, it is described as an engineering experiment rather than a result.

## Dataset used during development

The local development manifest records 678 tracks across 31 artists. The class distribution was highly imbalanced: artists had between 2 and 100 tracks. Audio is not included here because it is not mine to redistribute.

This imbalance matters. Artist classification accuracy can be misleading, and a random chunk split would leak material from the same song into train and validation. The cleaned pipeline therefore:

- splits at track level, separately within each artist;
- fits normalization statistics on training tracks only;
- fits artist prototypes on training embeddings only; and
- reports retrieval metrics on held-out tracks.

## 1. Hand-engineered spectral baseline

The first representation grouped voiced frames by musical note, then summarized:

- spectral centroid (brightness);
- second-to-first spectral peak magnitude; and
- third-to-first spectral peak magnitude.

This was useful for inspection because each dimension had a physical interpretation. It also exposed pitch as a confounder, which motivated comparing timbre at matched notes. The retained output contains only a three-track similarity matrix and plots, not a defensible held-out score. It should be treated as exploratory signal, not a benchmark.

## 2. K-nearest-neighbor artist probe

A distance-weighted KNN classifier used note-aligned harmonic ratios and swept `k=1..20`. This tested whether a small set of interpretable features separated singers.

The approach had two practical weaknesses:

1. missing notes made fixed-length vectors brittle; and
2. a few ratios discarded most of the spectral envelope and temporal context.

The code printed a stratified test report and cross-validation accuracy, but those console results were not retained. I do not report a number here.

## 3. Self-supervised VAE representation

The final direction reconstructs six-second log-mel chunks with an MLP variational autoencoder:

`128-bin log-mel × 258 frames → 1024 → 512 → 256 → 128 → 64-D latent`

The encoder uses LayerNorm, GELU, orthogonal initialization, gradient clipping, early stopping, and a KL-regularized reconstruction objective. Artist identity is not used by this objective. A song embedding is the mean of its deterministic chunk means.

This was the most useful representation operationally: it accepted variable-length recordings, preserved more of the vocal spectrum than the handcrafted baseline, and could be served with one forward pass per chunk. The original code references validation-loss checkpoints named `0.4913` and `0.4187`, but their run provenance and configurations were not preserved well enough for a fair comparison. They are evidence that validation reconstruction was monitored, not publishable benchmark results.

### Iterations preserved in the training logs

Seventeen TensorBoard run directories remain in the original workspace. Their hyperparameters document exploration of:

- chunk lengths of roughly 2, 6, and 10 seconds;
- latent widths of 12, 32, 48, and 64;
- three- and four-layer encoder/decoder stacks;
- learning rates of `1e-3`, `5e-3`, and `1e-4`;
- log-mel, MFCC, and combined feature paths in the implementation; and
- tag-IoU auxiliary weight `0.1` before returning to `0.0`.

The selected code configuration uses six-second, 128-bin log-mel chunks, a 64-dimensional latent, four hidden layers, and learning rate `1e-4`. The logs are evidence of iteration, but not a controlled ablation: splits, run intent, and all comparable outcome metrics were not recorded consistently. I therefore do not rank these variants after the fact.

## 4. Weak tag supervision

I experimented with adding a pairwise auxiliary objective:

`tag_IoU(artist_i, artist_j) × ||z_i - z_j||²`

The idea was to pull singers with overlapping human-written style tags closer together. In practice, positive-only attraction had no repulsive or variance-preserving term. It often compressed the latent geometry toward a few directions, producing cosine similarities near `-1` or `1`. I disabled this loss in the final configuration.

A better follow-up would use a contrastive objective with explicit negatives, balanced pair sampling, and variance/covariance regularization (for example, VICReg-style terms). Calling the current approach a kernel method would be inaccurate: tag IoU is used as a pair weight, not as a learned or fixed kernel classifier.

## Retained representation analysis

For the retained 31-artist similarity matrix, cosine similarity had:

- Pearson correlation with human tag IoU: **0.064**
- linear-regression R² against tag IoU: **0.004**
- artist pairs with `|cosine similarity| > 0.9`: **3.7%**

This is a useful negative result. The final unsupervised geometry was not globally aligned with the small, subjective tag vocabulary. It does **not** show that the embeddings are useless: reconstruction, held-out artist retrieval, same-singer retrieval, and robustness to recording conditions test different properties. The new training script records held-out top-1, top-3, and mean reciprocal rank so future claims can be tied to a specific run.

## Next experiments

1. Replace artist classification with same-singer retrieval and report Recall@K.
2. Add vocal-source separation or train only on isolated vocals to reduce accompaniment leakage.
3. Compare log-mel VAE against a convolutional VAE and pretrained audio encoders.
4. Use augmentations that preserve identity (gain, mild EQ, room response) with contrastive learning.
5. Track every run with configuration, commit, split manifest, and metrics.
6. Evaluate across microphones and songs, not only across random tracks.
