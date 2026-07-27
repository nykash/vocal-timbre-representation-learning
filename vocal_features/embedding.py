from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import Track
from .model import VocalTimbreVAE


@torch.inference_mode()
def embed_tracks(
    model: VocalTimbreVAE,
    tracks: list[Track],
    feature_paths: dict[str, Path],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Average deterministic chunk means into one embedding per track."""
    model.eval()
    embeddings: list[np.ndarray] = []
    for track in tracks:
        chunks = np.asarray(np.load(feature_paths[track.track_id]), dtype=np.float32)
        chunks = (chunks - mean) / scale
        chunk_embeddings: list[np.ndarray] = []
        for start in range(0, len(chunks), batch_size):
            batch = torch.from_numpy(chunks[start : start + batch_size]).to(device)
            latent_mean, _ = model.encode(batch)
            chunk_embeddings.append(latent_mean.cpu().numpy())
        embeddings.append(np.concatenate(chunk_embeddings).mean(axis=0))
    return np.stack(embeddings)


def fit_artist_prototypes(
    embeddings: np.ndarray, tracks: list[Track]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one diagonal Gaussian per artist in the learned latent space."""
    artists = np.array(sorted({track.artist for track in tracks}))
    labels = np.array([track.artist for track in tracks])
    global_variance = np.maximum(embeddings.var(axis=0, ddof=1), 1e-6)

    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    priors: list[float] = []
    for artist in artists:
        group = embeddings[labels == artist]
        means.append(group.mean(axis=0))
        variance = group.var(axis=0, ddof=1) if len(group) > 1 else global_variance
        variances.append(np.maximum(variance, 1e-6))
        priors.append(len(group) / len(tracks))
    return artists, np.stack(means), np.stack(variances), np.array(priors)


def posterior_probabilities(
    embedding: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    priors: np.ndarray,
    temperature: float = 5.0,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_likelihood = -0.5 * (
        np.log(2 * np.pi * variances).sum(axis=1)
        + (np.square(embedding - means) / variances).sum(axis=1)
    )
    logits = (log_likelihood + np.log(priors + 1e-12)) / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def retrieval_metrics(
    embeddings: np.ndarray,
    tracks: list[Track],
    artists: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    priors: np.ndarray,
) -> dict[str, float]:
    """Evaluate held-out tracks against prototypes fit on training tracks only."""
    ranks: list[int] = []
    for embedding, track in zip(embeddings, tracks):
        probabilities = posterior_probabilities(
            embedding, means, variances, priors, temperature=1.0
        )
        order = artists[np.argsort(probabilities)[::-1]]
        ranks.append(int(np.where(order == track.artist)[0][0]) + 1)
    return {
        "top_1_artist_accuracy": float(np.mean(np.array(ranks) <= 1)),
        "top_3_artist_accuracy": float(np.mean(np.array(ranks) <= 3)),
        "mean_reciprocal_rank": float(np.mean(1 / np.array(ranks))),
        "validation_tracks": len(tracks),
    }
