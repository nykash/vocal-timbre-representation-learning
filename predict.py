from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from vocal_features.config import ExperimentConfig
from vocal_features.data import FeatureStore, Track
from vocal_features.embedding import embed_tracks, posterior_probabilities
from vocal_features.model import VocalTimbreVAE


def predict(audio_path: Path, artifacts_dir: Path, temperature: float) -> None:
    config = ExperimentConfig.load(artifacts_dir / "config.json")
    standardizer = np.load(artifacts_dir / "standardizer.npz")
    prototypes = np.load(artifacts_dir / "artist_prototypes.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VocalTimbreVAE(
        config.input_dim, config.latent_dim, config.hidden_dims
    ).to(device)
    model.load_state_dict(
        torch.load(artifacts_dir / "model.pt", map_location=device, weights_only=True)
    )

    track = Track("query", audio_path.resolve(), "unknown")
    with tempfile.TemporaryDirectory(prefix="vocal-features-") as cache:
        feature_paths = FeatureStore(Path(cache), config).prepare_all([track])
        embedding = embed_tracks(
            model,
            [track],
            feature_paths,
            standardizer["mean"],
            standardizer["scale"],
            device,
        )[0]

    probabilities = posterior_probabilities(
        embedding,
        prototypes["means"],
        prototypes["variances"],
        prototypes["priors"],
        temperature,
    )
    order = np.argsort(probabilities)[::-1]
    print(f"Analyzed {audio_path.name}\n")
    for index in order[:5]:
        print(f"{str(prototypes['artists'][index]):24} {probabilities[index]:6.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a vocal clip and rank artist prototypes.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--temperature", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    predict(arguments.audio, arguments.artifacts, arguments.temperature)
