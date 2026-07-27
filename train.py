from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocal_features.config import ExperimentConfig
from vocal_features.data import (
    ChunkDataset,
    FeatureStore,
    fit_standardizer,
    read_metadata,
    split_by_track,
)
from vocal_features.embedding import (
    embed_tracks,
    fit_artist_prototypes,
    retrieval_metrics,
)
from vocal_features.model import VocalTimbreVAE, vae_loss


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: VocalTimbreVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features in loader:
            features = features.to(device)
            reconstruction, mean, log_variance = model(features)
            loss, reconstruction_loss, kl_loss = vae_loss(
                features, reconstruction, mean, log_variance, beta
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = len(features)
            totals["loss"] += loss.item() * batch_size
            totals["reconstruction"] += reconstruction_loss.item() * batch_size
            totals["kl"] += kl_loss.item() * batch_size
            examples += batch_size
    return {name: value / examples for name, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    tracks = read_metadata(args.metadata.resolve())
    train_tracks, validation_tracks = split_by_track(
        tracks, config.validation_fraction, config.seed
    )
    print(
        f"{len(tracks)} tracks, {len(set(t.artist for t in tracks))} artists "
        f"({len(train_tracks)} train / {len(validation_tracks)} validation)"
    )

    store = FeatureStore(args.cache.resolve(), config)
    feature_paths = store.prepare_all(tracks)
    mean, scale = fit_standardizer(
        [feature_paths[track.track_id] for track in train_tracks]
    )
    np.savez(output_dir / "standardizer.npz", mean=mean, scale=scale)

    train_dataset = ChunkDataset(train_tracks, feature_paths, mean, scale)
    validation_dataset = ChunkDataset(validation_tracks, feature_paths, mean, scale)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    device = choose_device(args.device)
    print(f"training on {device}")
    model = VocalTimbreVAE(
        config.input_dim, config.latent_dim, config.hidden_dims
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "model.pt"
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, config.beta, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, config.beta, optimizer=None
        )
        scheduler.step(validation_metrics["loss"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:>2} | train {train_metrics['loss']:.4f} | "
            f"validation {validation_metrics['loss']:.4f}"
        )

        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    train_embeddings = embed_tracks(
        model, train_tracks, feature_paths, mean, scale, device
    )
    validation_embeddings = embed_tracks(
        model, validation_tracks, feature_paths, mean, scale, device
    )
    artists, prototype_means, variances, priors = fit_artist_prototypes(
        train_embeddings, train_tracks
    )
    np.savez(
        output_dir / "artist_prototypes.npz",
        artists=artists,
        means=prototype_means,
        variances=variances,
        priors=priors,
    )

    metrics = {
        "best_validation_loss": best_loss,
        **retrieval_metrics(
            validation_embeddings,
            validation_tracks,
            artists,
            prototype_means,
            variances,
            priors,
        ),
        "history": history,
        "note": "Artist metrics use held-out tracks and prototypes fit only on training tracks.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))
    print(f"artifacts saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-supervised vocal timbre VAE.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache", type=Path, default=Path(".feature-cache"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
