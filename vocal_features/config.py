from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices that affect features, model shape, or optimization."""

    sample_rate: int = 22_050
    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 128
    chunk_seconds: float = 6.0

    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (1_024, 512, 256, 128)
    beta: float = 1.0

    batch_size: int = 256
    learning_rate: float = 1e-4
    max_epochs: int = 25
    early_stopping_patience: int = 4
    validation_fraction: float = 0.2
    seed: int = 42

    @property
    def frames_per_chunk(self) -> int:
        return round(self.chunk_seconds * self.sample_rate / self.hop_length)

    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frames_per_chunk

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        values["hidden_dims"] = tuple(values["hidden_dims"])
        return cls(**values)
