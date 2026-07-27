"""Vocal timbre representation learning from unlabeled spectrogram chunks."""

from .config import ExperimentConfig
from .model import VocalTimbreVAE

__all__ = ["ExperimentConfig", "VocalTimbreVAE"]
