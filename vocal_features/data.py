from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


@dataclass(frozen=True)
class Track:
    track_id: str
    audio_path: Path
    artist: str


def read_metadata(path: Path) -> list[Track]:
    """Read track_id,path,artist rows and resolve audio paths beside the CSV."""
    tracks: list[Track] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"track_id", "path", "artist"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"metadata must contain columns: {sorted(required)}")
        for row in reader:
            audio_path = (path.parent / row["path"]).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio for {row['track_id']} not found: {audio_path}")
            tracks.append(Track(row["track_id"].strip(), audio_path, row["artist"].strip()))

    if len({track.track_id for track in tracks}) != len(tracks):
        raise ValueError("track_id values must be unique")
    counts = {artist: sum(t.artist == artist for t in tracks) for artist in {t.artist for t in tracks}}
    too_small = [artist for artist, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(f"each artist needs at least two tracks; too few: {too_small}")
    return tracks


def split_by_track(
    tracks: list[Track], validation_fraction: float, seed: int
) -> tuple[list[Track], list[Track]]:
    """Make a deterministic per-artist split, keeping chunks from one track together."""
    rng = random.Random(seed)
    by_artist: dict[str, list[Track]] = {}
    for track in tracks:
        by_artist.setdefault(track.artist, []).append(track)

    train: list[Track] = []
    validation: list[Track] = []
    for artist_tracks in by_artist.values():
        rng.shuffle(artist_tracks)
        n_validation = max(1, round(len(artist_tracks) * validation_fraction))
        n_validation = min(n_validation, len(artist_tracks) - 1)
        validation.extend(artist_tracks[:n_validation])
        train.extend(artist_tracks[n_validation:])
    return train, validation


class FeatureStore:
    """Extract log-mel chunks once and keep reusable, per-track .npy files."""

    def __init__(self, cache_dir: Path, config: ExperimentConfig):
        self.cache_dir = cache_dir
        self.config = config
        cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, track: Track) -> Path:
        stat = track.audio_path.stat()
        fingerprint = "|".join(
            [
                str(track.audio_path),
                str(stat.st_mtime_ns),
                str(stat.st_size),
                str(self.config.sample_rate),
                str(self.config.n_fft),
                str(self.config.hop_length),
                str(self.config.n_mels),
                str(self.config.frames_per_chunk),
            ]
        )
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in track.track_id)
        return self.cache_dir / f"{safe_id}-{digest}.npy"

    def prepare(self, track: Track) -> Path:
        destination = self.path_for(track)
        if destination.exists():
            return destination

        audio, _ = librosa.load(
            track.audio_path, sr=self.config.sample_rate, mono=True
        )
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=20,
            fmax=self.config.sample_rate // 2,
        )
        log_mel = np.log(mel + 1e-6).astype(np.float32)
        width = self.config.frames_per_chunk
        chunks = [
            log_mel[:, start : start + width].reshape(-1)
            for start in range(0, log_mel.shape[1] - width + 1, width)
        ]
        if not chunks:
            raise ValueError(
                f"{track.audio_path} is shorter than {self.config.chunk_seconds:g} seconds"
            )
        np.save(destination, np.stack(chunks).astype(np.float32))
        return destination

    def prepare_all(self, tracks: list[Track]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for index, track in enumerate(tracks, start=1):
            print(f"[features {index:>4}/{len(tracks)}] {track.track_id}")
            paths[track.track_id] = self.prepare(track)
        return paths


def fit_standardizer(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean and variance without loading the corpus at once."""
    total = 0
    feature_sum: np.ndarray | None = None
    feature_square_sum: np.ndarray | None = None
    for path in paths:
        chunks = np.load(path, mmap_mode="r")
        chunks_64 = np.asarray(chunks, dtype=np.float64)
        current_sum = chunks_64.sum(axis=0)
        current_square_sum = np.square(chunks_64).sum(axis=0)
        feature_sum = current_sum if feature_sum is None else feature_sum + current_sum
        feature_square_sum = (
            current_square_sum
            if feature_square_sum is None
            else feature_square_sum + current_square_sum
        )
        total += len(chunks)

    if total == 0 or feature_sum is None or feature_square_sum is None:
        raise ValueError("no training chunks found")
    mean = feature_sum / total
    variance = np.maximum(feature_square_sum / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class ChunkDataset(Dataset):
    """Index cached chunks while retaining only a few tracks in memory."""

    def __init__(
        self,
        tracks: list[Track],
        paths: dict[str, Path],
        mean: np.ndarray,
        scale: np.ndarray,
        max_open_tracks: int = 8,
    ):
        self.paths = paths
        self.mean = mean
        self.scale = scale
        self.max_open_tracks = max_open_tracks
        self.index: list[tuple[str, int]] = []
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        for track in tracks:
            n_chunks = len(np.load(paths[track.track_id], mmap_mode="r"))
            self.index.extend((track.track_id, i) for i in range(n_chunks))

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, track_id: str) -> np.ndarray:
        if track_id in self._arrays:
            self._arrays.move_to_end(track_id)
            return self._arrays[track_id]
        array = np.load(self.paths[track_id], mmap_mode="r")
        self._arrays[track_id] = array
        if len(self._arrays) > self.max_open_tracks:
            self._arrays.popitem(last=False)
        return array

    def __getitem__(self, index: int) -> torch.Tensor:
        track_id, chunk_index = self.index[index]
        chunk = np.asarray(self._array(track_id)[chunk_index], dtype=np.float32)
        normalized = (chunk - self.mean) / self.scale
        return torch.from_numpy(normalized.copy())
