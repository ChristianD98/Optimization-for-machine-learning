from __future__ import annotations
import os
import zipfile
import tarfile
import warnings
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple, Union, Optional, Dict

import numpy as np
import pandas as pd
import librosa
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_ARCHIVE_NAME = "ESC-50-master.zip"
ESC50_META_CSV = "ESC-50-master/meta/esc50.csv"
ESC50_AUDIO_DIR = "ESC-50-master/audio"

SAMPLE_RATE = 44_100
DURATION = 5               # seconds
CLIP_SAMPLES = SAMPLE_RATE * DURATION   # 220 500
N_FFT = 1024
HOP_LENGTH = 128
N_MELS = 128
MEL_BINS = (N_MELS, 1723)   # rows × cols after STFT for 5‑s clip

# --------------------------------------------------------------------------- #
# Data‑augmentation helpers (from notebook)
# --------------------------------------------------------------------------- #
rng = np.random.default_rng()

def add_white_noise(x: np.ndarray, rate: float = 0.002) -> np.ndarray:
    """Add Gaussian noise with amplitude proportional to *rate*."""
    return x + rate * rng.standard_normal(len(x))

def shift_sound(x: np.ndarray, rate: int = 2) -> np.ndarray:
    """Circular time‑shift by *len(x)//rate* samples."""
    return np.roll(x, int(len(x) // rate))

def stretch_sound(x: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Time‑stretch (without pitch‑shift) and pad / crop back to original length."""
    input_length = len(x)
    x_stretch = librosa.effects.time_stretch(x, rate)
    if len(x_stretch) > input_length:
        return x_stretch[:input_length]
    return np.pad(x_stretch, (0, input_length - len(x_stretch)), mode="constant")

def one_hot_encode(labels: Sequence[int], num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[labels]

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, chunk: int = 8192) -> None:
    """Stream *url* to *dest* with progress‑bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
        ) as bar:
            for chunk_data in r.iter_content(chunk):
                fh.write(chunk_data)
                bar.update(len(chunk_data))

def _extract_archive(path: Path, dest_dir: Path) -> None:
    if path.suffix == ".zip":
        opener = zipfile.ZipFile
    elif path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        opener = tarfile.open
    else:
        raise ValueError(f"Unsupported archive: {path}")
    with opener(path, "r") as archive:
        archive.extractall(dest_dir)

def _calculate_melsp(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Return log‑mel spectrogram (dB) with shape `(n_mels, time)` in float32."""
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft, ref=np.max)
    mel = librosa.feature.melspectrogram(
        S=log_stft, n_mels=n_mels, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return mel.astype(np.float32)

def _fix_length(wave: np.ndarray, desired_length: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad / crop *wave* to *desired_length* samples."""
    if len(wave) > desired_length:
        return wave[:desired_length]
    elif len(wave) < desired_length:
        return np.pad(wave, (0, desired_length - len(wave)), mode="constant")
    return wave

# --------------------------------------------------------------------------- #
# Core dataset class
# --------------------------------------------------------------------------- #
class Esc50:
    """ESC‑50 environmental sound dataset loader."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "./data/esc50",
        train_folds: Sequence[int] | None = None,
        test_folds: Sequence[int] | None = None,
        augment: bool = False,
        cache_to_npz: bool = True,
        rng_seed: int = 42,
        augmentation_funcs: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.archive_path = self.data_dir / ESC50_ARCHIVE_NAME
        self.dataset_root = self.data_dir / "ESC-50-master"
        self.meta_csv_path = self.dataset_root / "meta" / "esc50.csv"
        self.audio_dir = self.dataset_root / "audio"

        # folds 1‑5; default CV pattern: [1,2,3,4] vs [5]
        self.train_folds = tuple(train_folds or (1, 2, 3, 4))
        self.test_folds = tuple(test_folds or (5,))

        self.augment = augment
        self.cache_to_npz = cache_to_npz
        self.rng = np.random.default_rng(rng_seed)

        self.augmentation_funcs = (
            list(augmentation_funcs)
            if augmentation_funcs is not None
            else [add_white_noise, shift_sound, stretch_sound]
        )

        # public attributes mirroring CIFAR style
        self.height, self.width, self.depth = *MEL_BINS, 1
        self.n_classes = 50

        # hold data arrays
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.normalized = False

        # trigger pipeline
        self._prepare()

    # -------------------------- Public API -------------------------- #
    def as_tf_dataset(self, batch_size: int = 32):
        """Return *tf.data.Dataset* objects (requires TensorFlow installed)."""
        import tensorflow as tf

        if self.x_train is None:
            raise RuntimeError("Data not yet loaded")
        ds_train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        ds_train = ds_train.shuffle(buffer_size=len(self.x_train)).batch(batch_size)
        ds_test = ds_test.batch(batch_size)
        return ds_train, ds_test

    def batch_generator(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yield NumPy mini‑batches (useful for PyTorch)."""
        x, y = self.x_train, self.y_train
        indices = np.arange(len(x))
        while True:
            if shuffle:
                self.rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                idx = indices[start : start + batch_size]
                yield x[idx], y[idx]

    # -------------------------- Internal --------------------------- #
    def _prepare(self) -> None:
        self._maybe_download()
        self._load_dataset()
        if self.normalize:
            self._normalize()

    def _maybe_download(self) -> None:
        if self.meta_csv_path.exists():
            return  # already downloaded / extracted
        print("ESC‑50 dataset not found locally; downloading …")
        _download(ESC50_URL, self.archive_path)
        _extract_archive(self.archive_path, self.data_dir)

    def _load_dataset(self) -> None:
        cache_file = self.data_dir / "esc50_mel.npz"
        if self.cache_to_npz and cache_file.exists():
            print("Loading cached mel‑spectrogram arrays …")
            data = np.load(cache_file)
            self.x_train, self.y_train = data["x_train"], data["y_train"]
            self.x_test, self.y_test = data["x_test"], data["y_test"]
            return

        print("Loading metadata …")
        meta = pd.read_csv(self.meta_csv_path)
        train_meta = meta[meta["fold"].isin(self.train_folds)]
        test_meta = meta[meta["fold"].isin(self.test_folds)]
        print(f"Train samples: {len(train_meta)}, test samples: {len(test_meta)}")

        x_train, y_train = self._process_split(train_meta, is_training=True)
        x_test, y_test = self._process_split(test_meta, is_training=False)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        if self.cache_to_npz:
            print("Saving cached mel‑spectrogram arrays …")
            np.savez_compressed(
                cache_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
            )

    def _process_split(
        self, subset_meta: pd.DataFrame, is_training: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = len(subset_meta)
        x_data = np.zeros((samples, *MEL_BINS, 1), dtype=np.float32)
        y_data = np.zeros(samples, dtype=np.int64)

        for i, row in tqdm(subset_meta.iterrows(), total=samples, desc="Processing audio"):
            filepath = self.audio_dir / row["filename"]
            wave, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            wave = _fix_length(wave)

            # optional augment – randomly pick one augmentation func
            if is_training and self.augment:
                func = self.rng.choice(self.augmentation_funcs)
                wave = func(wave)

            mel = _calculate_melsp(wave)  # shape (128, 1723)
            x_data[i, :, :, 0] = mel
            y_data[i] = row["target"]

        return x_data, one_hot_encode(y_data, self.n_classes)

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="ESC‑50 dataset helper")
    p.add_argument("--no-cache", action="store_false", dest="cache_to_npz",
                   help="Do not save/load the compressed NPZ cache.")
    p.add_argument("--no-aug", action="store_false", dest="augment",
                   help="Disable data‑augmentation for the training split.")
    args = p.parse_args()

    ds = Esc50(augment=args.augment,
               cache_to_npz=args.cache_to_npz)

    print("Train:", ds.x_train.shape, ds.y_train.shape)
    print("Test :", ds.x_test.shape, ds.y_test.shape)

    # demo iteration
    gen = ds.batch_generator(batch_size=8)
    x_batch, y_batch = next(gen)
    print("Mini‑batch shapes:", x_batch.shape, y_batch.shape)
