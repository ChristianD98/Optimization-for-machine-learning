#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clap_transfer.py
================
Compute CLAP (Contrastive Language–Audio Pre‑training) embeddings for the ESC-50
train / test splits and cache them to disk – analogous to the
`get_transfer_values_classic_networks()` helper used for ImageNet features.

The script exposes a single public function:

    get_transfer_values_clap(dataset, clap_model_name="laion/clap-htsat-fused")

which returns two NumPy arrays `(transfer_train, transfer_test)` containing the
frozen audio‑encoder representations.  Results are pickled to
`<data_dir>/<model_name>_<dataset.name>_{train|test}.pkl` to avoid redundant
computation.

Example
-------
>>> from esc50 import Esc50
>>> from clap_transfer import get_transfer_values_clap
>>> ds = Esc50(cache_to_npz=False, augment=False, normalize=False)
>>> tr, te = get_transfer_values_clap(ds)
>>> print(tr.shape)  # (1600, 512) – CLAP's default feature size

CLI Usage
---------
$ python clap_transfer.py                 # default LAION/HTS‑AT model
$ python clap_transfer.py --model openai/musicclip-audio

Requirements
------------
* `torch`
* `transformers>=4.38`
* `torchaudio`
* `tqdm`

Author: ChatGPT-o3 — May 2025  • License: MIT
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

__all__ = ["get_transfer_values_clap"]

def _load_clap(model_name: str, device: str):
    """Load CLAP **audio encoder** (text tower is unused here)."""
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()
    # freeze
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def _embed_audio(
    wav_paths: list[Path],
    processor: ClapProcessor,
    model: ClapModel,
    device: str,
    target_sr: int = 48_000,
) -> np.ndarray:
    """Return CLAP embeddings for a list of audio files."""
    feats = []
    for path in tqdm(wav_paths, desc="Extracting CLAP features"):
        # read waveform (mono) with torchaudio (handles resampling)
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

        inputs = processor(audios=waveform, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_audio_features(**{k: v.to(device) for k, v in inputs.items()})
        feats.append(emb.cpu().numpy())

    return np.concatenate(feats, axis=0)


def get_transfer_values_clap(
    dataset,
    clap_model_name: str = "laion/clap-htsat-fused",
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cached / freshly‑computed CLAP embeddings for *dataset*.

    Parameters
    ----------
    dataset : object
        Must expose ``name``, ``data_dir``, ``meta_csv_path`` (or ``audio_dir``)
        and `train_folds` / `test_folds` akin to *Esc50* loader.
    clap_model_name : str
        Any CLAP checkpoint compatible with *transformers* (default is LAION).
    device : str, optional
        "cuda", "mps" or "cpu".  Auto‑detects if *None*.
    """

    device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # paths for cached pickles
    fname_base = f"{clap_model_name.split('/')[-1]}_{dataset.name}"
    pkl_train = Path(dataset.data_dir) / f"{fname_base}_train.pkl"
    pkl_test = Path(dataset.data_dir) / f"{fname_base}_test.pkl"

    # load / compute train embeddings
    if pkl_train.exists():
        print("CLAP train embeddings found → loading cache …")
        with open(pkl_train, "rb") as f:
            transfer_train = pickle.load(f)
    else:
        print("Computing CLAP embeddings for training set …")
        processor, model = _load_clap(clap_model_name, device)
        wav_paths_train = [dataset.audio_dir / fn for fn in dataset.meta[dataset.meta["fold"].isin(dataset.train_folds)]["filename"]]
        transfer_train = _embed_audio(wav_paths_train, processor, model, device)
        with open(pkl_train, "wb") as f:
            pickle.dump(transfer_train, f)

    # load / compute test embeddings
    if pkl_test.exists():
        print("CLAP test embeddings found → loading cache …")
        with open(pkl_test, "rb") as f:
            transfer_test = pickle.load(f)
    else:
        print("Computing CLAP embeddings for test set …")
        # reuse previously loaded processor/model if they exist
        if "processor" not in locals():
            processor, model = _load_clap(clap_model_name, device)
        wav_paths_test = [dataset.audio_dir / fn for fn in dataset.meta[dataset.meta["fold"].isin(dataset.test_folds)]["filename"]]
        transfer_test = _embed_audio(wav_paths_test, processor, model, device)
        with open(pkl_test, "wb") as f:
            pickle.dump(transfer_test, f)

    return transfer_train, transfer_test


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from datasets.esc50 import Esc50

    ap = argparse.ArgumentParser(description="Cache CLAP embeddings for ESC‑50")
    ap.add_argument("--model", default="laion/clap-htsat-fused", help="HF model id")
    ap.add_argument("--data_dir", default="./data/esc50", help="ESC‑50 root dir")
    ap.add_argument("--device", default="", help="torch device (cpu, cuda, mps)")
    args = ap.parse_args()

    ds = Esc50(data_dir=args.data_dir, cache_to_npz=False, augment=False)

    get_transfer_values_clap(ds, clap_model_name=args.model, device=args.device or None)
