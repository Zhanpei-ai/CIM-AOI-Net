"""
Dataloader for AOI_Net.

AOI_Net.train uses load_raw_scanpaths, make_splits and fold-only preprocessing.
The legacy loaders below retain their original behavior for other callers.

Reads fixation/scanpath data from .xlsx files in a given directory, one file per
participant (or per recording). The original eight-feature schema is built in,
so the repository needs no extra configuration file. An existing config.json
can override the schema and raw-column mapping at import time.

The `result` label is label-encoded to contiguous integers 0..C-1 (no fixed
value mapping). AOI names are label-encoded into integers and returned
alongside each scanpath. Sequences are grouped per Participant_ID so that one
sample == one participant's full scanpath.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config():
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "features": [
                "X_pos", "Y_pos", "Pupil_Size_X", "Pupil_Size_Y",
                "Pupil_Diameter", "Dispersion_X", "Dispersion_Y", "Duration",
            ],
            "column_rename": {
                "Participant": "Participant_ID",
                "Fixation Position X [px]": "X_pos",
                "Fixation Position Y [px]": "Y_pos",
                "Fixation Average Pupil Size X [px]": "Pupil_Size_X",
                "Fixation Average Pupil Size Y [px]": "Pupil_Size_Y",
                "Fixation Average Pupil Diameter [mm]": "Pupil_Diameter",
                "Fixation Dispersion X [px]": "Dispersion_X",
                "Fixation Dispersion Y [px]": "Dispersion_Y",
                "Event Duration [ms]": "Duration",
                "AOI Name": "AOI",
                "result": "Label",
            },
        }


_CONFIG = _load_config()

# Use the optional config file or the original built-in schema.
FEATURE_COLS = _CONFIG["features"]
COLUMN_RENAME = _CONFIG["column_rename"]


# ASD is the positive class in the holdout evaluation.
FIXED_SEED = 2026
LABEL_MAPPING = {"1": 0, "2": 1}  # TD=0, ASD=1


@dataclass
class Scanpaths:
    sequences: list
    labels: np.ndarray
    aois: list
    participant_ids: list
    input_hashes: dict


def load_raw_scanpaths(paradigm_dir, max_seq_length=32):
    """Return unscaled, unpadded scanpaths, one sample per Participant_ID.

    AOI integers encode equality within each participant's graph only; no
    vocabulary is learned from validation/test participants. The original
    temporal and same-AOI edge rules apply, including background AOIs.
    """
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    paths = sorted(Path(paradigm_dir).glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files in {paradigm_dir}")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    data = pd.concat([pd.read_excel(path).rename(columns=COLUMN_RENAME)
                      for path in paths], ignore_index=True)
    required = {"Participant_ID", "Label", "AOI", *FEATURE_COLS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = data.loc[data["Label"].notna()].copy()
    if data.empty or data["Participant_ID"].isna().any():
        raise ValueError("Empty labelled dataset or missing participant IDs")
    raw_labels = pd.to_numeric(data["Label"], errors="raise")
    if not raw_labels.isin([1, 2]).all():
        raise ValueError("Expected result=1 (TD) or result=2 (ASD)")
    data["Label"] = raw_labels.astype(int) - 1
    sequences, labels, aois, ids = [], [], [], []
    for pid, group in data.groupby("Participant_ID", sort=True):
        if group["Label"].nunique() != 1:
            raise ValueError(f"Conflicting labels for participant {pid}")
        group = group.iloc[:max_seq_length]
        seq = group[FEATURE_COLS].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(seq).all():
            raise ValueError(f"Non-finite numeric features for participant {pid}")
        names = group["AOI"].fillna("__missing_aoi__").astype(str).to_numpy()
        _, encoded = np.unique(names, return_inverse=True)
        sequences.append(seq)
        labels.append(int(group["Label"].iloc[0]))
        aois.append(encoded.astype(np.int64))
        ids.append(str(pid))
    if len(ids) != len(set(ids)):
        raise ValueError("Participant IDs are not unique after string conversion")
    return Scanpaths(sequences, np.asarray(labels, dtype=np.int64), aois, ids, hashes)


def make_splits(labels, folds=5, seed=FIXED_SEED, test_size=0.2):
    labels = np.asarray(labels, dtype=int)
    if folds < 2 or not 0 < test_size < 1:
        raise ValueError("folds must be >=2 and test_size must be between 0 and 1")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("This evaluation requires both TD=0 and ASD=1")
    development, test = train_test_split(np.arange(len(labels)), test_size=test_size,
                                         stratify=labels, random_state=seed)
    development, test = np.sort(development), np.sort(test)
    counts = np.bincount(labels[development], minlength=2)
    if counts.min() < folds:
        raise ValueError(f"Cannot run {folds} internal folds: development class counts={counts.tolist()}")
    if len(np.unique(labels[test])) != 2:
        raise ValueError("Test split must contain both classes")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    inner = [(development[tr], development[va])
             for tr, va in splitter.split(development, labels[development])]
    seen = np.zeros(len(labels), dtype=int)
    for train, validation in inner:
        if set(train) & set(validation) or (set(train) | set(validation)) != set(development):
            raise ValueError("Invalid internal split")
        if (set(train) | set(validation)) & set(test):
            raise ValueError("Test participants entered internal CV")
        seen[validation] += 1
    if not np.all(seen[development] == 1) or np.any(seen[test]):
        raise ValueError("Each development participant must validate exactly once")
    return development, test, inner


def fit_preprocessing(sequences, train_indices):
    scaler = StandardScaler().fit(np.concatenate([sequences[i] for i in train_indices]))
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
            "feature_names": list(FEATURE_COLS)}


def pad_to_mask_and_seq(sequences, max_len=32):
    """
    sequences: list of np.ndarray (L_i, D)
    Returns:
        seqs:  [B, max_len, D]
        masks: [B, max_len]
    """
    seqs, masks = [], []
    for seq in sequences:
        L, D = seq.shape
        Lc = min(L, max_len)
        padded = np.zeros((max_len, D), dtype=np.float32)
        padded[:Lc] = seq[:Lc]
        mask = np.zeros((max_len,), dtype=np.float32)
        mask[:Lc] = 1.0
        seqs.append(padded)
        masks.append(mask)
    return np.array(seqs), np.array(masks)


def prepare_scanpaths(scanpaths, indices, preprocessing, max_seq_length=32):
    """Apply the training-fold scaler, then pad; never fit on held-out data."""
    mean = np.asarray(preprocessing["mean"])
    scale = np.asarray(preprocessing["scale"])
    sequences = [(scanpaths.sequences[i] - mean) / scale for i in indices]
    seqs, masks = pad_to_mask_and_seq(sequences, max_seq_length)
    labels = scanpaths.labels[indices]
    aois = [scanpaths.aois[i] for i in indices]
    return seqs, masks, labels, aois


def _read_and_prepare(paradigm_dir):
    """Read all .xlsx files under paradigm_dir and return one concatenated DataFrame."""
    all_data = pd.DataFrame()
    for file_name in os.listdir(paradigm_dir):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(paradigm_dir, file_name)
            data = pd.read_excel(file_path)
            data = data.rename(columns=COLUMN_RENAME)
            all_data = pd.concat([all_data, data], ignore_index=True)
    # Generic label encoding: map arbitrary `result` values to contiguous
    # integers 0..C-1. No hardcoded value mapping (e.g. 1->0, 2->1).
    all_data = all_data[all_data['Label'].notna()]
    label_encoder = LabelEncoder()
    all_data['Label'] = label_encoder.fit_transform(all_data['Label'].astype(str))
    return all_data


def load_paradigms(paradigm_dir, max_seq_length=256):
    """Load scanpaths from a directory of .xlsx files.

    Returns:
        sequences: list of np.ndarray (L_i, 8) — one standardized scanpath per participant
        labels:    list of int — per-participant label
        aois:      list of np.ndarray (L_i,) — label-encoded AOI sequence per participant
        scaler:    fitted StandardScaler
        classes:   AOI class labels from the LabelEncoder
    """
    all_data = _read_and_prepare(paradigm_dir)

    # Encode AOI names to integers
    le = LabelEncoder()
    all_data['AOI_enc'] = le.fit_transform(all_data['AOI'])
    print(le.classes_)

    # Global standardization across all data
    scaler = StandardScaler()
    all_data[FEATURE_COLS] = scaler.fit_transform(all_data[FEATURE_COLS])

    sequences, labels, aois = [], [], []
    for pid, group in all_data.groupby('Participant_ID'):
        seq = group[FEATURE_COLS].values      # node features for CNN/GNN
        sequences.append(seq)
        labels.append(group['Label'].iloc[0])  # per-participant label
        aois.append(group['AOI_enc'].values)   # integer AOI sequence

    # Print summary info
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Participants = {len(sequences)}")
    print(f"Max seq len = {max(seq_lengths)}")
    print(f"Avg seq len = {np.mean(seq_lengths):.1f}")

    return sequences, labels, aois, scaler, le.classes_


class ScanpathDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, aois, max_seq_length=256, padding_value=0):
        self.sequences = [self._pad_sequence(seq, max_seq_length, padding_value) for seq in sequences]
        self.labels = labels
        self.aois = aois
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def _pad_sequence(self, seq, max_length, padding_value):
        L, D = seq.shape  # sequence length L and feature dim D

        if L < max_length:
            # pad on the L dimension up to max_length
            padded_seq = np.pad(seq, ((0, max_length - L), (0, 0)), mode='constant', constant_values=padding_value)
        elif L > max_length:
            # truncate on the L dimension
            padded_seq = seq[:max_length, :]
        else:
            # exact length, return as-is
            padded_seq = seq

        return padded_seq

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = np.zeros((self.max_seq_length, 8))  # 8 input features
        seq_len = min(len(seq), self.max_seq_length)
        padded_seq[:seq_len] = seq[:seq_len]

        mask = np.zeros(self.max_seq_length)
        mask[:seq_len] = 1

        # additional info: AOI sequence
        aoi = self.aois[idx]

        return (
            torch.FloatTensor(padded_seq),
            torch.FloatTensor(mask),
            torch.LongTensor([self.labels[idx]]),
            aoi,
        )


def load_half_paradigms(paradigm_dir, max_seq_length=256):
    """Like load_paradigms, but sub-samples each scanpath down to half its length."""
    all_data = _read_and_prepare(paradigm_dir)

    # Encode AOI names to integers
    le = LabelEncoder()
    all_data['AOI_enc'] = le.fit_transform(all_data['AOI'])
    print("AOI classes:", le.classes_)

    # Global standardization across all data
    scaler = StandardScaler()
    all_data[FEATURE_COLS] = scaler.fit_transform(all_data[FEATURE_COLS])

    sequences, labels, aois = [], [], []

    for pid, group in all_data.groupby('Participant_ID'):
        full_seq = group[FEATURE_COLS].values          # shape: [L, 8]
        full_aois = group['AOI_enc'].values           # shape: [L]

        L = len(full_seq)
        if L == 0:
            continue

        # Keep only half of the points, uniformly sampled
        target_len = max(1, L // 2)                   # keep at least one point
        idx = np.linspace(0, L - 1, num=target_len, dtype=int)

        seq_half = full_seq[idx]                      # [L/2, 8]
        aois_half = full_aois[idx]                    # [L/2]

        sequences.append(seq_half)
        labels.append(group['Label'].iloc[0])
        aois.append(aois_half)

    # Print summary info
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Participants = {len(sequences)}")
    print(f"Max seq len = {max(seq_lengths)}")
    print(f"Avg seq len = {np.mean(seq_lengths):.1f}")

    return sequences, labels, aois, scaler
