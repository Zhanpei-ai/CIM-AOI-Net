"""
Dataloader for AOI_Net.

Reads fixation/scanpath data from .xlsx files in a given directory, one file per
participant (or per recording). The expected raw column names, their internal
rename, and the numeric feature columns are defined in config.json (loaded at
import time), so no column lists are hardcoded in this file.

The `result` label is label-encoded to contiguous integers 0..C-1 (no fixed
value mapping). AOI names are label-encoded into integers and returned
alongside each scanpath. Sequences are grouped per Participant_ID so that one
sample == one participant's full scanpath.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


_CONFIG = _load_config()

# Feature columns and raw-column -> canonical-name mapping come from config.json.
FEATURE_COLS = _CONFIG["features"]
COLUMN_RENAME = _CONFIG["column_rename"]


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
