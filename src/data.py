"""Dataset loading and preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.datasets import load_breast_cancer as sklearn_load_breast_cancer


class Dataset(NamedTuple):
    features: pd.DataFrame
    target: pd.Series
    name: str


TARGET_CANDIDATES = ("target", "outcome", "diagnosis", "class", "label")


def load_breast_cancer() -> Dataset:
    """Load the validated breast-cancer classification dataset shipped with sklearn."""
    raw = sklearn_load_breast_cancer(as_frame=True)
    return Dataset(raw.data, raw.target.rename("target"), "Breast Cancer")


def load_csv(path: str | Path, dataset_name: str = "Medical CSV") -> Dataset:
    """Load a CSV and infer its binary target column.

    Expected common target names include target (heart data) and Outcome (diabetes).
    """
    frame = pd.read_csv(path)
    normalized = {column.lower().strip(): column for column in frame.columns}
    target_column = next((normalized[name] for name in TARGET_CANDIDATES if name in normalized), None)
    if target_column is None:
        raise ValueError(
            "Could not find a target column. Name it one of: " + ", ".join(TARGET_CANDIDATES)
        )
    target = frame.pop(target_column)
    if target.nunique(dropna=True) != 2:
        raise ValueError("This application currently supports binary classification targets only.")
    return Dataset(frame, target, dataset_name)
