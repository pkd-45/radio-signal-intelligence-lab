from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .synth import CLASS_NAMES, estimate_metadata, make_pair


def split_for_group(group_id: int, seed: int) -> str:
    """Deterministic group split that prevents augmented siblings crossing splits."""
    value = (group_id * 2654435761 + seed) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def build_dataset(
    outdir: str | Path,
    n_per_class: int = 80,
    image_size: int = 64,
    seed: int = 20260720,
) -> Path:
    if n_per_class < 10 or n_per_class % 2:
        raise ValueError("n_per_class must be even and at least 10")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    images: list[np.ndarray] = []
    metadata: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    splits: list[str] = []
    rows: list[dict[str, object]] = []

    sample_id = 0
    groups_per_class = n_per_class // 2
    for class_id, class_name in enumerate(CLASS_NAMES):
        for local_group in range(groups_per_class):
            group_id = class_id * 100_000 + local_group
            split = split_for_group(group_id, seed)
            for augmentation in range(2):
                sample_seed = seed + class_id * 1_000_000 + local_group * 10 + augmentation
                pair, truth = make_pair(class_id, sample_seed, image_size, group_id)
                features = estimate_metadata(pair)
                images.append(pair)
                metadata.append(features)
                labels.append(class_id)
                groups.append(group_id)
                splits.append(split)
                row = {
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "group_id": group_id,
                    "augmentation": augmentation,
                    "split": split,
                    **asdict(truth),
                    "peak_on": float(features[0]),
                    "peak_off": float(features[1]),
                    "log_on_off_peak_ratio": float(features[2]),
                    "occupancy": float(features[3]),
                    "estimated_drift": float(features[4]),
                    "ridge_summary": float(features[5]),
                }
                rows.append(row)
                sample_id += 1

    npz_path = out / "dataset.npz"
    np.savez_compressed(
        npz_path,
        images=np.stack(images).astype(np.float32),
        metadata=np.stack(metadata).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        splits=np.asarray(splits),
        class_names=np.asarray(CLASS_NAMES),
    )

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        split: {name: 0 for name in CLASS_NAMES} for split in ("train", "val", "test")
    }
    for label, split in zip(labels, splits, strict=True):
        counts[split][CLASS_NAMES[label]] += 1
    (out / "dataset_summary.json").write_text(
        json.dumps(
            {
                "n_samples": len(labels),
                "image_shape": [2, image_size, image_size],
                "metadata_features": 6,
                "class_names": CLASS_NAMES,
                "split_class_counts": counts,
                "group_leakage_prevented": True,
                "seed": seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return npz_path


class RadioPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    def __init__(self, npz_path: str | Path, split: str):
        data = np.load(npz_path, allow_pickle=False)
        mask = data["splits"].astype(str) == split
        self.images = torch.from_numpy(data["images"][mask])
        self.metadata = torch.from_numpy(data["metadata"][mask])
        self.labels = torch.from_numpy(data["labels"][mask])
        self.sample_ids = np.flatnonzero(mask).astype(int)
        if len(self.labels) == 0:
            raise ValueError(f"split {split!r} is empty")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.images[index], self.metadata[index], self.labels[index], int(self.sample_ids[index])
