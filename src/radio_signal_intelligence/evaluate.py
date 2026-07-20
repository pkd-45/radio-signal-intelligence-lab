from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .dataset import RadioPairDataset
from .model import TinyMultimodalCNN, parameter_count
from .synth import CLASS_NAMES
from .triage import candidate_score, explanation


@torch.no_grad()
def evaluate_model(npz_path: str | Path, checkpoint_path: str | Path, outdir: str | Path) -> dict[str, float]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ds = RadioPairDataset(npz_path, "test")
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyMultimodalCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    metadata_all: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    images_all: list[np.ndarray] = []
    for images, metadata, labels, ids in loader:
        logits_all.append(model(images, metadata).numpy())
        labels_all.append(labels.numpy())
        metadata_all.append(metadata.numpy())
        sample_ids.append(np.asarray(ids))
        images_all.append(images.numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    metadata = np.concatenate(metadata_all)
    ids = np.concatenate(sample_ids)
    images = np.concatenate(images_all)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(labels, pred, labels=np.arange(len(CLASS_NAMES)))

    metrics = {
        "n_test": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "candidate_brier": float(brier_score_loss((labels == 1).astype(int), probs[:, 1])),
        "n_parameters": int(parameter_count(model)),
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    scores = candidate_score(probs[:, 1], metadata)
    order = np.argsort(scores)[::-1]
    with (out / "candidate_ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "sample_id",
                "truth",
                "predicted",
                "candidate_probability",
                "triage_score",
                "explanation",
            ]
        )
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    int(ids[idx]),
                    CLASS_NAMES[int(labels[idx])],
                    CLASS_NAMES[int(pred[idx])],
                    float(probs[idx, 1]),
                    float(scores[idx]),
                    explanation(metadata[idx]),
                ]
            )

    _plot_summary(images, labels, cm, scores, order, metrics, out)
    return metrics


def _plot_summary(
    images: np.ndarray,
    labels: np.ndarray,
    cm: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
    metrics: dict[str, float],
    out: Path,
) -> None:
    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 10, height_ratios=[1.0, 1.08], hspace=0.48, wspace=0.75)

    for class_id, name in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(grid[0, 2 * class_id : 2 * class_id + 2])
        idx = int(np.flatnonzero(labels == class_id)[0])
        display = images[idx, 0] - images[idx, 1]
        ax.imshow(display, origin="lower", aspect="auto")
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xlabel("frequency bin", fontsize=8)
        if class_id == 0:
            ax.set_ylabel("time bin", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    short_names = ["noise", "drift", "narrow RFI", "burst", "comb RFI"]
    ax_cm = fig.add_subplot(grid[1, 0:4])
    im = ax_cm.imshow(cm)
    ax_cm.set_xticks(range(len(CLASS_NAMES)), short_names, rotation=30, ha="right")
    ax_cm.set_yticks(range(len(CLASS_NAMES)), short_names)
    ax_cm.set_xlabel("predicted class")
    ax_cm.set_ylabel("true class")
    ax_cm.set_title("Held-out confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_scores = fig.add_subplot(grid[1, 4:7])
    data = [scores[labels == class_id] for class_id in range(len(CLASS_NAMES))]
    ax_scores.boxplot(data, tick_labels=short_names, showfliers=False)
    rng = np.random.default_rng(20260720)
    for class_id, values in enumerate(data, start=1):
        x = class_id + rng.normal(0, 0.045, len(values))
        ax_scores.scatter(x, values, s=13, alpha=0.55)
    ax_scores.set_ylim(-0.03, 1.03)
    ax_scores.set_ylabel("transparent triage score")
    ax_scores.set_title("Candidate ranking separates ON-only drift")
    ax_scores.tick_params(axis="x", rotation=30, labelsize=8)

    ax_text = fig.add_subplot(grid[1, 7:10])
    ax_text.axis("off")
    top_idx = int(order[0])
    lines = [
        "REFERENCE BENCHMARK",
        "",
        f"Test samples: {int(metrics['n_test'])}",
        f"Trainable parameters: {int(metrics['n_parameters']):,}",
        f"Macro-F1: {metrics['macro_f1']:.3f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}",
        f"Candidate Brier score: {metrics['candidate_brier']:.3f}",
        "",
        f"Top-ranked truth: {CLASS_NAMES[int(labels[top_idx])].replace('_', ' ')}",
        f"Top triage score: {scores[top_idx]:.3f}",
        "",
        "Controlled synthetic classes only.",
        "No real-data or technosignature claim.",
        "Optional blimpy adapter is kept separate",
        "from the reference benchmark.",
    ]
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle(
        "Radio Signal Intelligence Lab: ON/OFF dynamic spectra → compact multimodal model → actionable ranking",
        fontsize=15,
        y=0.98,
    )
    fig.savefig(out / "radio_signal_intelligence_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
