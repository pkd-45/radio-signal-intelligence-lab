from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import RadioPairDataset
from .model import TinyMultimodalCNN, parameter_count


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


@torch.no_grad()
def _metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    for images, metadata, labels, _ in loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        logits = model(images, metadata)
        losses.append(float(criterion(logits, labels).item()))
        labels_all.append(labels.cpu().numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    return {
        "loss": float(np.mean(losses)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_model(
    npz_path: str | Path,
    outdir: str | Path,
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 20260720,
) -> Path:
    _seed_everything(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RadioPairDataset(npz_path, "train")
    val_ds = RadioPairDataset(npz_path, "val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TinyMultimodalCNN().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_path = out / "tiny_multimodal_cnn.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, metadata, labels, _ in train_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(images, metadata)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        val = _metrics(model, val_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val["loss"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_f1={row['val_macro_f1']:.3f} val_bal_acc={row['val_balanced_accuracy']:.3f}"
        )
        if float(val["macro_f1"]) > best_f1:
            best_f1 = float(val["macro_f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "n_parameters": parameter_count(model),
                    "best_val_macro_f1": best_f1,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model_cpu = TinyMultimodalCNN()
    model_cpu.load_state_dict(checkpoint["model_state"])
    model_cpu.eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(out / "tiny_multimodal_cnn_scripted.pt"))

    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (out / "model_card.md").write_text(
        "# Tiny Multimodal CNN model card\n\n"
        f"- Parameters: {parameter_count(model_cpu):,}\n"
        f"- Best validation macro-F1: {best_f1:.3f}\n"
        "- Inputs: two 64x64 ON/OFF waterfall channels plus six transparent metadata features.\n"
        "- Output classes: noise, candidate drift, persistent narrowband RFI, broadband burst, spectral-comb RFI.\n"
        "- Scope: controlled synthetic benchmark only. Not a production technosignature detector.\n"
        "- Intended use: demonstrate dataset construction, compact multimodal modelling, evaluation, export, and candidate triage.\n",
        encoding="utf-8",
    )
    return best_path
