from pathlib import Path

import numpy as np
import torch

from radio_signal_intelligence.blimpy_adapter import tile_pair
from radio_signal_intelligence.dataset import build_dataset
from radio_signal_intelligence.model import TinyMultimodalCNN, parameter_count
from radio_signal_intelligence.synth import CLASS_NAMES, estimate_metadata, make_pair
from radio_signal_intelligence.triage import candidate_score


def test_generator_is_deterministic_and_finite():
    a, truth_a = make_pair(1, 123, 64, 7)
    b, truth_b = make_pair(1, 123, 64, 7)
    assert np.array_equal(a, b)
    assert truth_a == truth_b
    assert a.shape == (2, 64, 64)
    assert np.isfinite(a).all()


def test_candidate_has_on_dominance_and_drift():
    pair, _ = make_pair(1, 456, 64, 8)
    meta = estimate_metadata(pair)
    assert meta.shape == (6,)
    assert meta[0] > meta[1]
    assert abs(meta[4]) > 0.03


def test_dataset_is_balanced_and_group_safe(tmp_path: Path):
    path = build_dataset(tmp_path, n_per_class=20, image_size=32, seed=99)
    data = np.load(path)
    labels = data["labels"]
    groups = data["groups"]
    splits = data["splits"].astype(str)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    assert np.all(counts == 20)
    for group in np.unique(groups):
        assert len(set(splits[groups == group])) == 1


def test_model_is_small_and_has_correct_shape():
    model = TinyMultimodalCNN()
    logits = model(torch.zeros(3, 2, 64, 64), torch.zeros(3, 6))
    assert logits.shape == (3, len(CLASS_NAMES))
    assert parameter_count(model) < 30_000


def test_triage_rewards_candidate_like_metadata():
    probabilities = np.array([0.8, 0.8])
    metadata = np.array(
        [
            [8.0, 2.0, 1.2, 0.01, 0.18, 5.0],
            [8.0, 7.5, 0.05, 0.20, 0.00, 2.0],
        ]
    )
    scores = candidate_score(probabilities, metadata)
    assert scores[0] > scores[1]


def test_tile_pair_shape():
    on = np.zeros((128, 128), dtype=np.float32)
    off = np.ones((128, 128), dtype=np.float32)
    patches = tile_pair(on, off, size=64, stride=64)
    assert len(patches) == 4
    assert patches[0].shape == (2, 64, 64)


def test_llm_brief_is_bounded_and_does_not_call_api(tmp_path: Path):
    from radio_signal_intelligence.llm_brief import build_candidate_brief

    ranking = tmp_path / "ranking.csv"
    ranking.write_text(
        "rank,sample_id,truth,predicted,candidate_probability,triage_score,explanation\n"
        "1,4,candidate_drift,candidate_drift,0.9,0.8,ON-dominant\n",
        encoding="utf-8",
    )
    out = build_candidate_brief(ranking, tmp_path / "brief.md", top_n=1)
    text = out.read_text(encoding="utf-8")
    assert "must not change numerical scores" in text
    assert "0.8" in text
