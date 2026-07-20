from __future__ import annotations

import numpy as np


def candidate_score(candidate_probability: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    """Transparent ranking score, separate from the learned classifier.

    Rewards ON-dominant events and penalises high occupancy. This is a triage
    score, not a calibrated posterior probability of extraterrestrial origin.
    """
    ratio_term = 1.0 / (1.0 + np.exp(-(metadata[:, 2] - 0.35) * 3.0))
    occupancy_penalty = np.clip(1.0 - metadata[:, 3] / 0.08, 0.0, 1.0)
    drift_support = np.clip(np.abs(metadata[:, 4]) / 0.08, 0.25, 1.0)
    return candidate_probability * ratio_term * occupancy_penalty * drift_support


def explanation(metadata_row: np.ndarray) -> str:
    reasons: list[str] = []
    if metadata_row[2] > 0.35:
        reasons.append("ON-dominant")
    else:
        reasons.append("OFF counterpart present")
    if metadata_row[3] < 0.03:
        reasons.append("low occupancy")
    else:
        reasons.append("broad/high occupancy")
    if abs(metadata_row[4]) > 0.05:
        reasons.append("coherent drift")
    else:
        reasons.append("little drift")
    return "; ".join(reasons)
