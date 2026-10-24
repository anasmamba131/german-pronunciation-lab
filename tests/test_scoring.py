from __future__ import annotations

import numpy as np

from alignment import AlignedToken
from g2p import ExpectedPhone
from scoring import aggregate_score, score_tokens


def test_scoring_reports_evidence_backed_alternative() -> None:
    probabilities = np.full((2, 70), 1e-5, dtype=np.float64)
    probabilities[:, 30] = 0.20  # expected ç
    probabilities[:, 49] = 0.70  # competing ʃ
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    phone = ExpectedPhone("ç", 30, 0, "ch", 1, 3, True)
    span = AlignedToken(0, 30, 0, 2, -1.0)
    result = score_tokens(np.log(probabilities), [span], [phone])[0]
    assert result.status == "needs_work"
    assert result.best_alternative == "ʃ"
    assert result.alternative_probability > result.expected_probability


def test_stress_is_not_scored() -> None:
    probabilities = np.full((1, 70), 1 / 70, dtype=np.float64)
    phone = ExpectedPhone("ˈ", 56, 0, None, None, None, False)
    span = AlignedToken(0, 56, 0, 1, -1.0)
    result = score_tokens(np.log(probabilities), [span], [phone])[0]
    assert result.score is None
    assert aggregate_score([result]) is None

