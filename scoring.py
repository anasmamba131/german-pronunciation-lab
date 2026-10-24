from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from alignment import AlignedToken
from g2p import ExpectedPhone
from phoneme_inventory import BLANK_ID, MODEL_VOCAB, id_to_symbol


EXCLUDED_ALTERNATIVE_IDS = {
    BLANK_ID,
    0,
    MODEL_VOCAB["1"],
    MODEL_VOCAB["["],
    MODEL_VOCAB["]"],
    MODEL_VOCAB["¡"],
    MODEL_VOCAB["«"],
    MODEL_VOCAB["»"],
    MODEL_VOCAB["[UNK]"],
    MODEL_VOCAB["<s>"],
    MODEL_VOCAB["</s>"],
    MODEL_VOCAB["ˈ"],
    MODEL_VOCAB["ˌ"],
    MODEL_VOCAB["ː"],
    *range(61, 65),
    65,  # Unused output index in the checked v2 configuration.
}


@dataclass(frozen=True)
class TokenScore:
    score: float | None
    status: str
    confidence: float
    expected_probability: float | None
    alternative_probability: float | None
    best_alternative: str | None
    gop_log_ratio: float | None


def _experimental_gop_score(gop: float) -> float:
    # A monotonic display transform only. It is not a learner-calibrated probability.
    value = 100.0 / (1.0 + math.exp(-((gop + 0.20) / 0.85)))
    return round(max(0.0, min(100.0, value)), 1)


def score_tokens(
    log_probs: np.ndarray,
    alignment: Sequence[AlignedToken],
    expected: Sequence[ExpectedPhone],
) -> list[TokenScore]:
    if len(alignment) != len(expected):
        raise ValueError("Alignment and expected phoneme lengths differ.")
    labels = id_to_symbol()
    results: list[TokenScore] = []
    for span, phone in zip(alignment, expected, strict=True):
        if not phone.scoreable:
            results.append(TokenScore(None, "uncertain", 0.0, None, None, None, None))
            continue
        frame_probs = np.exp(log_probs[span.start_frame : span.end_frame])
        mean_probs = frame_probs.mean(axis=0)
        expected_probability = float(mean_probs[phone.token_id])
        candidates = mean_probs.copy()
        for token_id in EXCLUDED_ALTERNATIVE_IDS:
            if 0 <= token_id < len(candidates):
                candidates[token_id] = -1.0
        candidates[phone.token_id] = -1.0
        alternative_id = int(np.argmax(candidates))
        alternative_probability = float(candidates[alternative_id])
        gop = math.log(expected_probability + 1e-9) - math.log(
            alternative_probability + 1e-9
        )
        score = _experimental_gop_score(gop)

        winner = max(expected_probability, alternative_probability)
        top_two_share = winner / (expected_probability + alternative_probability + 1e-9)
        evidence = min(1.0, winner / 0.40)
        confidence = round(float(top_two_share * evidence), 3)
        margin = expected_probability - alternative_probability
        if expected_probability >= 0.20 and margin >= 0.08 and confidence >= 0.55:
            status = "correct"
            best_alternative = None
        elif alternative_probability >= 0.20 and margin <= -0.10 and confidence >= 0.55:
            status = "needs_work"
            best_alternative = labels.get(alternative_id)
        else:
            status = "uncertain"
            best_alternative = None
        results.append(
            TokenScore(
                score=score,
                status=status,
                confidence=confidence,
                expected_probability=round(expected_probability, 4),
                alternative_probability=round(alternative_probability, 4),
                best_alternative=best_alternative,
                gop_log_ratio=round(gop, 4),
            )
        )
    return results


def aggregate_score(scores: Sequence[TokenScore]) -> float | None:
    values = [score.score for score in scores if score.score is not None]
    if not values:
        return None
    return round(float(np.mean(values)), 1)


def aggregate_status(scores: Sequence[TokenScore]) -> str:
    relevant = [score.status for score in scores if score.score is not None]
    if not relevant:
        return "uncertain"
    if "needs_work" in relevant:
        return "needs_work"
    if relevant.count("uncertain") >= max(1, math.ceil(len(relevant) / 2)):
        return "uncertain"
    return "correct"
