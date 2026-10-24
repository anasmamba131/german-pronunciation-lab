from __future__ import annotations

import numpy as np

from alignment import ctc_viterbi_align, greedy_decode


def test_ctc_viterbi_alignment_finds_expected_spans() -> None:
    probabilities = np.full((5, 4), 0.01, dtype=np.float64)
    winners = [3, 1, 3, 2, 3]
    for frame, winner in enumerate(winners):
        probabilities[frame, winner] = 0.97
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probs = np.log(probabilities)
    spans = ctc_viterbi_align(log_probs, [1, 2], blank_id=3)
    assert [(span.start_frame, span.end_frame) for span in spans] == [(1, 2), (3, 4)]


def test_repeated_targets_require_intervening_blank() -> None:
    probabilities = np.full((4, 3), 0.01, dtype=np.float64)
    for frame, winner in enumerate([2, 1, 2, 1]):
        probabilities[frame, winner] = 0.98
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    spans = ctc_viterbi_align(np.log(probabilities), [1, 1], blank_id=2)
    assert spans[0].end_frame <= spans[1].start_frame


def test_greedy_decode_collapses_ctc() -> None:
    logits = np.array([[0, 4, 0], [0, 4, 0], [0, 0, 4], [4, 0, 0]], dtype=float)
    assert greedy_decode(logits, blank_id=2, labels={0: "a", 1: "b"}) == "ba"

