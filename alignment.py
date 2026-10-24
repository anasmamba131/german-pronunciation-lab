from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlignedToken:
    target_index: int
    token_id: int
    start_frame: int
    end_frame: int
    path_log_probability: float


def ctc_viterbi_align(
    log_probs: np.ndarray, targets: list[int], *, blank_id: int
) -> list[AlignedToken]:
    """Force a target token sequence onto frame-level CTC log probabilities."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [frames, vocabulary].")
    if not targets:
        raise AlignmentError("Expected phoneme sequence is empty.")
    frame_count, vocab_size = log_probs.shape
    if blank_id < 0 or blank_id >= vocab_size or any(
        token < 0 or token >= vocab_size for token in targets
    ):
        raise AlignmentError("Target or blank token is outside the model vocabulary.")

    minimum_frames = len(targets) + sum(
        left == right for left, right in zip(targets, targets[1:])
    )
    if frame_count < minimum_frames:
        raise AlignmentError(
            f"Audio produced {frame_count} frames, but alignment needs at least {minimum_frames}."
        )

    states = [blank_id]
    for token in targets:
        states.extend((token, blank_id))
    state_count = len(states)
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = float(log_probs[0, blank_id])
    if state_count > 1:
        previous[1] = float(log_probs[0, states[1]])
    backpointers = np.full((frame_count, state_count), -1, dtype=np.int32)

    for frame in range(1, frame_count):
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state, emitted in enumerate(states):
            candidates = [(previous[state], state)]
            if state >= 1:
                candidates.append((previous[state - 1], state - 1))
            if (
                state >= 2
                and emitted != blank_id
                and emitted != states[state - 2]
            ):
                candidates.append((previous[state - 2], state - 2))
            best_score, best_state = max(candidates, key=lambda item: item[0])
            if np.isfinite(best_score):
                current[state] = best_score + float(log_probs[frame, emitted])
                backpointers[frame, state] = best_state
        previous = current

    end_candidates = [(previous[state_count - 1], state_count - 1)]
    if state_count > 1:
        end_candidates.append((previous[state_count - 2], state_count - 2))
    end_score, state = max(end_candidates, key=lambda item: item[0])
    if not np.isfinite(end_score):
        raise AlignmentError("No valid CTC path could align this audio to the sentence.")

    path = np.empty(frame_count, dtype=np.int32)
    path[-1] = state
    for frame in range(frame_count - 1, 0, -1):
        state = int(backpointers[frame, state])
        if state < 0:
            raise AlignmentError("CTC backtracking encountered an unreachable state.")
        path[frame - 1] = state

    aligned: list[AlignedToken] = []
    for target_index, token_id in enumerate(targets):
        target_state = 2 * target_index + 1
        frames = np.flatnonzero(path == target_state)
        if not len(frames):
            raise AlignmentError(f"Target token {target_index} received no audio frame.")
        aligned.append(
            AlignedToken(
                target_index=target_index,
                token_id=token_id,
                start_frame=int(frames[0]),
                end_frame=int(frames[-1] + 1),
                path_log_probability=float(
                    np.mean(log_probs[frames, token_id], dtype=np.float64)
                ),
            )
        )
    return aligned


def greedy_decode(log_probs: np.ndarray, *, blank_id: int, labels: dict[int, str]) -> str:
    best = np.argmax(log_probs, axis=-1).tolist()
    collapsed: list[str] = []
    previous: int | None = None
    for token_id in best:
        if token_id != previous and token_id != blank_id:
            symbol = labels.get(token_id, "�")
            if symbol not in {"<s>", "</s>", "[UNK]"}:
                collapsed.append(" " if symbol == "|" else symbol)
        previous = token_id
    return "".join(collapsed).strip()

