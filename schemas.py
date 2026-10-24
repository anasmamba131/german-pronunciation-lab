from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PronunciationStatus = Literal["correct", "needs_work", "uncertain"]


class AudioDiagnostics(BaseModel):
    duration_seconds: float
    original_sample_rate: int
    analysis_sample_rate: int
    peak: float
    rms: float
    clipping_ratio: float
    warnings: list[str] = Field(default_factory=list)
    recording_id: str | None = None


class PhonemeResult(BaseModel):
    expected: str
    letters: str | None
    word_index: int
    start: float
    end: float
    score: float | None
    confidence: float
    status: PronunciationStatus
    expected_probability: float | None
    alternative_probability: float | None
    best_alternative: str | None = None
    scoring_note: str | None = None


class WordResult(BaseModel):
    word: str
    score: float | None
    status: PronunciationStatus
    start: float
    end: float
    phoneme_indices: list[int]


class CheckResponse(BaseModel):
    sentence: str
    normalized_sentence: str
    expected_phonemes: str
    decoded_phonemes: str
    overall_score: float | None
    score_status: Literal["experimental"] = "experimental"
    overall_status: PronunciationStatus
    words: list[WordResult]
    phonemes: list[PhonemeResult]
    audio: AudioDiagnostics
    timing_status: Literal["approximate"] = "approximate"
    methodology: str
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_id: str
    model_path: str
    model_downloaded: bool
    model_loaded: bool
    g2p_available: bool
    g2p_path: str | None
    device_setting: str

