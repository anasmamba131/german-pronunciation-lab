from __future__ import annotations

from schemas import CheckResponse


def test_response_schema_accepts_documented_shape() -> None:
    response = CheckResponse.model_validate(
        {
            "sentence": "ich",
            "normalized_sentence": "ich",
            "expected_phonemes": "ɪç",
            "decoded_phonemes": "ɪç",
            "overall_score": 88.2,
            "overall_status": "correct",
            "words": [
                {
                    "word": "ich",
                    "score": 88.2,
                    "status": "correct",
                    "start": 0.1,
                    "end": 0.4,
                    "phoneme_indices": [0],
                }
            ],
            "phonemes": [
                {
                    "expected": "ç",
                    "letters": "ch",
                    "word_index": 0,
                    "start": 0.2,
                    "end": 0.4,
                    "score": 88.2,
                    "confidence": 0.8,
                    "status": "correct",
                    "expected_probability": 0.7,
                    "alternative_probability": 0.1,
                }
            ],
            "audio": {
                "duration_seconds": 0.5,
                "original_sample_rate": 48000,
                "analysis_sample_rate": 16000,
                "peak": 0.4,
                "rms": 0.1,
                "clipping_ratio": 0.0,
            },
            "methodology": "experimental",
        }
    )
    assert response.score_status == "experimental"
    assert response.timing_status == "approximate"

