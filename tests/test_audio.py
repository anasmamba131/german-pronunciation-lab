from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from audio import AudioValidationError, load_wav_bytes


def make_wav(samples: np.ndarray, sample_rate: int = 48_000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        pcm = np.clip(samples, -1, 1)
        wav.writeframes((pcm * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_wav_validation_and_resampling() -> None:
    rate = 48_000
    time = np.arange(rate, dtype=np.float32) / rate
    samples = 0.15 * np.sin(2 * np.pi * 220 * time)
    audio = load_wav_bytes(make_wav(samples), target_sample_rate=16_000)
    assert audio.original_sample_rate == 48_000
    assert audio.sample_rate == 16_000
    assert len(audio.samples) == pytest.approx(16_000, abs=2)
    assert audio.duration_seconds == pytest.approx(1.0)


def test_silence_is_rejected() -> None:
    with pytest.raises(AudioValidationError, match="silent"):
        load_wav_bytes(make_wav(np.zeros(48_000, dtype=np.float32)))


def test_stereo_is_rejected() -> None:
    stereo_interleaved = np.ones(48_000, dtype=np.float32) * 0.1
    with pytest.raises(AudioValidationError, match="mono"):
        load_wav_bytes(make_wav(stereo_interleaved, channels=2))

