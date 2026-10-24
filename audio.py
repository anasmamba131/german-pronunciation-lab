from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    duration_seconds: float
    peak: float
    rms: float
    clipping_ratio: float
    warnings: tuple[str, ...]


def _decode_pcm(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    raise AudioValidationError(f"Unsupported PCM sample width: {sample_width * 8} bits")


def load_wav_bytes(
    data: bytes,
    *,
    target_sample_rate: int = 16_000,
    min_seconds: float = 0.3,
    max_seconds: float = 30.0,
    silence_rms: float = 0.003,
) -> AudioData:
    if len(data) < 44:
        raise AudioValidationError("The WAV file is empty or incomplete.")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            frames = wav.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(f"Invalid WAV file: {exc}") from exc

    if compression != "NONE":
        raise AudioValidationError("Only uncompressed PCM WAV audio is supported.")
    if channels != 1:
        raise AudioValidationError(
            f"Expected mono WAV audio, but received {channels} channels."
        )
    if sample_rate <= 0 or frame_count <= 0:
        raise AudioValidationError("The WAV file contains no audio frames.")

    samples = _decode_pcm(frames, sample_width)
    duration = float(frame_count / sample_rate)
    if duration < min_seconds:
        raise AudioValidationError(
            f"Recording is too short ({duration:.2f}s); record at least {min_seconds:.1f}s."
        )
    if duration > max_seconds:
        raise AudioValidationError(
            f"Recording is too long ({duration:.1f}s); limit is {max_seconds:.0f}s."
        )

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.995))
    if rms < silence_rms:
        raise AudioValidationError(
            "The recording is silent or too quiet. Move closer to the microphone and retry."
        )

    warnings: list[str] = []
    if clipping_ratio > 0.01:
        warnings.append("More than 1% of samples are clipped; reduce microphone level.")
    if rms < 0.015:
        warnings.append("The recording level is low; phoneme confidence may be unreliable.")

    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        samples = resample_poly(
            samples, target_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    else:
        samples = samples.astype(np.float32, copy=False)

    return AudioData(
        samples=samples,
        sample_rate=target_sample_rate,
        original_sample_rate=sample_rate,
        duration_seconds=duration,
        peak=peak,
        rms=rms,
        clipping_ratio=clipping_ratio,
        warnings=tuple(warnings),
    )

