from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv(
        "MODEL_ID", "q1805/hubert_large-german-IPA-v2"
    )
    model_revision: str = os.getenv(
        "MODEL_REVISION", "7c7e88c9f20df835ce51ef5bbcf1134c378a7761"
    )
    model_path: Path = Path(
        os.getenv(
            "MODEL_PATH",
            str(Path.home() / ".cache" / "german-pronunciation-model"),
        )
    )
    device: str = os.getenv("DEVICE", "auto").lower()
    espeak_path: str | None = os.getenv("ESPEAK_NG_PATH")
    sample_rate: int = 16_000
    max_audio_seconds: float = float(os.getenv("MAX_AUDIO_SECONDS", "30"))
    min_audio_seconds: float = float(os.getenv("MIN_AUDIO_SECONDS", "0.3"))
    silence_rms: float = float(os.getenv("SILENCE_RMS", "0.003"))
    preserve_recordings: bool = _env_bool("PRESERVE_RECORDINGS", True)
    recordings_dir: Path = Path(
        os.getenv("RECORDINGS_DIR", str(PROJECT_DIR / "recordings"))
    )
    cpu_threads: int = int(os.getenv("CPU_THREADS", "0"))


settings = Settings()

