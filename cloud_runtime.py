from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from acoustic_model import model_files_present
from config import settings as base_settings
from service import PronunciationService


MODEL_FILES = (
    "README.md",
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)


def model_path() -> Path:
    configured = os.getenv("MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "german-pronunciation-model"


def ensure_model_downloaded(destination: Path) -> Path:
    if model_files_present(destination):
        return destination

    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=base_settings.model_id,
        revision=base_settings.model_revision,
        local_dir=destination,
        allow_patterns=list(MODEL_FILES),
    )
    if not model_files_present(destination):
        raise RuntimeError(
            "The pronunciation model download did not finish correctly. "
            "Reboot the app and try again."
        )
    return destination


def create_service() -> PronunciationService:
    destination = ensure_model_downloaded(model_path())
    cloud_settings = replace(
        base_settings,
        model_path=destination,
        device="cpu",
        preserve_recordings=False,
        max_audio_seconds=15.0,
        cpu_threads=max(1, int(os.getenv("CPU_THREADS", "2"))),
    )
    service = PronunciationService(cloud_settings)
    if service.espeak_path is None:
        raise RuntimeError(
            "eSpeak NG is unavailable. Confirm that packages.txt contains espeak-ng."
        )
    service.model.load()
    return service
