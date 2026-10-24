from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phoneme_inventory import BLANK_ID, verify_runtime_vocab


class ModelNotInstalledError(RuntimeError):
    pass


EXPECTED_WEIGHT_BYTES = 1_262_093_600


@dataclass(frozen=True)
class AcousticOutput:
    log_probs: np.ndarray
    frame_count: int
    vocabulary_size: int
    device: str


def model_files_present(model_path: Path) -> bool:
    required = ("config.json", "vocab.json", "model.safetensors")
    return (
        model_path.is_dir()
        and all((model_path / name).is_file() for name in required)
        and (model_path / "model.safetensors").stat().st_size == EXPECTED_WEIGHT_BYTES
    )


class GermanPhonemeModel:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        cpu_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self.device_setting = device
        self.cpu_threads = cpu_threads
        self._processor = None
        self._model = None
        self._device = "unloaded"
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def _choose_device(self, torch_module) -> str:
        if self.device_setting == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device_setting not in {"cpu", "cuda"}:
            raise ValueError("DEVICE must be auto, cpu, or cuda.")
        if self.device_setting == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is unavailable.")
        return self.device_setting

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if not model_files_present(self.model_path):
                raise ModelNotInstalledError(
                    f"Model files are missing from {self.model_path}. "
                    "Run .\\scripts\\download_model.ps1 first."
                )
            try:
                import torch
                from transformers import AutoModelForCTC, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch/Transformers are not installed. Run .\\scripts\\setup.ps1."
                ) from exc

            if self.cpu_threads > 0:
                torch.set_num_threads(self.cpu_threads)
            chosen = self._choose_device(torch)
            processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            model = AutoModelForCTC.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            runtime_vocab = processor.tokenizer.get_vocab()
            verify_runtime_vocab(runtime_vocab)
            if int(model.config.pad_token_id) != BLANK_ID:
                raise RuntimeError(
                    f"Expected CTC blank ID {BLANK_ID}, got {model.config.pad_token_id}."
                )
            model.to(chosen)
            model.eval()
            self._processor = processor
            self._model = model
            self._device = chosen

    def infer(self, samples: np.ndarray, sample_rate: int) -> AcousticOutput:
        self.load()
        import torch

        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self._device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        with self._inference_lock, torch.inference_mode():
            output = self._model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            log_probs = torch.log_softmax(output.logits[0], dim=-1).cpu().numpy()
        return AcousticOutput(
            log_probs=log_probs.astype(np.float32, copy=False),
            frame_count=int(log_probs.shape[0]),
            vocabulary_size=int(log_probs.shape[1]),
            device=self._device,
        )
