from __future__ import annotations

import acoustic_model


def test_partial_weight_is_not_reported_as_downloaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(acoustic_model, "EXPECTED_WEIGHT_BYTES", 8)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"partial")
    assert acoustic_model.model_files_present(tmp_path) is False

    (tmp_path / "model.safetensors").write_bytes(b"complete")
    assert acoustic_model.model_files_present(tmp_path) is True
