from __future__ import annotations

import subprocess

import pytest

from g2p import GermanG2P, extract_words, map_phones_to_graphemes, normalize_text


def test_normalize_text_preserves_german_unicode() -> None:
    assert normalize_text("  Ich\u00a0 möchte   Deutsch sprechen. ") == "Ich möchte Deutsch sprechen."


def test_normalize_text_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Enter a German sentence"):
        normalize_text("  \n ")


def test_extract_words_tracks_offsets() -> None:
    words = extract_words("Grün, schön!")
    assert [(item.text, item.start, item.end) for item in words] == [
        ("Grün", 0, 4),
        ("schön", 6, 11),
    ]


def test_g2p_runner_and_grapheme_mapping() -> None:
    def fake_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, "ˈ_ɪ_ç\n".encode("utf-8"), b"")

    pronunciation = GermanG2P("espeak-ng.exe", runner=fake_runner).phonemize("Ich")
    assert pronunciation.display_ipa == "ˈɪç"
    ch = pronunciation.phones[-1]
    assert ch.symbol == "ç"
    assert ch.letters.casefold() == "ch"


def test_targeted_grapheme_groups() -> None:
    assert map_phones_to_graphemes("Pflicht", ["p", "f"])[0] == (0, 2)
    assert map_phones_to_graphemes("Zeit", ["t", "s"])[0] == (0, 1)
    assert map_phones_to_graphemes("grün", ["ɡ", "r", "y", "ː", "n"])[3] == (2, 3)

