from __future__ import annotations

import pytest

from phoneme_inventory import MODEL_VOCAB, UnsupportedPhonemeError, normalize_ipa, tokenize_ipa


def test_ipa_normalization_matches_character_inventory() -> None:
    assert normalize_ipa("t͡s_g: ") == "tsɡː"
    tokens = tokenize_ipa("ˈ_ɪ_ç")
    assert [token.symbol for token in tokens] == ["ˈ", "ɪ", "ç"]
    assert [token.token_id for token in tokens] == [
        MODEL_VOCAB["ˈ"],
        MODEL_VOCAB["ɪ"],
        MODEL_VOCAB["ç"],
    ]
    assert tokenize_ipa("yː")[1].scoreable is False


def test_unknown_ipa_is_rejected_instead_of_silently_changed() -> None:
    with pytest.raises(UnsupportedPhonemeError, match="outside"):
        tokenize_ipa("ɬ")


def test_declared_short_u_umlaut_proxy_is_never_scored() -> None:
    token = tokenize_ipa("ʏ")[0]
    assert token.symbol == "ʏ"
    assert token.token_id == MODEL_VOCAB["y"]
    assert token.scoreable is False


def test_spaces_become_word_delimiters_only_when_requested() -> None:
    assert [t.symbol for t in tokenize_ipa("ɪç dɔ", allow_spaces=True)] == [
        "ɪ",
        "ç",
        "|",
        "d",
        "ɔ",
    ]
