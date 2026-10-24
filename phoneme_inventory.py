from __future__ import annotations

import unicodedata
from dataclasses import dataclass


# Verified from q1805/hubert_large-german-IPA-v2/vocab.json at revision
# 7c7e88c9f20df835ce51ef5bbcf1134c378a7761. ID 65 is unused.
MODEL_VOCAB: dict[str, int] = {
    "|": 0,
    " ": 0,
    "1": 1,
    "[": 2,
    "]": 3,
    "a": 4,
    "b": 5,
    "d": 6,
    "e": 7,
    "f": 8,
    "h": 9,
    "i": 10,
    "j": 11,
    "k": 12,
    "l": 13,
    "m": 14,
    "n": 15,
    "o": 16,
    "p": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "¡": 27,
    "«": 28,
    "»": 29,
    "ç": 30,
    "ð": 31,
    "ø": 32,
    "ŋ": 33,
    "œ": 34,
    "ɐ": 35,
    "ɑ": 36,
    "ɒ": 37,
    "ɔ": 38,
    "ɕ": 39,
    "ə": 40,
    "ɛ": 41,
    "ɜ": 42,
    "ɡ": 43,
    "ɨ": 44,
    "ɪ": 45,
    "ɲ": 46,
    "ɹ": 47,
    "ɾ": 48,
    "ʃ": 49,
    "ʊ": 50,
    "ʌ": 51,
    "ʑ": 52,
    "ʒ": 53,
    "ʔ": 54,
    "ʲ": 55,
    "ˈ": 56,
    "ˌ": 57,
    "ː": 58,
    "̃": 59,
    "θ": 60,
    "—": 61,
    "“": 62,
    "”": 63,
    "…": 64,
    "[UNK]": 66,
    "[PAD]": 67,
    "<s>": 68,
    "</s>": 69,
}

BLANK_ID = 67
WORD_DELIMITER = "|"
NON_SCORING_TOKENS = {WORD_DELIMITER, "ˈ", "ˌ", "ː"}
DISPLAY_IGNORED = {WORD_DELIMITER}
TIE_MARKS = {"͡", "͜", "\u200d"}
NORMALIZATION_MAP = {
    "g": "ɡ",  # ASCII g -> IPA script g used by this model.
    ":": "ː",
}
# The selected model omits German short /ʏ/. Keep the expected display symbol,
# align it through the closest available front-rounded label, and never score it.
ALIGNMENT_ONLY_APPROXIMATIONS = {"ʏ": "y"}


class UnsupportedPhonemeError(ValueError):
    pass


@dataclass(frozen=True)
class InventoryToken:
    symbol: str
    token_id: int
    scoreable: bool


def normalize_ipa(raw: str) -> str:
    """Normalize eSpeak IPA into the model's character-level inventory."""
    text = unicodedata.normalize("NFC", raw.strip())
    text = text.replace("_", "")
    for mark in TIE_MARKS:
        text = text.replace(mark, "")
    return "".join(NORMALIZATION_MAP.get(char, char) for char in text)


def tokenize_ipa(raw: str, *, allow_spaces: bool = False) -> list[InventoryToken]:
    normalized = normalize_ipa(raw)
    result: list[InventoryToken] = []
    unsupported: set[str] = set()
    for char in normalized:
        if char.isspace():
            if allow_spaces and (not result or result[-1].symbol != WORD_DELIMITER):
                result.append(
                    InventoryToken(WORD_DELIMITER, MODEL_VOCAB[WORD_DELIMITER], False)
                )
            continue
        if char in ALIGNMENT_ONLY_APPROXIMATIONS:
            proxy = ALIGNMENT_ONLY_APPROXIMATIONS[char]
            result.append(InventoryToken(char, MODEL_VOCAB[proxy], False))
            continue
        if char in MODEL_VOCAB and char not in {"[", "]"}:
            result.append(
                InventoryToken(char, MODEL_VOCAB[char], char not in NON_SCORING_TOKENS)
            )
        else:
            unsupported.add(char)
    if unsupported:
        shown = " ".join(repr(char) for char in sorted(unsupported))
        raise UnsupportedPhonemeError(
            f"G2P produced symbols outside the acoustic model vocabulary: {shown}"
        )
    while result and result[-1].symbol == WORD_DELIMITER:
        result.pop()
    return result


def id_to_symbol() -> dict[int, str]:
    preferred = {value: key for key, value in MODEL_VOCAB.items()}
    preferred[0] = WORD_DELIMITER
    return preferred


def verify_runtime_vocab(runtime_vocab: dict[str, int]) -> None:
    required = {
        token: token_id
        for token, token_id in MODEL_VOCAB.items()
        if token not in {" "}
    }
    mismatches = {
        token: (token_id, runtime_vocab.get(token))
        for token, token_id in required.items()
        if runtime_vocab.get(token) != token_id
    }
    if mismatches:
        raise RuntimeError(f"Downloaded model vocabulary does not match: {mismatches}")
