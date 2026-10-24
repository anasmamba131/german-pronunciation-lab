from __future__ import annotations

import re
import os
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from phoneme_inventory import InventoryToken, tokenize_ipa


WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüßẞ]+(?:[-’'][A-Za-zÄÖÜäöüßẞ]+)*")
LANGUAGE_FLAG_RE = re.compile(r"\([a-z]{2,3}(?:-[a-z]{2,3})?\)", re.IGNORECASE)


class G2PError(RuntimeError):
    pass


@dataclass(frozen=True)
class WordToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ExpectedPhone:
    symbol: str
    token_id: int
    word_index: int
    letters: str | None
    grapheme_start: int | None
    grapheme_end: int | None
    scoreable: bool


@dataclass(frozen=True)
class ExpectedPronunciation:
    sentence: str
    words: tuple[WordToken, ...]
    phones: tuple[ExpectedPhone, ...]

    @property
    def model_token_ids(self) -> list[int]:
        return [phone.token_id for phone in self.phones]

    @property
    def display_ipa(self) -> str:
        return "".join(" " if p.symbol == "|" else p.symbol for p in self.phones)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ").replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Enter a German sentence.")
    if len(normalized) > 500:
        raise ValueError("Sentence is too long; use at most 500 characters.")
    if not WORD_RE.search(normalized):
        raise ValueError("The sentence does not contain a German word.")
    return normalized


def extract_words(sentence: str) -> tuple[WordToken, ...]:
    return tuple(
        WordToken(match.group(0), match.start(), match.end())
        for match in WORD_RE.finditer(sentence)
    )


def find_espeak(explicit_path: str | None = None) -> Path | None:
    candidates = [
        explicit_path,
        shutil.which("espeak-ng"),
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def parse_espeak_output(raw: str) -> list[InventoryToken]:
    cleaned = LANGUAGE_FLAG_RE.sub("", raw).strip()
    if not cleaned:
        raise G2PError("eSpeak NG returned no phonemes.")
    return tokenize_ipa(cleaned, allow_spaces=False)


def _find_cluster(word: str, choices: Sequence[str], start: int = 0) -> tuple[int, int] | None:
    lowered = word.casefold()
    matches: list[tuple[int, int]] = []
    for choice in choices:
        index = lowered.find(choice, start)
        if index >= 0:
            matches.append((index, index + len(choice)))
    return min(matches, default=None)


def map_phones_to_graphemes(word: str, symbols: Sequence[str]) -> list[tuple[int, int] | None]:
    """Best-effort ordered mapping; deliberately labelled heuristic in the API."""
    spans: list[tuple[int, int] | None] = [None] * len(symbols)
    reserved_until = 0

    grouped = (
        (("p", "f"), ("pf",)),
        (("t", "s"), ("tz", "z", "ts")),
        (("ʃ", "p"), ("sp",)),
        (("ʃ", "t"), ("st",)),
        (("a", "ɪ"), ("ei", "ai", "ay", "ey")),
        (("a", "ʊ"), ("au",)),
        (("ɔ", "ø"), ("eu", "äu")),
    )
    index = 0
    while index < len(symbols) - 1:
        pair = (symbols[index], symbols[index + 1])
        matched = False
        for phone_pair, clusters in grouped:
            if pair == phone_pair:
                found = _find_cluster(word, clusters, reserved_until)
                if found:
                    spans[index] = spans[index + 1] = found
                    reserved_until = found[1]
                    matched = True
                    break
        index += 2 if matched else 1

    preferences: dict[str, tuple[str, ...]] = {
        "ç": ("ch", "ig"),
        "x": ("ch",),
        "ʃ": ("sch", "sp", "st", "ch"),
        "y": ("ü",),
        "ø": ("ö",),
        "œ": ("ö",),
        "ɛ": ("ä", "e"),
        "ŋ": ("ng", "nk"),
        "ɡ": ("g",),
        "z": ("s",),
        "v": ("w", "v"),
        "f": ("f", "v", "ph"),
        "ʒ": ("j", "g"),
        "j": ("j",),
        "k": ("k", "ck", "c", "g"),
    }
    cursor = 0
    for i, symbol in enumerate(symbols):
        if spans[i] is not None or symbol in {"ˈ", "ˌ"}:
            continue
        if symbol == "ː" and i > 0:
            spans[i] = spans[i - 1]
            continue
        found = _find_cluster(word, preferences.get(symbol, (symbol,)), cursor)
        if found:
            spans[i] = found
            cursor = max(cursor, found[1])

    scoreable_indices = [
        i for i, symbol in enumerate(symbols) if symbol not in {"ˈ", "ˌ", "ː"}
    ]
    for rank, i in enumerate(scoreable_indices):
        if spans[i] is not None:
            continue
        if not word:
            continue
        start = min(len(word) - 1, round(rank * len(word) / max(1, len(scoreable_indices))))
        end = min(len(word), max(start + 1, round((rank + 1) * len(word) / max(1, len(scoreable_indices)))))
        spans[i] = (start, end)
    return spans


class GermanG2P:
    def __init__(
        self,
        executable: str | Path,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.executable = Path(executable)
        self._runner = runner

    def phonemize_word(self, word: str) -> list[InventoryToken]:
        command = [
            str(self.executable),
            "-q",
            "-v",
            "de",
            "-b",
            "1",
            "--ipa=3",
            "--stdin",
        ]
        try:
            environment = os.environ.copy()
            bundled_data = self.executable.parent / "espeak-ng-data"
            if bundled_data.is_dir():
                environment["ESPEAK_DATA_PATH"] = str(bundled_data)
            completed = self._runner(
                command,
                input=(word + "\n").encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=10,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise G2PError(f"Could not run eSpeak NG: {exc}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise G2PError(f"eSpeak NG failed ({completed.returncode}): {stderr}")
        raw = completed.stdout.decode("utf-8", errors="strict")
        return parse_espeak_output(raw)

    def phonemize(self, text: str) -> ExpectedPronunciation:
        sentence = normalize_text(text)
        words = extract_words(sentence)
        expected: list[ExpectedPhone] = []
        for word_index, word in enumerate(words):
            tokens = self.phonemize_word(word.text)
            symbols = [token.symbol for token in tokens]
            spans = map_phones_to_graphemes(word.text, symbols)
            if expected:
                expected.append(
                    ExpectedPhone("|", 0, word_index - 1, None, None, None, False)
                )
            for token, span in zip(tokens, spans, strict=True):
                expected.append(
                    ExpectedPhone(
                        symbol=token.symbol,
                        token_id=token.token_id,
                        word_index=word_index,
                        letters=word.text[span[0] : span[1]] if span else None,
                        grapheme_start=(word.start + span[0]) if span else None,
                        grapheme_end=(word.start + span[1]) if span else None,
                        scoreable=token.scoreable,
                    )
                )
        return ExpectedPronunciation(sentence, words, tuple(expected))
