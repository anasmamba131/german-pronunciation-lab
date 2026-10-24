from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from acoustic_model import GermanPhonemeModel, model_files_present
from alignment import ctc_viterbi_align, greedy_decode
from audio import load_wav_bytes
from config import Settings
from g2p import ExpectedPhone, GermanG2P, find_espeak
from phoneme_inventory import ALIGNMENT_ONLY_APPROXIMATIONS, BLANK_ID, id_to_symbol
from schemas import (
    AudioDiagnostics,
    CheckResponse,
    HealthResponse,
    PhonemeResult,
    WordResult,
)
from scoring import aggregate_score, aggregate_status, score_tokens


METHODOLOGY = (
    "Experimental GOP-like score: each expected token is CTC-Viterbi aligned; "
    "the mean expected-token posterior over its aligned frames is compared with "
    "the strongest competing phoneme using a log-posterior ratio, then mapped "
    "monotonically to 0–100. Thresholds and the display mapping have not been "
    "calibrated on German learner-error labels."
)


class PronunciationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.espeak_path = find_espeak(settings.espeak_path)
        self.model = GermanPhonemeModel(
            settings.model_path,
            device=settings.device,
            cpu_threads=settings.cpu_threads,
        )

    def health(self) -> HealthResponse:
        self.espeak_path = find_espeak(self.settings.espeak_path)
        return HealthResponse(
            model_id=self.settings.model_id,
            model_path=str(self.settings.model_path),
            model_downloaded=model_files_present(self.settings.model_path),
            model_loaded=self.model.loaded,
            g2p_available=self.espeak_path is not None,
            g2p_path=str(self.espeak_path) if self.espeak_path else None,
            device_setting=self.settings.device,
        )

    def _save_original(self, wav_bytes: bytes) -> str | None:
        if not self.settings.preserve_recordings:
            return None
        self.settings.recordings_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        recording_id = f"{stamp}_{uuid.uuid4().hex[:10]}"
        target = self.settings.recordings_dir / f"{recording_id}.wav"
        target.write_bytes(wav_bytes)
        return recording_id

    def analyze(self, sentence: str, wav_bytes: bytes) -> CheckResponse:
        self.espeak_path = find_espeak(self.settings.espeak_path)
        if self.espeak_path is None:
            raise RuntimeError(
                "eSpeak NG is not installed. Run .\\scripts\\install_espeak.ps1."
            )
        audio = load_wav_bytes(
            wav_bytes,
            target_sample_rate=self.settings.sample_rate,
            min_seconds=self.settings.min_audio_seconds,
            max_seconds=self.settings.max_audio_seconds,
            silence_rms=self.settings.silence_rms,
        )
        recording_id = self._save_original(wav_bytes)
        pronunciation = GermanG2P(self.espeak_path).phonemize(sentence)
        acoustic = self.model.infer(audio.samples, audio.sample_rate)
        aligned = ctc_viterbi_align(
            acoustic.log_probs,
            pronunciation.model_token_ids,
            blank_id=BLANK_ID,
        )
        token_scores = score_tokens(
            acoustic.log_probs, aligned, pronunciation.phones
        )
        seconds_per_frame = audio.duration_seconds / acoustic.frame_count
        decoded = greedy_decode(
            acoustic.log_probs, blank_id=BLANK_ID, labels=id_to_symbol()
        )

        visible_results: list[PhonemeResult] = []
        response_index_by_target: dict[int, int] = {}
        for target_index, (phone, span, score) in enumerate(
            zip(pronunciation.phones, aligned, token_scores, strict=True)
        ):
            if phone.symbol == "|":
                continue
            response_index_by_target[target_index] = len(visible_results)
            note = None
            if phone.symbol in ALIGNMENT_ONLY_APPROXIMATIONS:
                proxy = ALIGNMENT_ONLY_APPROXIMATIONS[phone.symbol]
                note = (
                    f"The acoustic vocabulary has no /{phone.symbol}/ label; "
                    f"it was aligned through /{proxy}/ and was not scored."
                )
            elif phone.symbol == "ː":
                note = (
                    "Vowel length is aligned but not scored until duration thresholds "
                    "are calibrated on learner speech."
                )
            elif score.status == "needs_work" and score.best_alternative:
                note = (
                    f"The model supported /{score.best_alternative}/ more strongly "
                    f"than expected /{phone.symbol}/ in this aligned region."
                )
            elif score.status == "uncertain" and phone.scoreable:
                note = "Model evidence was not strong enough to identify an error."
            visible_results.append(
                PhonemeResult(
                    expected=phone.symbol,
                    letters=phone.letters,
                    word_index=phone.word_index,
                    start=round(span.start_frame * seconds_per_frame, 3),
                    end=round(span.end_frame * seconds_per_frame, 3),
                    score=score.score,
                    confidence=score.confidence,
                    status=score.status,
                    expected_probability=score.expected_probability,
                    alternative_probability=score.alternative_probability,
                    best_alternative=score.best_alternative,
                    scoring_note=note,
                )
            )

        words: list[WordResult] = []
        for word_index, word in enumerate(pronunciation.words):
            target_indices = [
                i
                for i, phone in enumerate(pronunciation.phones)
                if phone.word_index == word_index and phone.symbol != "|"
            ]
            response_indices = [response_index_by_target[i] for i in target_indices]
            word_scores = [token_scores[i] for i in target_indices]
            starts = [visible_results[i].start for i in response_indices]
            ends = [visible_results[i].end for i in response_indices]
            words.append(
                WordResult(
                    word=word.text,
                    score=aggregate_score(word_scores),
                    status=aggregate_status(word_scores),
                    start=min(starts, default=0.0),
                    end=max(ends, default=0.0),
                    phoneme_indices=response_indices,
                )
            )

        warnings = list(audio.warnings)
        warnings.append(
            "Phoneme-to-letter spans are heuristic in milestone 1 and may be imperfect."
        )
        approximated = sorted(
            {
                phone.symbol
                for phone in pronunciation.phones
                if phone.symbol in ALIGNMENT_ONLY_APPROXIMATIONS
            }
        )
        if approximated:
            warnings.append(
                "Unscored inventory approximation used for: "
                + ", ".join(f"/{phone}/" for phone in approximated)
            )
        return CheckResponse(
            sentence=sentence,
            normalized_sentence=pronunciation.sentence,
            expected_phonemes=pronunciation.display_ipa,
            decoded_phonemes=decoded,
            overall_score=aggregate_score(token_scores),
            overall_status=aggregate_status(token_scores),
            words=words,
            phonemes=visible_results,
            audio=AudioDiagnostics(
                duration_seconds=round(audio.duration_seconds, 3),
                original_sample_rate=audio.original_sample_rate,
                analysis_sample_rate=audio.sample_rate,
                peak=round(audio.peak, 4),
                rms=round(audio.rms, 4),
                clipping_ratio=round(audio.clipping_ratio, 5),
                warnings=list(audio.warnings),
                recording_id=recording_id,
            ),
            methodology=METHODOLOGY,
            warnings=warnings,
        )
