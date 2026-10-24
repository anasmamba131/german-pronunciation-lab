from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import streamlit as st

from cloud_runtime import create_service, model_path
from schemas import CheckResponse


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="German Pronunciation Lab",
    page_icon=":material/graphic_eq:",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.html(APP_DIR / "app_styles.css")


@st.cache_resource(
    show_spinner="Downloading and loading the German speech model…",
    show_time=True,
)
def pronunciation_service():
    return create_service()


def request_key(sentence: str, audio_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(sentence.encode("utf-8"))
    digest.update(b"\0")
    digest.update(audio_bytes)
    return digest.hexdigest()


def status_label(status: str) -> str:
    return {
        "correct": "Strong match",
        "needs_work": "Needs review",
        "uncertain": "Uncertain",
    }.get(status, status.replace("_", " ").title())


def render_hero() -> None:
    st.html(
        """
        <header class="lab-hero">
          <div>
            <p class="eyebrow">PUBLIC CLOUD · RECORDINGS NOT SAVED · EXPERIMENTAL</p>
            <h1>German<br><span>Pronunciation Lab</span></h1>
            <p class="lede">
              Read one sentence aloud. The German speech model aligns every
              expected IPA sound to your recording and shows where the evidence
              is strong—or uncertain.
            </p>
          </div>
          <div class="system-card" aria-label="Cloud engine status">
            <span class="health-dot ready"></span>
            <div>
              <strong>Cloud engine ready</strong>
              <small>The model loads on your first check</small>
            </div>
          </div>
        </header>
        """
    )


def render_speech_button(sentence: str) -> None:
    """Speak the sentence with a German voice available in the browser."""
    button_id = f"tts-{hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:12]}"
    safe_button_id = json.dumps(button_id)
    safe_sentence = (
        json.dumps(sentence, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    st.html(
        f"""
        <button id="{button_id}" class="tts-button" type="button">
          <span aria-hidden="true">&#128266;</span>
          <span class="tts-label">Hear sentence</span>
        </button>
        <script>
          (() => {{
            const button = document.getElementById({safe_button_id});
            if (!button || button.dataset.bound === "true") return;
            button.dataset.bound = "true";
            button.addEventListener("click", () => {{
              const sentence = {safe_sentence};
              if (!sentence.trim() || !("speechSynthesis" in window)) return;
              window.speechSynthesis.cancel();
              const utterance = new SpeechSynthesisUtterance(sentence);
              utterance.lang = "de-DE";
              const voices = window.speechSynthesis.getVoices();
              const germanVoice = voices.find((voice) =>
                voice.lang && voice.lang.toLowerCase().startsWith("de")
              );
              if (germanVoice) utterance.voice = germanVoice;
              const label = button.querySelector(".tts-label");
              label.textContent = "Speaking…";
              utterance.onend = () => {{ label.textContent = "Hear sentence"; }};
              utterance.onerror = () => {{ label.textContent = "Hear sentence"; }};
              window.speechSynthesis.speak(utterance);
            }});
          }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def result_copy(status: str) -> str:
    return {
        "correct": "The model found strong evidence for the expected pronunciation.",
        "needs_work": "Some sounds deserve another listen. Review the highlighted phonemes below.",
        "uncertain": "The model evidence is uncertain. Try a clearer recording in a quiet room.",
    }.get(status, "Review the aligned pronunciation evidence below.")


def render_result(result: CheckResponse) -> None:
    score = result.overall_score
    score_text = "—" if score is None else str(round(score))
    ring_score = 0 if score is None else max(0, min(100, round(score)))
    overall_status = html.escape(status_label(result.overall_status))
    copy = html.escape(result_copy(result.overall_status))

    st.html(
        f"""
        <section class="result-head">
          <div class="score-ring" style="--score:{ring_score}">
            <div><strong>{score_text}</strong><span>experimental</span></div>
          </div>
          <div>
            <p class="eyebrow">OVERALL RESULT</p>
            <h2>{overall_status}</h2>
            <p>{copy}</p>
          </div>
        </section>
        <section class="transcription-grid">
          <div>
            <span>Expected IPA</span>
            <strong>/{html.escape(result.expected_phonemes)}/</strong>
          </div>
          <div>
            <span>Greedy model output</span>
            <strong>/{html.escape(result.decoded_phonemes or 'No stable output')}/</strong>
          </div>
        </section>
        <div class="section-heading">
          <div><p class="eyebrow">WORD BY WORD</p><h2>Aligned pronunciation</h2></div>
          <div class="legend">
            <span class="good">strong</span>
            <span class="warn">review</span>
            <span class="muted">uncertain</span>
          </div>
        </div>
        """
    )

    for word in result.words:
        score_label = "—" if word.score is None else f"{round(word.score)}%"
        phone_html: list[str] = []
        for phone_index in word.phoneme_indices:
            if phone_index < 0 or phone_index >= len(result.phonemes):
                continue
            phone = result.phonemes[phone_index]
            phone_score = "—" if phone.score is None else f"{round(phone.score)}%"
            phone_html.append(
                f"""
                <div class="phone {phone.status}">
                  <strong>{html.escape(phone.expected)}</strong>
                  <small>{html.escape(phone.letters or '·')} · {phone_score}</small>
                </div>
                """
            )

        st.html(
            f"""
            <article class="word-card {word.status}">
              <div class="word-name">
                <strong>{html.escape(word.word)}</strong>
                <small>{word.start:.2f}–{word.end:.2f} s</small>
              </div>
              <div class="phone-list">{''.join(phone_html)}</div>
              <div class="word-score">{score_label}</div>
            </article>
            """
        )

    warnings = list(dict.fromkeys([*result.audio.warnings, *result.warnings]))
    with st.container(key="methodology"):
        with st.expander("How this experimental score works", icon=":material/science:"):
            st.write(result.methodology)
            for warning in warnings:
                st.warning(warning, icon=":material/warning:")
            st.caption(
                f"Recording: {result.audio.duration_seconds:.1f} s · "
                f"{result.audio.analysis_sample_rate:,} Hz analysis · "
                f"timing {result.timing_status}"
            )


render_hero()

with st.container(key="workbench", gap="small"):
    st.html('<div class="step-label"><span>01</span> Sentence</div>')
    sentence = st.text_area(
        "German sentence",
        value="Entschuldigung, können Sie mir helfen?",
        max_chars=500,
        help="Keep the sentence short for faster CPU analysis.",
        label_visibility="collapsed",
        key="sentence_input",
    )
    render_speech_button(sentence.strip())

    st.html(
        """
        <div class="record-guide">
          <span class="record-dot"></span>
          <div>
            <strong>Record</strong>
            <small>Press the microphone, then read the sentence naturally.</small>
          </div>
        </div>
        """
    )
    recording = st.audio_input(
        "Record yourself reading the sentence",
        sample_rate=16_000,
        key="learner_recording",
        help="A clear recording of 3–8 seconds works best.",
        label_visibility="collapsed",
    )

    audio_bytes = recording.getvalue() if recording is not None else b""
    current_key = request_key(sentence.strip(), audio_bytes) if audio_bytes else None
    analyze = st.button(
        "Check pronunciation  →",
        type="primary",
        disabled=not sentence.strip() or not audio_bytes,
        width="stretch",
        key="check_pronunciation",
    )

if analyze:
    try:
        with st.container(key="analysis_status"):
            with st.status("Listening closely…", expanded=True) as status:
                status.write("Loading the shared German HuBERT model")
                service = pronunciation_service()
                status.write("Aligning the recording with the expected phonemes")
                result = service.analyze(sentence.strip(), audio_bytes)
                status.update(
                    label="Pronunciation analysis complete",
                    state="complete",
                    expanded=False,
                )
        st.session_state["analysis_result"] = result.model_dump(mode="json")
        st.session_state["analysis_request_key"] = current_key
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")

stored_key = st.session_state.get("analysis_request_key")
stored_result = st.session_state.get("analysis_result")
if stored_result and stored_key == current_key:
    with st.container(key="results", gap="small"):
        render_result(CheckResponse.model_validate(stored_result))

st.html(
    f"""
    <footer class="lab-footer">
      Acoustic model: q1805/hubert_large-german-IPA-v2 · CC BY-NC 4.0 ·
      CPU inference · temporary cache: {html.escape(str(model_path()))}
    </footer>
    """
)
