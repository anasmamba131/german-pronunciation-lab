from __future__ import annotations

import hashlib

import streamlit as st

from cloud_runtime import create_service, model_path
from schemas import CheckResponse


st.set_page_config(
    page_title="German Pronunciation Lab",
    page_icon=":material/graphic_eq:",
    layout="centered",
)


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


def render_result(result: CheckResponse) -> None:
    score = result.overall_score
    with st.container(border=True):
        score_column, status_column = st.columns(2)
        score_column.metric(
            "Overall model match",
            "—" if score is None else f"{round(score)}%",
            border=True,
        )
        status_column.metric(
            "Assessment",
            status_label(result.overall_status),
            border=True,
        )
        if score is not None:
            st.progress(max(0, min(100, round(score))))
        st.caption(
            "Experimental model evidence—not a clinical, accent-neutral, or "
            "educationally validated grade."
        )

    st.subheader("Pronunciation evidence")
    st.markdown(f"**Expected IPA:** `/{result.expected_phonemes}/`")
    decoded = result.decoded_phonemes or "No stable greedy output"
    st.markdown(f"**Model output:** `/{decoded}/`")

    for word in result.words:
        word_score = "—" if word.score is None else f"{round(word.score)}%"
        with st.container(border=True, horizontal=True):
            st.markdown(f"**{word.word}**")
            st.caption(f"{word.start:.2f}–{word.end:.2f} s")
            st.badge(
                f"{word_score} · {status_label(word.status)}",
                color={
                    "correct": "green",
                    "needs_work": "orange",
                    "uncertain": "gray",
                }.get(word.status, "gray"),
            )

    phone_rows = []
    for phone in result.phonemes:
        phone_rows.append(
            {
                "Letters": phone.letters or "—",
                "Expected": f"/{phone.expected}/",
                "Score": "—" if phone.score is None else f"{round(phone.score)}%",
                "Status": status_label(phone.status),
                "Alternative": (
                    f"/{phone.best_alternative}/" if phone.best_alternative else "—"
                ),
                "Time": f"{phone.start:.2f}–{phone.end:.2f} s",
            }
        )
    st.dataframe(phone_rows, hide_index=True, width="stretch")

    warnings = [*result.audio.warnings, *result.warnings]
    for warning in dict.fromkeys(warnings):
        st.warning(warning, icon=":material/warning:")

    with st.expander("Method and audio diagnostics", icon=":material/science:"):
        st.write(result.methodology)
        st.json(
            {
                "duration_seconds": result.audio.duration_seconds,
                "original_sample_rate": result.audio.original_sample_rate,
                "analysis_sample_rate": result.audio.analysis_sample_rate,
                "peak": result.audio.peak,
                "rms": result.audio.rms,
                "clipping_ratio": result.audio.clipping_ratio,
                "timing_status": result.timing_status,
                "score_status": result.score_status,
            }
        )


st.title("German pronunciation lab")
st.write(
    "Record a German sentence and compare its phoneme evidence with the expected "
    "pronunciation. The first analysis may take a few minutes while the model downloads."
)
st.badge("Experimental", icon=":material/science:", color="orange")

sentence = st.text_area(
    "German sentence",
    value="Entschuldigung, können Sie mir helfen?",
    max_chars=500,
    help="Keep the sentence short for faster CPU analysis.",
)
recording = st.audio_input(
    "Record yourself reading the sentence",
    sample_rate=16_000,
    key="learner_recording",
    help="A clear recording of 3–8 seconds works best.",
)

audio_bytes = recording.getvalue() if recording is not None else b""
current_key = request_key(sentence.strip(), audio_bytes) if audio_bytes else None

analyze = st.button(
    "Check pronunciation",
    type="primary",
    icon=":material/hearing:",
    disabled=not sentence.strip() or not audio_bytes,
    width="stretch",
)

if analyze:
    try:
        with st.status("Preparing the speech engine…", expanded=True) as status:
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
    render_result(CheckResponse.model_validate(stored_result))

st.caption(
    f"Model: q1805/hubert_large-german-IPA-v2 · CPU inference · "
    f"temporary model cache: {model_path()}"
)
