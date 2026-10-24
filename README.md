# German Pronunciation Lab — Streamlit edition

A public-cloud Streamlit edition of the experimental German pronunciation checker.
It records a short WAV in the browser, generates expected German IPA with eSpeak NG,
runs the pinned `q1805/hubert_large-german-IPA-v2` model, aligns phonemes, and displays
experimental phone and word evidence.

## Important limitations

- The score is experimental and is not a clinical, educationally validated, or
  accent-neutral judgment.
- The acoustic model is licensed CC BY-NC 4.0 and this app is for non-commercial use.
- The first analysis downloads approximately 1.26 GB of model weights and then loads
  them into memory. Streamlit Community Cloud may throttle or stop the app if it exceeds
  the available memory.
- Recordings are processed in memory and are not intentionally persisted by the app.
- The Windows-only Thorsten TTS is excluded. Anki notes can be loaded from a
  `collection.anki21` or `collection.anki2` file for the current browser session only.
  The temporary server-side copy is deleted immediately after parsing and the file is
  never committed to the public app. Visitors must upload their own database.

## Use an Anki deck privately

1. Open the public app and find **Your private study deck**.
2. Select **Upload collection.anki21 or collection.anki2**.
3. Choose the small extracted Anki database, not the large `.apkg` package.
4. Pick a lesson/tag, browse the notes, and select **Use for pronunciation**.

The selected database and study position are isolated to that Streamlit browser
session. Closing the tab, clearing the session, or an app restart requires another
upload. Uploaded databases are limited to 20 MB and 10,000 notes.

## Run locally

Use Python 3.12 and install eSpeak NG first. Then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MODEL_PATH = "$env:USERPROFILE\models\german-pronunciation-model"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Create an app and select `streamlit_app.py` as the entrypoint.
4. Select Python 3.12 in Advanced settings.
5. Deploy and watch the build logs. `packages.txt` installs Linux eSpeak NG and
   `requirements.txt` installs the Python dependencies.

No secrets are required.
