from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from anki_reader import clean_field, load_anki_database


def make_database(tmp_path: Path) -> bytes:
    path = tmp_path / "collection.anki21"
    model = {
        "42": {
            "flds": [
                {"name": "German"},
                {"name": "English"},
                {"name": "German Sentence"},
            ]
        }
    }
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE col (models TEXT)")
        connection.execute(
            "CREATE TABLE notes (id INTEGER, mid INTEGER, flds TEXT, tags TEXT)"
        )
        connection.execute("INSERT INTO col VALUES (?)", (json.dumps(model),))
        connection.execute(
            "INSERT INTO notes VALUES (?, ?, ?, ?)",
            (
                1,
                42,
                "Hilfe\x1fhelp\x1f<b>Kannst</b> du {{c1::mir}} helfen?",
                " A1-L1 favorite ",
            ),
        )
    return path.read_bytes()


def test_load_anki_database(tmp_path: Path):
    result = load_anki_database(make_database(tmp_path))

    assert result["tags"] == ["A1-L1", "favorite"]
    assert result["notes"][0]["fields"] == {
        "German": "Hilfe",
        "English": "help",
        "German Sentence": "Kannst du mir helfen?",
    }


def test_clean_field_removes_audio_and_markup():
    assert clean_field("<div>Hallo&nbsp;Welt</div>[sound:test.mp3]") == "Hallo Welt"


def test_rejects_non_sqlite_data():
    with pytest.raises(ValueError, match="valid Anki"):
        load_anki_database(b"not a database")
