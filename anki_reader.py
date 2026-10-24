from __future__ import annotations

from contextlib import closing
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any


MAX_DATABASE_BYTES = 20 * 1024 * 1024
MAX_NOTES = 10_000
SQLITE_HEADER = b"SQLite format 3\x00"
SOUND_RE = re.compile(r"\[sound:[^\]]+\]", re.IGNORECASE)
CLOZE_RE = re.compile(r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_field(value: str | None) -> str:
    if not value:
        return ""

    value = SOUND_RE.sub("", value)
    value = CLOZE_RE.sub(lambda match: match.group(1), value)
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        text = " ".join(parser.parts)
    except Exception:
        text = value
    return WHITESPACE_RE.sub(" ", unescape(text)).strip()


def natural_key(value: str) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def _model_fields(models: dict[str, Any], model_id: int) -> list[str]:
    model = models.get(str(model_id), {})
    return [
        field.get("name", f"Field {index}")
        for index, field in enumerate(model.get("flds", []))
    ]


def load_anki_database(database_bytes: bytes) -> dict[str, Any]:
    """Read an uploaded Anki SQLite database without retaining it on disk."""
    if not database_bytes.startswith(SQLITE_HEADER):
        raise ValueError("This is not a valid Anki SQLite database.")
    if len(database_bytes) > MAX_DATABASE_BYTES:
        raise ValueError("The Anki database is larger than the 20 MB session limit.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".anki21", delete=False) as temporary:
            temporary.write(database_bytes)
            temporary_path = Path(temporary.name)

        database_uri = f"{temporary_path.as_uri()}?mode=ro"
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            try:
                connection.execute("PRAGMA trusted_schema = OFF")
            except sqlite3.DatabaseError:
                pass

            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"col", "notes"}.issubset(table_names):
                raise ValueError("The uploaded database does not contain Anki notes.")

            model_row = connection.execute("SELECT models FROM col LIMIT 1").fetchone()
            models = json.loads(model_row[0]) if model_row and model_row[0] else {}
            rows = connection.execute(
                "SELECT id, mid, flds, tags FROM notes ORDER BY id LIMIT ?",
                (MAX_NOTES + 1,),
            ).fetchall()

        if len(rows) > MAX_NOTES:
            raise ValueError(f"This deck has more than {MAX_NOTES:,} notes.")

        notes: list[dict[str, Any]] = []
        tags: set[str] = set()
        for note_id, model_id, fields_blob, tags_blob in rows:
            field_names = _model_fields(models, model_id)
            field_values = fields_blob.split("\x1f")
            fields = {
                (
                    field_names[index]
                    if index < len(field_names)
                    else f"Field {index}"
                ): clean_field(value)
                for index, value in enumerate(field_values)
            }
            note_tags = [tag for tag in tags_blob.strip().split() if tag]
            tags.update(note_tags)
            notes.append(
                {
                    "id": int(note_id),
                    "tags": note_tags,
                    "fields": fields,
                }
            )

        if not notes:
            raise ValueError("The uploaded Anki database contains no notes.")

        return {
            "notes": notes,
            "tags": sorted(tags, key=natural_key),
        }
    except (json.JSONDecodeError, sqlite3.DatabaseError) as exc:
        raise ValueError("The Anki database could not be read safely.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
