"""Export transcripts to a meetings-mcp server.

meetings-mcp is a separate self-hosted archive: it watches an `incoming/`
directory for per-meeting staging folders, each holding `transcript.md` plus a
`metadata.json` sidecar, and ingests them into a searchable SQLite index.

The transcript.md it parses is the AudioLogger/WhisperX layout:

    # Recording: 2026-11-14 15:32:06
    **Duration:** 01:23:45
    **Source:** tapeback
    **Model:** large-v3-turbo

    **[00:00:01] Me:** first utterance
    **[00:12:40] Speaker 1:** reply

This module renders Segments into that layout, writes the staging folder under
the local export dir, and uploads it with `scp` — the same mechanism the
Windows clients' Sync-Meetings.ps1 uses. Upload failure is not fatal: the
staging folder stays on disk and `sync_pending` (the `tapeback meetings-sync`
command, or the next export) retries it.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tapeback import const

if TYPE_CHECKING:
    from tapeback.models import Segment
    from tapeback.settings import Settings

StatusCallback = Callable[[str], None]

TRANSCRIPT_NAME = "transcript.md"
METADATA_NAME = "metadata.json"

# The recorder names sessions YYYY-MM-DD_HH-MM-SS, but `tapeback start <name>`
# and `tapeback process <file> --name <name>` accept anything, so a session can
# carry no timestamp — meetings-mcp then falls back to metadata.recorded_at.
_SESSION_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


def _noop_status(msg: str) -> None:
    pass


def _hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _session_recorded_at(session_name: str) -> str | None:
    """Naive local 'YYYY-MM-DDTHH:MM:SS' from a session named YYYY-MM-DD_HH-MM-SS."""
    m = _SESSION_TS_RE.match(session_name)
    if not m:
        return None
    return f"{m[1]}T{m[2]}:{m[3]}:{m[4]}"


def _speaker_label(speaker: str | None) -> str:
    """Map tapeback labels onto the AudioLogger convention meetings-mcp archives.

    The mic channel is "You" in tapeback notes and "Me" in AudioLogger
    transcripts; "Me" keeps speaker attribution consistent with meetings
    ingested from the Windows machines.
    """
    if speaker == const.SPEAKER_YOU:
        return "Me"
    return speaker or const.SPEAKER_OTHER


def export_root(settings: Settings) -> Path:
    """Local staging root for outgoing bundles. Default: XDG data dir."""
    if settings.meetings_export_dir is not None:
        return settings.meetings_export_dir
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "tapeback" / "meetings-export"


def render_transcript(
    segments: list[Segment],
    *,
    session_name: str,
    duration_seconds: float,
    language: str,
    model: str,
) -> str:
    """Render segments as an AudioLogger-format transcript.md.

    One utterance per segment, chronological — the format meetings-mcp's
    parser splits into searchable chunks. Segments carry their final speaker
    already, so no merging is needed here.
    """
    lines: list[str] = []
    recorded_at = _session_recorded_at(session_name)
    if recorded_at:
        lines.append(f"# Recording: {recorded_at.replace('T', ' ')}")
    else:
        lines.append(f"# Recording: {session_name}")
    lines += [
        "",
        f"**Duration:** {_hms(duration_seconds)}",
        "**Source:** tapeback",
        f"**Model:** {model}",
        "",
        "---",
        "",
    ]
    for seg in sorted(segments, key=lambda s: s.start):
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"**[{_hms(seg.start)}] {_speaker_label(seg.speaker)}:** {text}")
    lines.append("")
    return "\n".join(lines)


def _write_bundle(
    root: Path,
    *,
    session_name: str,
    markdown: str,
    metadata: dict[str, object],
) -> Path:
    """Write <root>/<session>-<rand>/transcript.md + metadata.json; return the dir."""
    staging = root / f"{session_name}-{secrets.token_hex(3)}"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / TRANSCRIPT_NAME).write_text(markdown, encoding="utf-8")
    (staging / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return staging


def _upload(staging: Path, remote: str, incoming: str) -> bool:
    """scp the staging folder into <remote>:<incoming>/. True on success."""
    result = subprocess.run(
        ["scp", "-q", "-r", str(staging), f"{remote}:{incoming}/"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def sync_pending(
    settings: Settings,
    on_status: StatusCallback = _noop_status,
) -> int:
    """Upload every staging folder still in the export dir. Returns sent count.

    A folder that still holds transcript.md was never uploaded (successful
    uploads remove it). Called after each export and by `tapeback meetings-sync`.
    """
    if not settings.meetings_remote:
        return 0
    root = export_root(settings)
    if not root.is_dir():
        return 0
    sent = 0
    for staging in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (staging / TRANSCRIPT_NAME).is_file():
            continue
        if _upload(staging, settings.meetings_remote, settings.meetings_incoming):
            shutil.rmtree(staging, ignore_errors=True)
            sent += 1
        else:
            on_status(f"meetings-mcp: upload failed, kept for retry: {staging.name}")
    return sent


def export(
    segments: list[Segment],
    *,
    session_name: str,
    info: dict[str, str | float],
    settings: Settings,
    on_status: StatusCallback = _noop_status,
) -> Path | None:
    """Render + stage + upload one transcript for meetings-mcp. None if disabled.

    Enabled by TAPEBACK_MEETINGS_MACHINE; without it the pipeline changes
    nothing. Never raises: an export problem must not break the pipeline that
    already saved the note — the bundle (if written) stays queued for retry.
    """
    if not settings.meetings_machine:
        return None
    try:
        duration = float(info.get("duration", 0.0))
        language = str(info.get("language", settings.language))
        markdown = render_transcript(
            segments,
            session_name=session_name,
            duration_seconds=duration,
            language=language,
            model=settings.stt_model,
        )
        metadata: dict[str, object] = {
            "source_machine": settings.meetings_machine,
            "source_application": "tapeback",
            "language": language,
            "transcription_engine": "faster-whisper"
            if settings.stt_backend == "local"
            else settings.stt_backend,
            "transcription_model": settings.stt_model,
            "duration_seconds": duration,
        }
        recorded_at = _session_recorded_at(session_name)
        if recorded_at:
            metadata["recorded_at"] = recorded_at

        staging = _write_bundle(
            export_root(settings),
            session_name=session_name,
            markdown=markdown,
            metadata=metadata,
        )
        on_status(f"meetings-mcp: staged {staging.name}")

        if settings.meetings_remote:
            sent = sync_pending(settings, on_status)
            if sent:
                on_status(f"meetings-mcp: uploaded {sent} transcript(s)")
        return staging
    except Exception as exc:  # export must never kill the pipeline that saved the note
        on_status(f"meetings-mcp: export failed ({exc}); note was still saved")
        return None
