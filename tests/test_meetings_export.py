"""meetings-mcp export: the AudioLogger-format bundle and its scp upload."""

import json
import os
from pathlib import Path

from tapeback.meetings_export import export, render_transcript, sync_pending
from tapeback.models import Segment
from tapeback.settings import Settings


def _segments() -> list[Segment]:
    return [
        Segment(start=12.4, end=15.0, text="reply from the call", speaker="Speaker 1"),
        Segment(start=1.2, end=4.0, text="first thing I said", speaker="You"),
        Segment(start=40.0, end=42.0, text="unattributed", speaker=None),
        Segment(start=50.0, end=51.0, text="   ", speaker="You"),
    ]


def _export_settings(tmp_path, **overrides) -> Settings:
    """Export enabled into a tmp staging dir, no remote configured."""
    return Settings(
        vault_path=tmp_path / "vault",
        meetings_machine="pop-desktop",
        meetings_export_dir=tmp_path / "export",
        **overrides,
    )


def _fake_scp(tmp_path, monkeypatch, rc: int) -> Path:
    """An `scp` on PATH that appends its argv to a log and exits with rc.

    Returns the log file; call again to change the answer (network down, then back).
    """
    bin_dir = tmp_path / "fake-scp-bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "scp_calls.log"
    script = bin_dir / "scp"
    script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit {rc}\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return log


def test_render_transcript_audiologger_layout():
    md = render_transcript(
        _segments(),
        session_name="2026-11-14_15-32-06",
        duration_seconds=3723.0,
        language="en",
        model="large-v3-turbo",
    )
    lines = md.splitlines()

    assert lines[0] == "# Recording: 2026-11-14 15:32:06"
    assert "**Duration:** 01:02:03" in lines
    assert "**Source:** tapeback" in lines
    assert "**Model:** large-v3-turbo" in lines

    utterances = [line for line in lines if line.startswith("**[")]
    # Sorted by start, the blank-text segment dropped.
    assert utterances == [
        "**[00:00:01] Me:** first thing I said",
        "**[00:00:12] Speaker 1:** reply from the call",
        "**[00:00:40] Other:** unattributed",
    ]


def test_render_transcript_custom_session_name_has_no_timestamp():
    md = render_transcript(
        _segments(),
        session_name="weekly-standup",
        duration_seconds=10.0,
        language="en",
        model="m",
    )
    assert md.splitlines()[0] == "# Recording: weekly-standup"


def test_export_disabled_without_machine(tmp_path):
    settings = Settings(vault_path=tmp_path / "vault", meetings_export_dir=tmp_path / "export")
    assert export(_segments(), session_name="s", info={}, settings=settings) is None
    assert not (tmp_path / "export").exists()


def test_export_writes_bundle(tmp_path):
    staging = export(
        _segments(),
        session_name="2026-11-14_15-32-06",
        info={"duration": 125.0, "language": "en"},
        settings=_export_settings(tmp_path),
    )
    assert staging is not None
    assert (staging / "transcript.md").read_text().startswith("# Recording: 2026-11-14 15:32:06")

    metadata = json.loads((staging / "metadata.json").read_text())
    assert metadata == {
        "source_machine": "pop-desktop",
        "source_application": "tapeback",
        "language": "en",
        "transcription_engine": "faster-whisper",
        "transcription_model": "large-v3-turbo",
        "duration_seconds": 125.0,
        "recorded_at": "2026-11-14T15:32:06",
    }


def test_export_remote_stt_engine_label(tmp_path):
    settings = _export_settings(tmp_path, stt_backend="openai", stt_model="whisper-1")
    staging = export(_segments(), session_name="s", info={}, settings=settings)
    assert staging is not None
    metadata = json.loads((staging / "metadata.json").read_text())
    assert metadata["transcription_engine"] == "openai"
    assert metadata["transcription_model"] == "whisper-1"


def test_sync_pending_uploads_and_clears(tmp_path, monkeypatch):
    log = _fake_scp(tmp_path, monkeypatch, rc=0)
    settings = _export_settings(tmp_path, meetings_remote="me@host.tailnet.ts.net")
    staging = export(
        _segments(),
        session_name="2026-11-14_15-32-06",
        info={},
        settings=settings,
    )
    assert staging is not None
    # export() already synced: the bundle is gone after a successful upload.
    assert not staging.exists()
    calls = log.read_text().strip()
    assert "-q -r" in calls
    assert "me@host.tailnet.ts.net:/opt/meetings/incoming/" in calls
    assert sync_pending(settings) == 0


def test_sync_pending_retries_failed_upload(tmp_path, monkeypatch):
    settings = _export_settings(tmp_path, meetings_remote="me@host.tailnet.ts.net")
    log = _fake_scp(tmp_path, monkeypatch, rc=1)
    staging = export(
        _segments(),
        session_name="2026-11-14_15-32-06",
        info={},
        settings=settings,
    )
    assert staging is not None
    # Upload failed — the bundle stays queued on disk.
    assert (staging / "transcript.md").is_file()

    # Network comes back: the same bundle uploads and is removed.
    _fake_scp(tmp_path, monkeypatch, rc=0)
    assert sync_pending(settings) == 1
    assert not staging.exists()
    assert log.read_text().count("me@host.tailnet.ts.net") == 2


def test_sync_pending_no_remote_is_noop(tmp_path):
    settings = _export_settings(tmp_path)
    staging = export(
        _segments(),
        session_name="s",
        info={},
        settings=settings,
    )
    assert staging is not None
    assert sync_pending(settings) == 0
    # Without a remote nothing is sent and nothing is deleted.
    assert (staging / "transcript.md").is_file()


def test_export_never_raises(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("tapeback.meetings_export._write_bundle", _boom)
    assert (
        export(
            _segments(),
            session_name="s",
            info={},
            settings=_export_settings(tmp_path),
        )
        is None
    )
