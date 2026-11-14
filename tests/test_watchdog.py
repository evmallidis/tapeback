"""Auto-stop watchdog: the duration cap and the both-channels-silent trigger."""

import time
from pathlib import Path

import numpy as np

from tapeback._watchdog import RecordingWatchdog, _tail_rms
from tapeback.settings import Settings


def _pcm(path: Path, value: int) -> Path:
    """A 'recording' whose whole content is one constant 16-bit sample value."""
    samples = np.full(96000 * 4, value, dtype=np.int16)
    path.write_bytes(samples.tobytes())
    return path


def _watchdog(tmp_path, settings, started=0.0):
    return RecordingWatchdog(
        _pcm(tmp_path / "monitor.wav", 0),
        _pcm(tmp_path / "mic.wav", 0),
        settings,
        started_monotonic=started,
    )


def test_duration_cap_fires_at_exactly_the_cap(tmp_path):
    wd = _watchdog(tmp_path, Settings(auto_stop_hours=1.0))
    assert wd.check_once(3599.9) is None
    assert wd.check_once(3600.0) == "recording reached the 1.0h auto-stop cap"


def test_cap_zero_disables_the_trigger(tmp_path):
    wd = _watchdog(tmp_path, Settings(auto_stop_hours=0.0))
    assert wd.check_once(999999.0) is None


def test_silence_trigger_needs_sustained_silence(tmp_path):
    wd = _watchdog(tmp_path, Settings(auto_stop_silence_minutes=30.0))
    # First silent check only starts the clock.
    assert wd.check_once(0.0) is None
    assert wd.check_once(60.0) is None
    assert wd.check_once(1800.0) == "silent on both channels for 30.0 min"


def test_speech_resets_the_silence_clock(tmp_path):
    wd = _watchdog(tmp_path, Settings(auto_stop_silence_minutes=30.0))
    assert wd.check_once(0.0) is None
    # Speech on the monitor at t=60 — the silence streak breaks.
    _pcm(tmp_path / "monitor.wav", 10000)
    assert wd.check_once(60.0) is None
    _pcm(tmp_path / "monitor.wav", 0)
    # The clock restarts from t=120, not from t=0.
    assert wd.check_once(120.0) is None
    assert wd.check_once(1919.0) is None
    assert wd.check_once(1920.0) == "silent on both channels for 30.0 min"


def test_one_audible_channel_keeps_the_recording(tmp_path):
    settings = Settings(auto_stop_silence_minutes=30.0)
    wd = RecordingWatchdog(
        _pcm(tmp_path / "monitor.wav", 0),
        _pcm(tmp_path / "mic.wav", 10000),
        settings,
        started_monotonic=0.0,
    )
    assert wd.check_once(0.0) is None
    assert wd.check_once(100000.0) is None


def test_silence_zero_disables_the_trigger(tmp_path):
    wd = _watchdog(tmp_path, Settings(auto_stop_silence_minutes=0.0))
    assert wd.check_once(0.0) is None
    assert wd.check_once(100000.0) is None


def test_missing_or_short_file_reads_as_silence(tmp_path):
    assert _tail_rms(tmp_path / "nope.wav", 96000, 2.0) == 0.0


def test_tail_rms_reads_only_the_tail(tmp_path):
    # Loud start, silent tail: the measurement window is the last seconds only.
    samples = np.concatenate(
        [np.full(96000 * 4, 10000, dtype=np.int16), np.zeros(96000 * 4, dtype=np.int16)]
    )
    path = tmp_path / "audio.wav"
    path.write_bytes(samples.tobytes())
    assert _tail_rms(path, 96000, 2.0) == 0.0
    assert _tail_rms(path, 96000, 10.0) > 0.0


def test_close_without_start_is_safe(tmp_path):
    # Auto-stop off (the default): the watchdog is constructed but never started.
    wd = _watchdog(tmp_path, Settings())
    wd.close()


def test_thread_sets_stop_reason_and_close_joins(tmp_path):
    # The real thread path: a tiny cap and a fast poll, no fake clock.
    wd = _watchdog(tmp_path, Settings(auto_stop_hours=0.000001))
    wd._poll_seconds = 0.01
    wd.start()
    for _ in range(100):
        if wd.stop_reason:
            break
        time.sleep(0.02)
    assert wd.stop_reason is not None
    wd.close()
