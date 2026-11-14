"""Auto-stop a recording that outlived the meeting.

Two independent triggers, both off by default (0):

- TAPEBACK_AUTO_STOP_HOURS — hard cap on one recording. A recording left
  running overnight is stopped and still transcribed.
- TAPEBACK_AUTO_STOP_SILENCE_MINUTES — stop when *both* channels (mic and
  system audio) stayed silent this long: the meeting ended but recording kept
  running.

The watchdog never kills the recorder itself — it only sets `stop_reason`.
`tapeback start` polls it in its wait loop and then runs the normal
stop-and-process path, so an auto-stopped recording is transcribed exactly
like a Ctrl+C one.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from tapeback.settings import Settings

# Seconds between level checks, and the tail of each channel measured per check.
POLL_SECONDS = 15.0
LEVEL_WINDOW_SECONDS = 2.0
# RMS of 16-bit samples below this counts as silence. Quiet room tone sits in
# the tens; speech is hundreds to thousands. 100 rejects line noise without
# swallowing a softly spoken participant.
SILENCE_RMS_THRESHOLD = 100.0


def _tail_rms(path: Path, bytes_per_second: int, seconds: float) -> float:
    """RMS of the last `seconds` of a WAV being written, or 0 when unreadable.

    A recording in progress has no final header sizes, but the raw sample data
    is appended as it arrives, so the tail reads fine. A channel that vanished
    or is too short counts as silent — the watchdog should not hold a recording
    open because a file cannot be read.
    """
    try:
        size = path.stat().st_size
        count = int(bytes_per_second * seconds)
        if size <= count:
            return 0.0
        with path.open("rb") as fh:
            fh.seek(-count, 2)
            samples = np.frombuffer(fh.read(count), dtype=np.int16)
    except OSError:
        return 0.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class RecordingWatchdog:
    """Watches one recording; sets stop_reason when a trigger fires."""

    def __init__(
        self,
        monitor_path: Path,
        mic_path: Path,
        settings: Settings,
        *,
        started_monotonic: float | None = None,
        poll_seconds: float = POLL_SECONDS,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.stop_reason: str | None = None
        self._monitor_path = monitor_path
        self._mic_path = mic_path
        self._settings = settings
        self._started = time.monotonic() if started_monotonic is None else started_monotonic
        self._poll_seconds = poll_seconds
        self._on_status = on_status or (lambda msg: None)
        self._silent_since: float | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tapeback-watchdog")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Stop the background thread; the recording itself is untouched."""
        self._closed.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=5)

    def check_once(self, now: float) -> str | None:
        """Evaluate both triggers at monotonic time `now`; return the reason, if any.

        Separated from the thread so a test can drive time directly.
        """
        hours = self._settings.auto_stop_hours
        if hours > 0 and now - self._started >= hours * 3600:
            self.stop_reason = f"recording reached the {hours}h auto-stop cap"
            return self.stop_reason

        minutes = self._settings.auto_stop_silence_minutes
        if minutes > 0:
            # 2 bytes/sample, mono, s16le — the recorder's fixed format.
            bps = self._settings.sample_rate * 2
            quiet = all(
                _tail_rms(path, bps, LEVEL_WINDOW_SECONDS) < SILENCE_RMS_THRESHOLD
                for path in (self._monitor_path, self._mic_path)
            )
            if not quiet:
                self._silent_since = None
            elif self._silent_since is None:
                self._silent_since = now
            elif now - self._silent_since >= minutes * 60:
                self.stop_reason = f"silent on both channels for {minutes} min"
                return self.stop_reason
        return None

    def _run(self) -> None:
        while not self._closed.wait(self._poll_seconds):
            if self.check_once(time.monotonic()):
                self._on_status(f"Auto-stop: {self.stop_reason}")
                return
