"""System tray tests — state transitions and menu/SNI updates."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.fixtures import requires_dbus_next

pytestmark = requires_dbus_next

from tapeback.tray import TrayState, _register_with_retry, build_menu_items  # noqa: E402

# --- Menu layout (pure function) ---


def test_build_menu_items_idle_state():
    """In IDLE state, only Start (and shared items) are visible."""
    items = build_menu_items(TrayState.IDLE)
    by_label = {item.label: item for item in items}
    assert by_label["Start Recording"].visible is True
    assert by_label["Stop Recording"].visible is False
    assert by_label["Processing..."].visible is False


def test_build_menu_items_recording_state():
    """In RECORDING state, only Stop is visible."""
    items = build_menu_items(TrayState.RECORDING)
    by_label = {item.label: item for item in items}
    assert by_label["Start Recording"].visible is False
    assert by_label["Stop Recording"].visible is True
    assert by_label["Processing..."].visible is False


def test_build_menu_items_processing_state():
    """In PROCESSING state, only the disabled Processing item is visible."""
    items = build_menu_items(TrayState.PROCESSING)
    by_label = {item.label: item for item in items}
    assert by_label["Start Recording"].visible is False
    assert by_label["Stop Recording"].visible is False
    assert by_label["Processing..."].visible is True
    assert by_label["Processing..."].enabled is False


def test_build_menu_items_always_has_status_quit_separator():
    items = build_menu_items(TrayState.IDLE)
    labels = [item.label for item in items]
    assert "Status" in labels
    assert "Quit" in labels
    assert any(item.type == "separator" for item in items)


def test_build_menu_items_wires_callbacks():
    start = MagicMock()
    stop = MagicMock()
    items = build_menu_items(TrayState.IDLE, on_start=start, on_stop=stop)
    by_label = {item.label: item for item in items}
    assert by_label["Start Recording"].on_clicked is start
    assert by_label["Stop Recording"].on_clicked is stop


# --- Initial state ---


def test_initial_state_idle(tray_app):
    assert tray_app._state == TrayState.IDLE


def test_tooltip_text_idle(tray_app):
    assert tray_app._tooltip_text() == "Idle"


def test_tooltip_text_processing(tray_app):
    tray_app._state = TrayState.PROCESSING
    assert tray_app._tooltip_text() == "Processing..."


def test_tooltip_text_recording_includes_session(tray_app):
    tray_app._state = TrayState.RECORDING
    tray_app._recorder.get_session_info.return_value = {
        "session_name": "test-meeting",
        "started_at": "2026-05-21T10:00:00",
    }
    assert "test-meeting" in tray_app._tooltip_text()


# --- Start recording ---


def test_on_start_spawns_thread_when_idle(tray_app):
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start()
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()
    assert tray_app._state == TrayState.RECORDING


def test_on_start_ignored_when_recording(tray_app):
    tray_app._state = TrayState.RECORDING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start()
    mock_thread.assert_not_called()


def test_on_start_ignored_when_processing(tray_app):
    tray_app._state = TrayState.PROCESSING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start()
    mock_thread.assert_not_called()


def test_do_start_success(tray_app):
    tray_app._state = TrayState.RECORDING
    tray_app._recorder.start.return_value = "2026-05-21_10-00-00"
    # detect_devices probes pactl on the host; mock it so the test is hermetic
    # (CI runners and minimal containers don't ship pulseaudio-utils).
    with patch("tapeback.tray.detect_devices"):
        tray_app._do_start()
    tray_app._recorder.start.assert_called_once_with(tray_app._settings)
    assert tray_app._state == TrayState.RECORDING


def test_do_start_failure_resets_to_idle_via_threadsafe(tray_app):
    """Failed _do_start schedules an IDLE state update on the loop."""
    tray_app._state = TrayState.RECORDING
    tray_app._loop = MagicMock()  # pretend an asyncio loop is running
    tray_app._recorder.start.side_effect = RuntimeError("parecord not found")
    with patch("tapeback.tray.detect_devices"):
        tray_app._do_start()
    tray_app._loop.call_soon_threadsafe.assert_called_once()
    args = tray_app._loop.call_soon_threadsafe.call_args.args
    assert args[1] == TrayState.IDLE


# --- Stop recording ---


def test_on_stop_spawns_thread_when_recording(tray_app):
    tray_app._state = TrayState.RECORDING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_stop()
    mock_thread.assert_called_once()
    assert tray_app._state == TrayState.PROCESSING


def test_on_stop_ignored_when_idle(tray_app):
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_stop()
    mock_thread.assert_not_called()


def test_do_stop_and_process_success_returns_to_idle(tray_app):
    tray_app._state = TrayState.PROCESSING
    tray_app._loop = MagicMock()
    with patch("tapeback.tray.stop_and_process", return_value=Path("/vault/meetings/x.md")):
        tray_app._do_stop_and_process()
    tray_app._loop.call_soon_threadsafe.assert_called_once()
    args = tray_app._loop.call_soon_threadsafe.call_args.args
    assert args[1] == TrayState.IDLE


def test_do_stop_and_process_failure_returns_to_idle(tray_app):
    tray_app._state = TrayState.PROCESSING
    tray_app._loop = MagicMock()
    with patch("tapeback.tray.stop_and_process", side_effect=RuntimeError("transcription failed")):
        tray_app._do_stop_and_process()
    tray_app._loop.call_soon_threadsafe.assert_called_once()
    args = tray_app._loop.call_soon_threadsafe.call_args.args
    assert args[1] == TrayState.IDLE


# --- Quit ---


def test_quit_during_recording_stops_recorder(tray_app, caplog):
    tray_app._state = TrayState.RECORDING
    tray_app._loop = MagicMock()
    session = Path("/home/u/.local/state/tapeback/sessions/meeting")
    tray_app._recorder.stop.return_value = (session / "monitor.wav", session / "mic.wav")
    import asyncio  # noqa: PLC0415

    tray_app._stopped = asyncio.Event()
    with caplog.at_level("INFO", logger="tapeback.tray"):
        tray_app._on_quit()
    tray_app._recorder.stop.assert_called_once()
    tray_app._loop.call_soon_threadsafe.assert_called_once()
    # The session that was kept, not just the root it lives under.
    assert (
        "Recording stopped on quit, files preserved in "
        "/home/u/.local/state/tapeback/sessions/meeting/" in caplog.text
    )


def test_quit_while_idle_does_not_stop_recorder(tray_app):
    tray_app._loop = MagicMock()
    import asyncio  # noqa: PLC0415

    tray_app._stopped = asyncio.Event()
    tray_app._on_quit()
    tray_app._recorder.stop.assert_not_called()


def test_quit_during_processing_does_not_stop_recorder(tray_app):
    tray_app._state = TrayState.PROCESSING
    tray_app._loop = MagicMock()
    import asyncio  # noqa: PLC0415

    tray_app._stopped = asyncio.Event()
    tray_app._on_quit()
    tray_app._recorder.stop.assert_not_called()


# --- _update_state pushes to SNI + Menu ---


def test_update_state_sets_icon_tooltip_and_refreshes_menu(tray_app):
    tray_app._update_state(TrayState.RECORDING)
    name, pixmaps = tray_app._sni.set_icon.call_args.args
    assert name == "tapeback-recording"
    assert len(pixmaps) == 1
    assert pixmaps[0][0] == 22 and pixmaps[0][1] == 22
    tray_app._sni.set_tooltip.assert_called_once()
    tray_app._menu.replace_items.assert_called_once()


# --- watcher registration retry (login race) ---


def test_register_with_retry_succeeds_once_watcher_appears(monkeypatch):
    """Autostart fires before the watcher claims its name; retries must win."""
    import asyncio  # noqa: PLC0415

    calls: list[str] = []

    async def late_watcher(bus, name):
        calls.append(name)
        if len(calls) < 3:
            raise RuntimeError("ServiceUnknown")

    monkeypatch.setattr("tapeback.tray.register_with_watcher", late_watcher)

    assert asyncio.run(
        _register_with_retry(
            MagicMock(), ":1.9/StatusNotifierItem", asyncio.Event(), delay=0
        )
    )
    assert calls == [":1.9/StatusNotifierItem"] * 3


def test_register_with_retry_gives_up_when_watcher_never_appears(monkeypatch):
    """No watcher at all (GNOME without AppIndicator) — stop retrying, stay quit-able."""
    import asyncio  # noqa: PLC0415

    async def never(bus, name):
        raise RuntimeError("ServiceUnknown")

    monkeypatch.setattr("tapeback.tray.register_with_watcher", never)

    assert not asyncio.run(
        _register_with_retry(
            MagicMock(), ":1.9/x", asyncio.Event(), attempts=2, delay=0
        )
    )


def test_register_with_retry_stops_when_tray_quits(monkeypatch):
    """Quit during the wait between attempts ends the loop instead of hanging."""
    import asyncio  # noqa: PLC0415

    async def never(bus, name):
        raise RuntimeError("ServiceUnknown")

    monkeypatch.setattr("tapeback.tray.register_with_watcher", never)
    stopped = asyncio.Event()
    stopped.set()

    assert not asyncio.run(
        _register_with_retry(MagicMock(), ":1.9/x", stopped, delay=60)
    )
