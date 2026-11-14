"""Tray icon pixmaps — COSMIC's status-area applet does not resolve themed icon
names reliably, so the tray ships real ARGB bitmaps instead of icon names only.

A filled circle per state (grey idle, red recording, yellow processing),
generated with numpy — no image libraries, no asset files.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# [width, height, ARGB32 bytes] — the SNI IconPixmap wire format. A list, not a
# tuple: dbus-next requires structs to be Python lists.
Pixmap = list

ICON_SIZE = 22


def circle_pixmap(rgb: tuple[int, int, int], size: int = ICON_SIZE) -> Pixmap:
    """A filled circle with a transparent margin, as one SNI pixmap entry."""
    y, x = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    radius = size / 2 - 1
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2

    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[mask, 0] = 255  # alpha
    pixels[mask, 1] = rgb[0]  # red
    pixels[mask, 2] = rgb[1]  # green
    pixels[mask, 3] = rgb[2]  # blue
    return [size, size, pixels.tobytes()]


IDLE_PIXMAP = circle_pixmap((150, 150, 150))
RECORDING_PIXMAP = circle_pixmap((230, 60, 60))
PROCESSING_PIXMAP = circle_pixmap((230, 190, 60))


def _png_bytes(width: int, height: int, argb: bytes) -> bytes:
    """Encode raw ARGB pixels as a PNG, stdlib only (zlib + struct)."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    # PNG scanlines are filter-byte + RGBA; our buffer is ARGB, so swap channels.
    arr = np.frombuffer(argb, dtype=np.uint8).reshape(height, width, 4)
    rgba = arr[:, :, [1, 2, 3, 0]]
    raw = b"".join(b"\x00" + row.tobytes() for row in rgba)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


_STATE_RGB = (
    ("tapeback-idle", (150, 150, 150)),
    ("tapeback-recording", (230, 60, 60)),
    ("tapeback-processing", (230, 190, 60)),
)

# Sizes COSMIC panels typically request; hicolor/apps is the fallback context.
_HICOLOR_SIZES = (24, 32, 48)


def write_icon_theme(directory: Path, size: int = 48) -> Path:
    """Write tapeback-{idle,recording,processing}.png into `directory`.

    The names are deliberately tapeback-specific: COSMIC's freedesktop-icons
    lookup does a flat search under IconThemePath, so a real PNG at
    <dir>/<IconName>.png resolves regardless of the system icon theme.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, rgb in _STATE_RGB:
        width, height, argb = circle_pixmap(rgb, size=size)
        (directory / f"{name}.png").write_bytes(_png_bytes(width, height, argb))
    return directory


def install_hicolor_icons(data_home: Path) -> None:
    """Install the state icons into the user's hicolor theme.

    applets that ignore IconThemePath resolve IconName through the standard
    freedesktop lookup: <XDG_DATA_HOME>/icons/hicolor/<N>x<N>/apps/<name>.png.
    """
    for size in _HICOLOR_SIZES:
        apps_dir = data_home / "icons" / "hicolor" / f"{size}x{size}" / "apps"
        apps_dir.mkdir(parents=True, exist_ok=True)
        for name, rgb in _STATE_RGB:
            width, height, argb = circle_pixmap(rgb, size=size)
            (apps_dir / f"{name}.png").write_bytes(_png_bytes(width, height, argb))
