"""Shared test helpers.

Temp directories are created inside the workspace (`var/testtmp`) rather than
the OS temp dir: on Windows the sandbox may deny writes to `%TEMP%`, which
would turn every test into a permission error instead of a real assertion.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_tmp() -> Path:
    path = ROOT / "var" / "testtmp" / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_tmp(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
