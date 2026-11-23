"""Sandbox helpers for untrusted code execution.

The agent's own calculator uses an AST whitelist, not this module. This is for
generated/foreign code (e.g. the codegen project or future code tools):

* prefer `docker_command()` — no network, read-only root, dropped caps, pids and
  memory limits, non-root user;
* `run_python()` is a *degraded* local fallback: no network isolation, and
  resource limits only on POSIX. It exists so tests and offline demos can run;
  do not treat it as a security boundary on Windows.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SandboxSpec:
    image: str = "python:3.11-slim"
    network: bool = False
    read_only: bool = True
    memory: str = "256m"
    pids: int = 128
    timeout_s: float = 10.0


def docker_command(
    spec: SandboxSpec,
    command: Sequence[str],
    *,
    workdir: Path | str | None = None,
) -> list[str]:
    """Build a hardened `docker run` command (not executed by this helper)."""
    args = ["docker", "run", "--rm", "--cap-drop", "ALL", "--pids-limit", str(spec.pids)]
    args += ["--memory", spec.memory, "--user", "1000:1000"]
    args += ["--network", "bridge" if spec.network else "none"]
    if spec.read_only:
        args += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"]
    if workdir is not None:
        args += ["-v", f"{Path(workdir).resolve()}:/work:ro", "-w", "/work"]
    args += [spec.image, *command]
    return args


def run_python(code: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    """Run Python in an isolated-mode subprocess. Degraded on Windows.

    Uses files instead of pipes for stdout/stderr so it also works under the
    DSH Windows sandbox, where child-process named pipes are denied.
    """
    with tempfile.TemporaryDirectory(prefix="de-sandbox-") as td:
        out_path = Path(td) / "stdout.txt"
        err_path = Path(td) / "stderr.txt"
        try:
            with out_path.open("w", encoding="utf-8") as out, err_path.open(
                "w", encoding="utf-8"
            ) as err:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", code],
                    stdout=out,
                    stderr=err,
                    stdin=subprocess.DEVNULL,
                    timeout=timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "stdout": "", "stderr": ""}
        return {
            "ok": completed.returncode == 0,
            "stdout": out_path.read_text(encoding="utf-8", errors="replace")[-2000:],
            "stderr": err_path.read_text(encoding="utf-8", errors="replace")[-2000:],
            "returncode": completed.returncode,
        }
