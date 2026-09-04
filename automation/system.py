"""
automation/system.py

Read-only system monitoring: the assistant can report on the machine's
state (platform, CPU, memory, disk, battery, uptime) but cannot change
anything here. Deliberately no destructive control — this is "tell me how
the laptop is doing", not "reconfigure it".

No third-party dependency: uses the standard library plus a couple of
macOS command-line tools (sysctl, pmset), each gathered independently so a
missing tool or a non-mac platform degrades to "unavailable" for that one
metric rather than failing the whole report.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import time

logger = logging.getLogger("automation.system")


def _run(cmd: list) -> "str | None":
    """Run a short command, returning stdout stripped, or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _platform() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _cpu() -> str:
    cores = os.cpu_count() or "?"
    try:
        load1, load5, load15 = os.getloadavg()   # Unix/macOS
        return f"{cores} cores, load {load1:.2f} / {load5:.2f} / {load15:.2f} (1/5/15 min)"
    except (OSError, AttributeError):
        return f"{cores} cores"


def _memory() -> "str | None":
    total = _run(["sysctl", "-n", "hw.memsize"])
    if not (total and total.isdigit()):
        return None
    return f"{int(total) / 1e9:.1f} GB total"


def _disk() -> str:
    du = shutil.disk_usage("/")
    used_pct = du.used / du.total * 100 if du.total else 0
    return (f"{du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB "
            f"({used_pct:.0f}% used)")


def _battery() -> "str | None":
    out = _run(["pmset", "-g", "batt"])
    if not out:
        return None
    # e.g. "...  95%; discharging; 3:21 remaining present: true"
    m = re.search(r"(\d+)%; ?([a-zA-Z ]+?);", out)
    if not m:
        return None
    pct, state = m.group(1), m.group(2).strip()
    return f"{pct}% ({state})"


def _uptime() -> "str | None":
    out = _run(["sysctl", "-n", "kern.boottime"])
    if not out:
        return None
    m = re.search(r"sec\s*=\s*(\d+)", out)
    if not m:
        return None
    seconds = max(0, int(time.time()) - int(m.group(1)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


# Ordered so the summary reads top-to-bottom sensibly. Each is independent:
# one returning None just omits that line.
_GATHERERS = {
    "Platform": _platform,
    "CPU": _cpu,
    "Memory": _memory,
    "Disk": _disk,
    "Battery": _battery,
    "Uptime": _uptime,
}


def _format_system_info() -> str:
    lines = []
    for label, gather in _GATHERERS.items():
        try:
            value = gather()
        except Exception as e:
            logger.debug("System metric %s failed: %s", label, e)
            value = None
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) if lines else "No system information available."


def execute(action: str, args: dict, grant=None):
    if action == "system_info":
        return _format_system_info()
    raise NotImplementedError(f"automation.system: '{action}' not yet implemented")
