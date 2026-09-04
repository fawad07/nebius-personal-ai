"""
automation/reports.py

Activity reports: summarise what the assistant has done over a recent
window (tool usage, how busy, when) into readable markdown the assistant
can show or save.

The data comes from the events already stored in the encrypted memory DB.
That store isn't reachable from the stateless automation layer, so — like
files.WORKSPACE_ROOT and system._GATHERERS — the event source is injected
at startup (run_assistant.build_agent sets _EVENT_SOURCE to
memory.get_events_since). The report generation itself is pure: it takes a
list of event dicts and returns a string, so it's easy to test without a
database.
"""

from __future__ import annotations

import time
from collections import Counter

from core.behavior import bucket_for   # reuse the part-of-day buckets

# Injected at startup: a callable(since_timestamp) -> list[event dict].
# None means "no data source wired" — execute() then says so rather than
# guessing.
_EVENT_SOURCE = None

MAX_DAYS = 365   # clamp so an absurd request can't scan an unbounded window


def generate_report(events: list, period_days: int, now: float | None = None) -> str:
    """
    Pure: build a markdown activity summary from event dicts. Each event is
    {"timestamp": float, "tool": str | None, ...}; tool-less events
    (plain conversation) are counted as chatter, not actions.
    """
    now = now if now is not None else time.time()
    actions = [e for e in events if e.get("tool")]

    lines = [f"# Activity report — last {period_days} day(s)", ""]
    if not actions:
        lines.append("No tool activity in this period.")
        return "\n".join(lines)

    tools = Counter(e["tool"] for e in actions)
    days = {time.strftime("%Y-%m-%d", time.localtime(e["timestamp"])) for e in actions}
    buckets = Counter(bucket_for(e["timestamp"]) for e in actions)
    busiest = buckets.most_common(1)[0][0]

    lines += [
        f"- Total actions: {len(actions)}",
        f"- Active days: {len(days)}",
        f"- Busiest part of day: {busiest}",
        "",
        "## Tools used",
    ]
    for tool, count in tools.most_common():
        lines.append(f"- {tool}: {count}")

    lines += ["", "## Most recent"]
    for e in sorted(actions, key=lambda e: e["timestamp"], reverse=True)[:5]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["timestamp"]))
        lines.append(f"- {when}  {e['tool']}")

    return "\n".join(lines)


def execute(action: str, args: dict, grant=None):
    if action == "activity_report":
        if _EVENT_SOURCE is None:
            return "Activity data is unavailable (no event source configured)."
        try:
            days = int(args.get("days", 7))
        except (TypeError, ValueError):
            days = 7
        days = max(1, min(MAX_DAYS, days))
        since = time.time() - days * 86400
        events = _EVENT_SOURCE(since)
        return generate_report(events, days)

    raise NotImplementedError(f"automation.reports: '{action}' not yet implemented")
