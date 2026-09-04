"""
core/behavior.py

Behavioral learning: notice which tools the admin tends to use, and when,
and offer them as suggestions — a lightweight trial-and-error loop rather
than deep learning.

The model:
  - counts (tool, time-of-day bucket) occurrences from past events, so it
    can answer "around this time, what does the admin usually do?"
  - treats each suggestion as a multi-armed-bandit arm: accepting one
    upweights it, dismissing one downweights it, so useless suggestions
    fade and useful ones surface.

Honest scope: with a small, mostly on-demand tool set, file actions are
only weakly time-patterned, so this is more a demonstration of the
mechanism (and the roadmap's Phase 4 slot) than a high-value daily
feature. It is deliberately simple — frequency counts and a reward
multiplier, no library.

Privacy: only the admin's own events are ever learned from. A shared-user
session would need the same notice/consent flow as security_monitor
(Addendum Section 5) before its actions could feed this.

This module is pure: it takes a list of event dicts and returns
suggestions, with no storage or LLM coupling, so it is easy to test.
State (the counts + reward weights) round-trips via to_dict/from_dict for
the caller to persist.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

# Coarse time-of-day buckets: fine enough to separate "morning admin work"
# from "evening", without pretending to minute-level precision we don't have.
BUCKETS = ("night", "morning", "afternoon", "evening")


def bucket_for(ts: float) -> str:
    """Map a timestamp to a coarse part-of-day bucket (local time)."""
    hour = time.localtime(ts).tm_hour
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


@dataclass
class Suggestion:
    tool: str
    bucket: str
    score: float        # frequency x learned reward weight

    def __str__(self) -> str:
        return f"{self.tool} (usually done in the {self.bucket})"


@dataclass
class BehaviorModel:
    """
    Learns tool-by-time-of-day habits and suggests likely next actions.

    min_events: below this many observed events, suggest nothing — a few
    data points are not a habit, and premature suggestions annoy.
    """
    min_events: int = 20

    # counts[bucket][tool] -> how many times that tool was used in that bucket
    counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    # reward[tool] -> bandit multiplier, nudged by accept/dismiss feedback
    reward: dict = field(default_factory=lambda: defaultdict(lambda: 1.0))
    total_events: int = 0

    # -- learning -----------------------------------------------------------

    def update(self, event: dict) -> None:
        """Record one past action. event: {"tool": str, "timestamp": float}."""
        tool = event.get("tool")
        if not tool:
            return   # conversational turns / tool-less events carry no habit
        b = bucket_for(event.get("timestamp", time.time()))
        self.counts[b][tool] += 1
        self.total_events += 1

    def observe_feedback(self, tool: str, accepted: bool) -> None:
        """
        Bandit reward: a suggestion the admin accepted becomes more likely
        to be offered again; a dismissed one, less. Bounded so a tool can
        neither be silenced forever nor dominate.
        """
        factor = 1.25 if accepted else 0.8
        self.reward[tool] = max(0.2, min(3.0, self.reward[tool] * factor))

    # -- suggesting ---------------------------------------------------------

    def suggest_tasks(self, context: dict, k: int = 3) -> list:
        """
        Up to k likely actions for the current time of day, best first.
        Empty until min_events observations exist (cold start).
        """
        if self.total_events < self.min_events:
            return []
        b = context.get("bucket") or bucket_for(context.get("timestamp", time.time()))
        here = self.counts.get(b, {})
        scored = [Suggestion(tool, b, count * self.reward[tool])
                  for tool, count in here.items() if count > 0]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "min_events": self.min_events,
            "total_events": self.total_events,
            "counts": {b: dict(tools) for b, tools in self.counts.items()},
            "reward": dict(self.reward),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorModel":
        m = cls(min_events=d.get("min_events", 20))
        m.total_events = d.get("total_events", 0)
        for b, tools in d.get("counts", {}).items():
            for tool, n in tools.items():
                m.counts[b][tool] = n
        for tool, w in d.get("reward", {}).items():
            m.reward[tool] = w
        return m
