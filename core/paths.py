"""
core/paths.py — anchors all data/config/model paths to the project root.

These used to be CWD-relative, which meant launching the assistant from
any other directory quietly pointed every lookup at the wrong place —
and because auth fails closed, a missing embedding file became an admin
lockout rather than an obvious error.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
