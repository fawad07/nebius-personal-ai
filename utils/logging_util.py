import contextvars
import logging
import logging.config
import uuid

import yaml

# Per-session / per-turn correlation IDs, propagated automatically to any
# asyncio task spawned (via asyncio.create_task) from a context where they've
# been set, since asyncio copies the current contextvars.Context on task
# creation. This lets every log line emitted anywhere in the pipeline during
# a given turn -- across the concurrent speaker-ID/SER/speak/barge-in tasks
# -- be tied back to the session and turn that produced it.
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="-")


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def new_turn_id() -> str:
    return uuid.uuid4().hex[:8]


class CorrelationIdFilter(logging.Filter):
    """Injects the active session_id/turn_id (from contextvars) into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        record.turn_id = turn_id_var.get()
        return True


def setup_logging(config_path: str = "config/logging.yaml"):
    """
    Initialize logging using YAML configuration, and attach a correlation-id
    filter to every configured handler so log records carry session_id/turn_id.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        logging.warning("Failed to load logging config. Using basic logging.")

    correlation_filter = CorrelationIdFilter()
    seen_handler_ids = set()
    all_loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]
    for logger_obj in all_loggers:
        for handler in getattr(logger_obj, "handlers", []):
            if id(handler) not in seen_handler_ids:
                handler.addFilter(correlation_filter)
                seen_handler_ids.add(id(handler))
