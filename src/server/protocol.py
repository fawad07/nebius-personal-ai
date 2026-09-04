"""Tiny JSON message protocol shared by the voice server and browser client."""

import json


def encode(msg_type: str, **fields) -> str:
    """Serialize a control message to JSON text (audio travels as binary)."""
    return json.dumps({"type": msg_type, **fields})


def decode(raw: str) -> dict:
    """Parse an inbound JSON control message."""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or "type" not in obj:
        raise ValueError("message must be a JSON object with a 'type'")
    return obj
