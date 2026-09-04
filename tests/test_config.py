import pytest

from utils.config_loader import load_config
from utils.config_schema import ConfigError, validate_config


def test_real_settings_validates():
    cfg = load_config("config/settings.yaml")
    app = validate_config(cfg)
    assert app.audio.rate == 16000
    assert app.llm.model
    # Newly wired knobs are recognized (not "dead" config).
    assert isinstance(app.speaker.auto_enroll, bool)
    assert app.session.barge_in_speech_chunks >= 1


def test_unknown_section_rejected():
    with pytest.raises(ConfigError):
        validate_config({"not_a_section": {}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        validate_config({"llm": {"modle": "gpt-4o-mini"}})


def test_empty_config_uses_defaults():
    app = validate_config({})
    assert app.stt.model_path == "medium"
    assert app.session.barge_in_speech_chunks == 2
