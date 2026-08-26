import pytest
from pydantic import ValidationError

from config.config import Config


def test_invalid_device_rejected(tmp_path):
    with pytest.raises(ValidationError):
        Config(db_path="x.db", collection="c", data_dir=tmp_path, device="tpu")


def test_missing_data_dir_rejected(tmp_path):
    with pytest.raises(ValidationError):
        Config(db_path="x.db", collection="c", data_dir=tmp_path / "does-not-exist")


def test_env_var_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    monkeypatch.setenv("RAG_COLLECTION", "from_env")
    cfg = Config(db_path="x.db", data_dir=tmp_path)
    assert cfg.device == "cpu"
    assert cfg.collection == "from_env"


def test_explicit_arg_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_COLLECTION", "from_env")
    cfg = Config(db_path="x.db", collection="explicit", data_dir=tmp_path)
    assert cfg.collection == "explicit"


def test_config_is_frozen(tmp_path):
    cfg = Config(db_path="x.db", collection="c", data_dir=tmp_path, device="cpu")
    with pytest.raises(ValidationError):
        cfg.device = "cuda"
