from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Pipeline configuration.

    Every field can be overridden via an ``RAG_``-prefixed environment
    variable (e.g. ``RAG_DEVICE=cpu``) or a ``.env`` file, in addition to
    being passed explicitly.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", frozen=True)

    db_path: str = "data/output/rag_demo.db"
    collection: str = "demo_collection"
    data_dir: Path = Path("")
    model_name: str = "intfloat/e5-large-v2"
    device: str = "cuda"
    trust_remote_code: bool = True

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if value not in {"cpu", "cuda"}:
            raise ValueError(f"Invalid device '{value}'. Expected 'cpu' or 'cuda'.")
        return value

    @field_validator("data_dir")
    @classmethod
    def _validate_data_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"data_dir does not exist: {value}")
        return value
