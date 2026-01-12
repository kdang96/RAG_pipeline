from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable configuration container."""

    db_path: str
    collection: str
    data_dir: Path = Path("")
    model_name: str = "intfloat/e5-large-v2"
    device: str = "cuda"
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                f"Invalid device '{self.device}'. Expected 'cpu' or 'cuda'."
            )

        if not self.data_dir.exists():
            raise ValueError(f"data_dir does not exist: {self.data_dir}")
