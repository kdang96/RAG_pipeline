    
import logging
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config import Config

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     stream=sys.stdout,
# )



# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Embedder:
    """Lazy‑loaded SentenceTransformer wrapper."""

    _model: SentenceTransformer | None = None

    @classmethod
    def model(cls, config: Config) -> SentenceTransformer:
        if cls._model is None:
            logger.info("Loading SentenceTransformer (%s) …", config.model_name)
            cls._model = SentenceTransformer(
                model_name_or_path=config.model_name,
                device=config.device,
                trust_remote_code=config.trust_remote_code,
            )
        return cls._model

    @classmethod
    def embed(cls, texts: list[str], config: Config) -> list[np.ndarray]:
        model = cls.model(config)
        return model.encode(list(texts), convert_to_numpy=True)