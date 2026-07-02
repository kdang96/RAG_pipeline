import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str, trust_remote_code: bool) -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model '%s' on %s", model_name, device)
    return SentenceTransformer(
        model_name_or_path=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def embed_texts(
    texts: list[str],
    config: Config
) -> np.ndarray:
    model = _load_model(config.model_name, config.device, config.trust_remote_code)
    return model.encode(texts, convert_to_numpy=True)
