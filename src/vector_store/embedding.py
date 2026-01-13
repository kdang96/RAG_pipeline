    
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config import Config

logger = logging.getLogger(__name__)
    

def embed_texts(
    texts: list[str],
    config: Config
) -> np.ndarray:
    model = SentenceTransformer(
        model_name_or_path=config.model_name,
        device=config.device,
        trust_remote_code=config.trust_remote_code,
    )
    return model.encode(texts, convert_to_numpy=True)