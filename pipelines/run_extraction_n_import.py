from pathlib import Path
from src.ingest.docx_extract_n_chunk import get_all_chunks
from src.data_import.milvus_import_data import process_pipeline
from src.config.config import Config
from src.config.logging_config import setup_logging



def run_pipline(docx_dir: str):
    # --------------------------------------------------------------------------- #
    # Convert DOCX to chunks
    # --------------------------------------------------------------------------- #
    chunks = get_all_chunks(docx_dir)

    # --------------------------------------------------------------------------- #
    # Import in Milvus vector database
    # --------------------------------------------------------------------------- #
    process_pipeline(
        Config(
            data_dir=Path("data/processed/chunks.jsonl"),
            db_path="data/output/rag_demo.db",
            collection="demo_collection",
            model_name="intfloat/e5-large-v2",
            device="cuda",
            trust_remote_code=True,
        ),
        chunks
        )



if __name__ == "__main__":
    setup_logging()
    run_pipline(r"data/input")