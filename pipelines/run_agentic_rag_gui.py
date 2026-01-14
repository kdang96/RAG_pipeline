"""
This is a demo of an agentic search over MS Word data stored in a Milvus vector database.
It is performed using the GPT-OSS:20b model, and the e5-large-v2 embedding model.
The LLM decides ("tool_choice": "auto") whether to answer with or without using a tool.
The avilable tool allows for a similarity search of the vector database.
The results are then summarised in natural language by the same LLM.
"""


import logging
import typer
from pathlib import Path
import gradio as gr
from config.logging_config import setup_logging
from retrieval.agentic_rag import rag_flow
from config.config import Config


def chatbot_fn(message, history, cfg):
    answer, reasoning = rag_flow(message, cfg)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, history, reasoning


def app(config: Config):
        # Demo Gradio interface created inside main so we can capture `config` and pass it to the rag() function
    with gr.Blocks() as demo:
        gr.Markdown("## Demo Agentic RAG ChatBot")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Ask me something")
                clear = gr.Button("Clear Chat")
            with gr.Column(scale=2):
                reasoning_box = gr.Textbox(label="Model Reasoning Trace", lines=15)

        chat_state = gr.State([])            # holds chat history
        config_state = gr.State(value=config)  # hold the Config so it can be passed to rag()

        msg.submit(chatbot_fn, [msg, chat_state, config_state], [chatbot, chat_state, reasoning_box])
        clear.click(lambda: ([], [], ""), None, [chatbot, chat_state, reasoning_box])

    demo.launch()


def main(
    data_dir: Path = typer.Option("data/processed/chunks.jsonl", help="Root directory containing the JSONL files"),
    db_path: str = typer.Option("data/output/rag_demo.db", help="Target Milvus database file"),
    collection: str = typer.Option("demo_collection", help="Milvus collection name"),
    model_name: str = typer.Option(
        "intfloat/e5-large-v2", help="SentenceTransformer model name or path"
    ),
    device: str = typer.Option("cuda", help="Device to run the embedding model on"),
    trust_remote_code: bool = typer.Option(True, help="Whether to trust remote code when loading the model")
) -> None:
    logger = logging.getLogger(__name__)
    config = Config(
        data_dir=data_dir,
        db_path=db_path,
        collection=collection,
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    
    logger.info("Running with config: %s", config)

    app(config)


if __name__ == "__main__":
    setup_logging()
    typer.run(main)