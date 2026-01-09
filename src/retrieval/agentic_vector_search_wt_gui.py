import json
from typing import Tuple
import typer
from pathlib import Path
import logging
import sys


import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from langchain_ollama.chat_models import ChatOllama
from openai import OpenAI
from test_pipeline.pipelines.evaluation.rag_eval_metrics import calc_faithfulness, calc_relevancy_score
from ragas import SingleTurnSample
from sentence_transformers import SentenceTransformer
from src.vector_store.milvus import (
    get_collection_fields,
    query,
    search,
    Config
)


"""
This is a demo of an agentic search over MS Word data stored in a Milvus vectord database.
It is performed using GPT-OSS and Mistral 24B/Qwen3:8b models.
The agent uses tools to seearch/query the database and perform value counts, then summarises the
results using a separate LLM.
"""

# --------------------------------------------------------------------------- #
# Prompt and Funcitons
# --------------------------------------------------------------------------- #

# Tools available to the agent:
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "similarity_search",
            "description": "Search the Milvus database using vector similarity",
            "parameters": {
                "type": "string",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "The vector field in the Milvus collection to search in",
                    },
                    "input": {
                        "type": "string",
                        "description": "This can either be the user query or anything semantically similar.",
                    },
                    "output_fields": {
                        "type": "array",
                        "description": "List of fields to include in the output of the search",
                    },
                },
                "required": ["field", "input", "output_fields"],
            },
        },
    },
]

# Base Prompt:
BASE_PROMPT = """You are an intelligent assistant that can answer questions using ReAct framework.
Do not hallucinate! You won't yet have the answers until you use the tools and incorporate the results.

Field: database field to search in, one of {fields}\n
For fields that have a vector equivalent, use the Search function on the vector field.

User: {user_query}\n
"""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

def format_entities_for_llm(entities: list[dict]) -> str:
    """
    Convert a dictionary of entities into a clean,
    descriptive, LLM-friendly text block.
    """
    lines = []

    for entity in entities:
        for key, value in entity.items():
            if type(value) is not float:
                lines.append(f"  - {key}: {value}")
            else:
                lines.append(f"  - {key}: {value:.5f}")
        lines.append("")  # blank line between entities

    return "\n".join(lines)

def visualise_metrics(eval_metrics) -> Figure:
    # Create bar chart with matplotlib
    fig, ax = plt.subplots(figsize=(4, 3))
    bar = ax.bar(
        eval_metrics.keys(),
        eval_metrics.values(),
        color=["#4CAF50", "#2196F3", "#FFC107"],
    )
    ax.set_ylim([0, 1])
    ax.bar_label(bar)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")

    return fig



# Model definitions
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API
    api_key="ollama",  # Dummy key
)

# agentic_llm = ChatOllama(model="gpt-oss", temperature=1, num_ctx=131072, keep_alive=0, num_gpu=24)
summarising_llm = ChatOllama(
    model="mistral-small3.2:24b-instruct-2506-q4_K_M",
    temperature=0,
    num_ctx=132768,
    keep_alive=0,
)

# summarising_llm = ChatOllama(model="qwen3:4b", temperature=0, num_ctx=132768, keep_alive=0, num_gpu=40)
embedding_model = SentenceTransformer(
    model_name_or_path="intfloat/e5-large-v2", device="cuda", trust_remote_code=True
)



# ReAct loop for agentic search
def react_agent(question: str, config: Config, max_iters=3):
    fields_n_descriptions = get_collection_fields(config.db_path, config.collection)
    print(fields_n_descriptions)

    prompt = BASE_PROMPT.format(user_query=question, fields=fields_n_descriptions)
    history = prompt

    # this works with OpenAI library, Ollama integration
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": history}],
        tools=TOOLS,
        tool_choice="auto",
        temperature=1,
    )

    reasoning = response.choices[0].message.model_extra["reasoning"]

    print("Model reasoning:\n", reasoning)
    print("Function:", response.choices[0].message.tool_calls[0].function.name)
    print("Tools args:\n", response.choices[0].message.tool_calls[0].function.arguments)

    # response = gpt_oss_general_query(history, temperature=0)
    # text = response.strip()

    # Look for an Action
    obs, llm_table = tools(response, config)

    print(f"User query:{question}\n Available information:" + llm_table)
    result = summarising_llm.invoke(
        f"""User query:{question}\n An agentic tool has been used to extract the below data in response to the user query. Output it in natural language.
                                         Returned results should include but not be limited to the relevant project titles and work order(WO) numbers (e.g, U-WO000XXX) when outputting specific project info.
                                         Available information:"""
        + llm_table
    )

    eval_metrics = evaluate(
        user_query=question, observations=obs, llm_response=result.content
    )

    try:
        # for Qwen model
        result = result.content.split("</think>\n\n", 1)[1]
    except Exception:
        result = result.content

    return result, reasoning, eval_metrics


# Here is defined function logic that the agent can use
def tools(llm_response, config: Config) -> Tuple[list[dict], str]:
    action = llm_response.choices[0].message.tool_calls[0].function.name
    args = json.loads(llm_response.choices[0].message.tool_calls[0].function.arguments)
    field = args["field"]
    action_input = args["input"]

    if action.lower() == "similarity_search":
        output_fields = args["output_fields"]

        if "scope" not in output_fields:
            output_fields.append("scope")

        obs = search(
            config=config,
            user_query=action_input,
            search_col="chunk_vector",
            output_fields = ["doc_title","chunk"],
            search_radius = 0.78,
            k_limit = 5
        )

        obs = [x for x in obs]
        llm_table = format_entities_for_llm(obs)
    else:
        obs = f"Unknown action: {action}"
        llm_table = str(obs)

    return obs, llm_table



def evaluate(user_query: str, observations: list[str], llm_response: str):

    return 0


def chatbot_fn(message, history, cfg):
    answer, reasoning, eval_metrics = react_agent(message, cfg)
    history.append((message, answer))
    fig = visualise_metrics(eval_metrics)  # Generate updated metrics plot
    return history, history, reasoning, fig

def main(
    data_dir: Path = typer.Option("src/demo_docx/chunks.jsonl", help="Root directory containing the JSONL files"),
    db_path: str = typer.Option("rag_demo.db", help="Target Milvus database file"),
    collection: str = typer.Option("demo_collection", help="Milvus collection name"),
    model_name: str = typer.Option(
        "intfloat/e5-large-v2", help="SentenceTransformer model name or path"
    ),
    device: str = typer.Option("cuda", help="Device to run the embedding model on"),
    trust_remote_code: bool = typer.Option(True, help="Whether to trust remote code when loading the model")
) -> None:
    config = Config(
        data_dir=data_dir,
        db_path=db_path,
        collection=collection,
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    
    logger.info("Running with config: %s", config)

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
                metrics_plot = gr.Plot(label="Evaluation Metrics")

        chat_state = gr.State([])            # holds chat history
        config_state = gr.State(value=config)  # hold the Config so it can be passed to rag()

        msg.submit(chatbot_fn, [msg, chat_state, config_state], [chatbot, chat_state, reasoning_box, metrics_plot])
        clear.click(lambda: ([], [], "", visualise_metrics({})), None, [chatbot, chat_state, reasoning_box, metrics_plot])

    demo.launch()


if __name__ == "__main__":
    typer.run(main)