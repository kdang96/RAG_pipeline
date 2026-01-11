"""
This is a demo of an agentic search over MS Word data stored in a Milvus vector database.
It is performed using the GPT-OSS:20b model, and the e5-large-v2 embedding model.
The LLM decides ("tool_choice": "auto") whether to answer with or without using a tool.
The avilable tool allows for a similarity search of the vector database.
The results are then summarised in natural language by the same LLM.
"""


from typing import Tuple
import typer
from pathlib import Path
import logging
import sys
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from langchain_ollama.chat_models import ChatOllama
import ollama
from sentence_transformers import SentenceTransformer
from src.utils.general_util import format_entities_for_llm
from src.vector_store.milvus import (
    get_collection_fields,
    search,
    Config
)


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
                "type": "object",
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


llm = ChatOllama(model="gpt-oss:20b", temperature=1, num_ctx=131072, keep_alive=0, num_gpu=24)

embedding_model = SentenceTransformer(
    model_name_or_path="intfloat/e5-large-v2", device="cuda", trust_remote_code=True
)



# ReAct loop for agentic search
def react_agent(question: str, config: Config, max_iters=3):
    fields_n_descriptions = get_collection_fields(config.db_path, config.collection)
    print(fields_n_descriptions)

    prompt = BASE_PROMPT.format(user_query=question, fields=fields_n_descriptions)


    response = ollama.chat(
                    model="gpt-oss:20b",
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOLS,
                    options = {
                        "tool_choice": "auto",
                        "temperature": 1
                        }
                )
    reasoning = response.message.thinking
    print("Model reasoning:\n", reasoning)

    # Look for an Action
    obs, obs_formatted = tools(response, config)

    print(f"User query:{question}\n Available information:" + obs_formatted)

    messages = [
        {"role": "system", 
         "content": """ An agentic tool has been used to extract the below data in response to the user query. Output it in natural language.
                        When returning results clearly define what information originates from the retrieved document and what out of you weights.
                        Showing data provenance is very important."""},
        {"role": "user", 
         "content": f"""User query:{question}\nAvailable information:""" + obs_formatted }
    ]


    final_response = ollama.chat(
                model="gpt-oss:20b",
                messages=messages,
                options = {
                    "tool_choice": "none",
                    "temperature": 1
                    }
            )

    return final_response.message.content, reasoning 


# Here is defined function logic that the agent can use
def tools(llm_response, config: Config) -> Tuple[list[dict], str]:
    action = llm_response.message.tool_calls[0].function.name
    args = llm_response.message.tool_calls[0].function.arguments
    action_input = args["input"]

    print("Function:", action)
    print("Tools args:\n", args)

    if action.lower() == "similarity_search":
        output_fields = args["output_fields"]

        if "scope" not in output_fields:
            output_fields.append("scope")

        obs = search(
            config=config,
            user_queries=[action_input],
            search_col="combined_vector",
            output_fields = ["doc_title", "chunk_id", "heading_2", "heading_3", "heading_4", "chunk"],
            search_radius = 0.815,
            k_limit = 10
        )
        
        if obs:
            obs = [x[0]["entity"] for x in obs]
            obs_formatted = format_entities_for_llm(obs)
        else:
            obs = "No information was retrieved."
    else:
        obs = f"Unknown action: {action}"
        obs_formatted = obs

    return obs, obs_formatted


def chatbot_fn(message, history, cfg):
    answer, reasoning = react_agent(message, cfg)
    history.append((message, answer))
    return history, history, reasoning

def main(
    data_dir: Path = typer.Option("src/demo_docx/chunks.jsonl", help="Root directory containing the JSONL files"),
    db_path: str = typer.Option("data/output/rag_demo.db", help="Target Milvus database file"),
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

        chat_state = gr.State([])            # holds chat history
        config_state = gr.State(value=config)  # hold the Config so it can be passed to rag()

        msg.submit(chatbot_fn, [msg, chat_state, config_state], [chatbot, chat_state, reasoning_box])
        clear.click(lambda: ([], [], ""), None, [chatbot, chat_state, reasoning_box])

    demo.launch()


if __name__ == "__main__":
    typer.run(main)