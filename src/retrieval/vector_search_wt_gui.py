import logging
from pathlib import Path
import sys
import typer
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
import gradio as gr
from openai import OpenAI
from src.evaluation.rag_eval_metrics import calc_relevancy_score
from src.vector_store.milvus import Config, search
from ragas import SingleTurnSample
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer
from pprint import pprint


"""
This is a demo of an agentic search over Word data stored in a Milvus vectord database.
It is performed using Mistral Small 3.2 models.
The agent uses tools to seearch/query the database and perform value counts, then summarises the
results using a separate LLM.
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

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API
    api_key="ollama"                       # Dummy key
)

# Model definitions
summarising_llm = ChatOllama(model="gpt-oss:20b", temperature=1, num_ctx=131072, keep_alive=0, num_gpu=24)
embedding_model = SentenceTransformer(
    model_name_or_path = "intfloat/e5-large-v2",
    device = "cuda",
    trust_remote_code = True
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


# ReAct loop for agentic search
def rag(question: str, config: Config, max_iters=3):
    # Round #1 of RAG search - document level
    obs1 = search(
        config=config,
        user_query=question,
        search_col="chunk_vector",
        output_fields = ["doc_title","chunk"],
        search_radius = 0.78,
        k_limit = 5
    )
    print("Observation 1:")
    pprint(obs1)

    obs_content = [e["entity"] for e in obs1]

    if obs1:
        entities = format_entities_for_llm(obs_content)

        result = summarising_llm.invoke(f"""Given the RAG results: {entities}
                                        provide a detailed response to the original user prompt.{question}
                                        Make sure to mention any relevant document names and numbers in your answer.
                                        Only use the provided RAG results to formulate your answer. Do not make up any facts!""")
        print(f"Final result:\n {result.text}")

        eval_metrics = evaluate(
                            user_query=question,
                            observations=obs1, 
                            llm_response=result.content
                            )
        

        result = result.content
        reasoning = ""

        return result, entities, eval_metrics


    else:
        result = f"""The RAG search returned no relevant documents for the user query: {question}
                                        Please try another search query."""
        print(result)
    
        return result, "", {}


def evaluate(
        user_query:str, 
        observations:list[str], 
        llm_response:str) -> dict:


    def evaluation_2():
        # EVALUATION 2
        dataset2 = SingleTurnSample(
                user_input = user_query,
                retrieved_contexts = [str(x) for x in observations],
                response = llm_response
            )

        rel_score, generated_questions = calc_relevancy_score(dataset2, summarising_llm, embedding_model)
        rel_score = round(rel_score, 3)

        return rel_score
    
    relevancy_score = evaluation_2()


    try:
        eval_metrics = {
            'relevancy_score': relevancy_score,
            }
    except:
        eval_metrics = {
            'relevancy_score': 0,
            }

    return eval_metrics


def chatbot_fn(message, history, cfg):
    answer, reasoning, eval_metrics = rag(message, cfg)
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
        gr.Markdown("## NCC SOI Chatbot (Demo)")

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