"""
Minimal agentic retrieval-augmented QA.

An LLM is given access to a single retrieval tool (`similarity_search`) and may
invoke it to fetch document chunks from a Milvus vector store before answering
a user query. The agent performs a single tool invocation at most and does not
use multi-step planning or reflection.

This module is intended for qualitative demonstration only; retrieval quality
is evaluated separately using an offline pipeline.
"""


from typing import Tuple
import logging
from ollama._types import ChatResponse
import ollama
from utils.general_util import format_entities_for_llm
from config.config import Config
from vector_store.milvus import (
    get_collection_fields,
    search
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
You can choose between using the tool available or answering from your own weights, but you need to make that decision explicit.

The tool gives you access to a database on information relating to legislation and treaties from the European Union.

If you are to use the tools, the below-mentioned information is important.

Field: database field to search in, one of {fields}\n
For fields that have a vector equivalent, use the Search function on the vector field.

User: {user_query}\n
"""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)


# ReAct loop for agentic search
def rag_flow(question: str, config: Config, history: list[dict] | None = None):
    fields_n_descriptions = get_collection_fields(config.db_path, config.collection)
    logger.info(f"Fields & descriptions: {fields_n_descriptions}")

    prompt = BASE_PROMPT.format(user_query=question, fields=fields_n_descriptions)

    prior_turns = history or []

    initial_response = ollama.chat(
                    model="gpt-oss:20b",
                    messages=[*prior_turns, {"role": "user", "content": prompt}],
                    tools=TOOLS,
                    options = {
                        "tool_choice": "auto",
                        "temperature": 1
                        }
                )
    initial_reasoning = initial_response.message.thinking
    logger.info("Model initial reasoning:\n%s", initial_reasoning)

    if initial_response.message.tool_calls:
        # Look for an Action
        obs, obs_formatted = tools(initial_response, config)

        logger.info(f"User query:{question}\n Available information:" + obs_formatted)

        messages = [
            *prior_turns,
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
        
        return final_response.message.content, initial_reasoning 

    return initial_response.message.content, initial_reasoning


# Here is defined function logic that the agent can use
def tools(llm_response: ChatResponse, config: Config) -> Tuple[list[dict], str]:
    """
    Resolve the LLM's tool call by executing the requested function.

    The function currently supports only the ``similarity_search`` action.
    It performs a vector similarity query on the Milvus collection and
    returns both raw and LLM‑friendly formatted results.
    """
    action = llm_response.message.tool_calls[0].function.name
    args = llm_response.message.tool_calls[0].function.arguments
    action_input = args["input"]

    logger.info("Function: %s", action)
    logger.info("Tools args:\n%s", args)

    if action.lower() == "similarity_search":
        obs = search(
            config=config,
            user_queries=[action_input],
            search_col="combined_vector",
            output_fields = ["doc_title", "chunk_id", "heading_2", "heading_3", "heading_4", "chunk"],
            search_radius = 0.815,
            k_limit = 10
        )
        
        if obs[0]:
            obs = [x[0]["entity"] for x in obs]
            obs_formatted = format_entities_for_llm(obs)
        else:
            obs = "No information was retrieved."
            obs_formatted = obs
    else:
        obs = f"Unknown action: {action}"
        obs_formatted = obs

    return obs, obs_formatted