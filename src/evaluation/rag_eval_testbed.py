import json

import requests
from rag_eval_metrics import multi_eval

from test_pipeline.pipelines.data_import.utils.milvus_util import search


def read_json(file_name: str):
    with open(file_name, "r") as f:
        return json.load(f)


def run_through_llm(user_query: str, retrieved_text: str):
    ollama_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "mistral24b-instr-gpu40",
        "prompt": f"""[INST] <<SYSTEM>>
                "You are a helpful assistant that answers questions based on given documents only."
            <</SYSTEM>> Question: {user_query}\n\nDocuments: {retrieved_text} [/INST]""",
        "stream": False,
        "options": {"num_ctx": 132768, "temperature": 0},
        "keep_alive": 0,
    }

    response = requests.post(ollama_url, json=payload)
    # text_response = response.content['response']

    try:
        text_response = response.json()["response"]
    except Exception:
        text_response = response.text

    return text_response


# Test 1
# query = ["Have we manufactured with unidirectional material?"]
# expected_responses = ["Manufacture a Prepreg bracket using UD and Twill fabrics, to hold a camera over the SQCDPE board to broadcast meetings on Teams."]

# # Test 2
# query = ["Have we manufactured any aircraft parts? Give me their WO numbers."]
# expected_responses = ["WO numers of project about manufacturing of aerospace components."]

# # Test 3
# query = ["Has James Waldock submitted any DISs?"]
# expected_responses = ["Yes, James Waldock submitted S-WO000140, S-WO000147, S-WO000163, S-WO000181."]

# Test 4
query = [
    "How many DISs has James Waldock submitted? What was their WO numbers and who was the PM?"
]
expected_responses = [
    "James Waldock has submitted four DISs. Their WO numbers are S-WO000140, S-WO000147, S-WO000163, S-WO000181. The PM on all of them is Andrew Bruton."
]
retrieved_json = search(
    db_name="milvus_demo.db",
    collection_name="DISs_e5_large_v2_described",
    user_query=query,
    search_col="natural_language_vector",
    output_fields=[
        "wo_number",
        "project_title",
        "originator",
        "natural_language_descr",
        "project_manager",
    ],
    search_radius=0.78,
    k_limit=8,
)
# retrieved_json = read_json("test_pipeline/data_files/JW_15_08_25_dis_manually_described.json")
# llm_descr = read_json("test_pipeline/data_files/JW_15_08_25_dis_llm_described.json")
retrieved_descriptions = [x["entity"]["natural_language_descr"] for x in retrieved_json]
retrieved_distances = [x["distance"] for x in retrieved_json]

# print(retrieved_distances)

llm_response = run_through_llm(user_query=query, retrieved_text=retrieved_descriptions)
# llm_response = "This document indicates that James Waldock has submitted a DIS. There might be more DISs submitted by him in the remaining 30 entities that are not listed."

print("Query:", query)
print(retrieved_descriptions)
print("LLM Response:", llm_response)

# EVALUATION 1
dataset = []

dataset.append(
    {
        "user_input": query[0],
        # "retrieved_contexts":[str(x) for x in list(retrieved_json)],
        "retrieved_contexts": retrieved_descriptions,
        "response": llm_response,
        "reference": expected_responses[0],
    }
)

result = multi_eval(dataset)
print(result)
# result = list(result)
# print(f"LLM {result[0][0]}: {result[0][1]} \n- computed using user_input, reference and the retrieved_contexts. Range:[0,1]. Higher values indicating better performance. \n")
# print(f"{result[1][0]}: {result[1][1]} \n- measures how factually consistent a response is with the retrieved context. Range:[0,1]. Higher values indicating better consistency.\n")
# print(f"""{result[2][0]}: {result[2][1]} \n- compares and evaluates the factual accuracy of the generated response with the reference. Range:[0,1]. Higher values indicating better consistency. The metric uses the LLM to first break down the response and reference into claims and then uses natural language inference to determine the factual overlap between the response and the reference. Factual overlap is quantified using precision, recall, and F1 score \n""")


# EVALUATION 2
# dataset2 = SingleTurnSample(
#         user_input = query[0],
#         retrieved_contexts = [str(x) for x in list(retrieved_json)],
#         response = llm_response
#     )

# rel_score = calc_relevancy_score(dataset2)
