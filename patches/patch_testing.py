from langchain_ollama.chat_models import ChatOllama
from rag_evaluation.rag_eval_metrics import multi_eval
from sentence_transformers import SentenceTransformer

# summarising_llm = ChatOllama(model="mistral24b-instr-gpu40", temperature=0, num_ctx=132768, keep_alive=40, num_gpu=40)
summarising_llm = ChatOllama(
    model="qwen3:4b", temperature=0, num_ctx=132768, keep_alive=0, num_gpu=40
)
embedding_model = SentenceTransformer(
    model_name_or_path="intfloat/e5-large-v2", device="cuda", trust_remote_code=True
)

dataset = []

dataset.append(
    {
        "user_input": "What is the project title for work order S-WO003067?",
        "retrieved_contexts": [
            "The project title for work order S-WO003067 is **Baker Hughes Life of Field Cap Build**."
        ],
        "response": "The project title for work order S-WO003067 is **Baker Hughes Life of Field Cap Build**.",
        "reference": "Baker Hughes Life of Field Cap Build",
    }
)

result, hidden_variables = multi_eval(dataset, summarising_llm)
# result, hidden_variables = calc_relevancy_score(dataset, summarising_llm, embedding_model)
print(hidden_variables)
