import json

import ollama

# This works poorly for our purposes!


def read_json(file_name: str):
    with open(file_name, "r") as f:
        return json.load(f)


def save_json(file_name: str, data: list):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


all_proj_data = read_json(
    r"test_pipeline/data_files/DIS json/15_08_25_dis_data_flat2_eval_version.json"
)

splits = [0, 20, 40, 60, 80, 100]
full_qna_set = []

for i, value in enumerate(splits[:-1]):
    data_batch = all_proj_data[value : splits[i + 1]]
    prompt = f"Generate a list of 20 question–answer pairs for the following entities:\n {json.dumps(data_batch, indent=2)}"

    response = ollama.generate(
        model="qwen3:32b",
        prompt=prompt,
        system="""You will be given a list of JSON objects.
                Generate a list of 20 question-answer pairs. Each question has to be directly answerable from the user prompt. Use only factoid questions.
                Output as JSON with keys: "question": "<generated question>", "expected_answer": "<generated answer>".""",
    )

    output = response.response
    print(output)
    full_qna_set.append(list(full_qna_set))

save_json(r"test_pipeline/data_files/DIS json/qna.json", output)
