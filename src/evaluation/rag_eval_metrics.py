import asyncio

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextEntityRecall, Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from test_pipeline.patches.context_entity_recall_patch import (
    patch_context_entity_recall,
)
from test_pipeline.patches.pydantic_prompt_patch_for_qwen_llms import (
    patch_pydantic_parsing_for_qwen_llm_output,
)
from test_pipeline.patches.response_relevancy_patch import patch_response_relevancy


def calc_faithfulness(dataset: list[dict], eval_llm: object):
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    evaluator_llm = LangchainLLMWrapper(eval_llm)

    config = RunConfig(timeout=800)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[Faithfulness()],
        llm=evaluator_llm,
        run_config=config,
    )

    return result.scores[0]


def multi_eval(dataset: list[dict], eval_llm: object):
    # A monkey-patch to get access to hidden variables from within the ContextEntityRecall class of the RAGAS library
    patch_context_entity_recall()
    patch_pydantic_parsing_for_qwen_llm_output()

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    evaluator_llm = LangchainLLMWrapper(eval_llm)
    metric1 = ContextEntityRecall()

    config = RunConfig(timeout=800)
    result = evaluate(
        dataset=evaluation_dataset,
        # metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        metrics=[metric1],
        llm=evaluator_llm,
        run_config=config,
    )

    hidden_variables = {
        "ground_truth": metric1._last_ground_truth,
        "contexts": metric1._last_contexts,
    }

    return result.scores[0], hidden_variables


def calc_relevancy_score(dataset: list[dict], eval_llm, embedding_model):
    # A monkey-patch to get access to hidden variables from within the ResponseRelevancy class of the RAGAS library
    patch_response_relevancy()

    evaluator_llm = LangchainLLMWrapper(eval_llm)

    # Wrapper class to adapt SentenceTransformer for RAGAS
    class SentenceTransformerWrapper:
        # def __init__(self, model_name):
        #     self.model = SentenceTransformer(model_name)
        def __init__(self, model):
            self.model = model

        def embed_query(self, text):
            # Use encode with query prefix for E5 models
            return self.model.encode([f"query: {text}"], convert_to_numpy=True)[0]

        def embed_documents(self, texts):
            # Use encode with passage prefix for E5 models
            return self.model.encode(
                [f"passage: {text}" for text in texts], convert_to_numpy=True
            )

    async def calculate_relevancy_score(model):
        # Initialize the embedding model with the wrapper
        embedding_model = SentenceTransformerWrapper(model)

        scorer = ResponseRelevancy(
            llm=evaluator_llm, embeddings=embedding_model, strictness=1
        )
        result = await scorer.single_turn_ascore(dataset, timeout=1500)
        generated_questions = scorer._generated_questions
        result = result.item()
        return result, generated_questions

    # def main():
    #     return result
    result, generated_questions = asyncio.run(
        calculate_relevancy_score(embedding_model)
    )
    print(f"\nGenerated questions for Relevancy Score:\n{generated_questions}")
    print(
        f"Relevancy score: {result} - measures how relevant a response is to the user input. Higher scores indicate better alignment with the user input, while lower scores are given if the response is incomplete or includes redundant information."
    )

    return result, generated_questions
