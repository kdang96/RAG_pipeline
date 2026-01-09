import logging
import typing as t
from functools import wraps

import numpy as np
from ragas.metrics._answer_relevance import ResponseRelevanceOutput, ResponseRelevancy

# This whole module serves as a monkey-patch so that the hidden value "gen_questions" in ResponseRelevancy._calculate_score can extracted

logger = logging.getLogger(__name__)

_original_calc_score = ResponseRelevancy._calculate_score
hidden_variables = []


@wraps(_original_calc_score)
def wrapped(self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict) -> float:
    question = row["user_input"]
    gen_questions = [answer.question for answer in answers]
    committal = np.any([answer.noncommittal for answer in answers])
    if all(q == "" for q in gen_questions):
        logger.warning("Invalid JSON response. Expected dictionary with key 'question'")
        score = np.nan
    else:
        cosine_sim = self.calculate_similarity(question, gen_questions)
        score = cosine_sim.mean() * int(not committal)

    self._generated_questions = gen_questions

    return score


def patch_response_relevancy():
    # Replace the original method with your wrapper
    ResponseRelevancy._calculate_score = wrapped
    print("[Patch] ResponseRelevancy._calculate_score patched successfully.")


def remove_patch():
    # Replace the original method with your wrapper
    ResponseRelevancy._calculate_score = _original_calc_score
    print(
        "[Patch] ResponseRelevancy._calculate_score restored to original successfully."
    )
