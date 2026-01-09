from functools import wraps

from ragas.metrics import ContextEntityRecall

"""
This whole module serves as a monkey-patch so that the hidden values
"ground_truth.entities" and "contexts.entities"  in ContextEntityRecall._ascore can be extracted
"""

_original_ascore = ContextEntityRecall._ascore


@wraps(_original_ascore)
async def wrapped(self, *args, **kwargs):
    # Extract parameters like the original method
    row = kwargs.get("row") or args[0]
    callbacks = kwargs.get("callbacks") or args[1]

    # Recreate what the method does
    ground_truth, contexts = row["reference"], row["retrieved_contexts"]
    ground_truth = await self.get_entities(ground_truth, callbacks=callbacks)
    contexts = await self.get_entities("\n".join(contexts), callbacks=callbacks)

    # Attaches the hiden values to the instance so that they are easily accessible from outside the library
    self._last_ground_truth = ground_truth.entities
    self._last_contexts = contexts.entities

    # Continue with the library's logic
    return self._compute_score(ground_truth.entities, contexts.entities)


def patch_context_entity_recall():
    # Replace the original method with your wrapper
    ContextEntityRecall._ascore = wrapped
    print("[Patch] ContextEntityRecall._ascore patched successfully.")


def remove_patch():
    # Replace the original method with your wrapper
    ContextEntityRecall._ascore = _original_ascore
    print("[Patch] ContextEntityRecall._ascore restored to original successfully.")
