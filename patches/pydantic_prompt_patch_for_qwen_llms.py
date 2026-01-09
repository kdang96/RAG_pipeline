from __future__ import annotations

import logging
import typing as t

from langchain_core.prompt_values import StringPromptValue as PromptValue
from pydantic import BaseModel
from ragas.callbacks import ChainType, new_group
from ragas.exceptions import RagasOutputParserException

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.llms.base import BaseRagasLLM

logger = logging.getLogger(__name__)

# type variables for input and output models
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)

from functools import wraps  # noqa: E402

from ragas.prompt.pydantic_prompt import PydanticPrompt, RagasOutputParser  # noqa: E402

"""
This whole module serves as a monkey-patch so that the Qwen model can be used with ContextEntityRecall
without the Pydantic parsing of the output from the model failing, because it outputs its
reasoning chain with the ouput in the same string.
"""

_original_generate_multiple = PydanticPrompt.generate_multiple
hidden_variables = []


@wraps(_original_generate_multiple)
async def wrapped(
    self,
    llm: BaseRagasLLM,
    data: InputModel,
    n: int = 1,
    temperature: t.Optional[float] = None,
    stop: t.Optional[t.List[str]] = None,
    callbacks: t.Optional[Callbacks] = None,
    retries_left: int = 3,
) -> t.List[OutputModel]:
    """
    Generate multiple outputs using the provided language model and input data.

    Parameters
    ----------
    llm : BaseRagasLLM
        The language model to use for generation.
    data : InputModel
        The input data for generation.
    n : int, optional
        The number of outputs to generate. Default is 1.
    temperature : float, optional
        The temperature parameter for controlling randomness in generation.
    stop : List[str], optional
        A list of stop sequences to end generation.
    callbacks : Callbacks, optional
        Callback functions to be called during the generation process.
    retries_left : int, optional
        Number of retry attempts for an invalid LLM response

    Returns
    -------
    List[OutputModel]
        A list of generated outputs.

    Raises
    ------
    RagasOutputParserException
        If there's an error parsing the output.
    """
    callbacks = callbacks or []

    processed_data = self.process_input(data)
    prompt_rm, prompt_cb = new_group(
        name=self.name,
        inputs={"data": processed_data},
        callbacks=callbacks,
        metadata={"type": ChainType.RAGAS_PROMPT},
    )
    prompt_value = PromptValue(text=self.to_string(processed_data))
    resp = await llm.generate(
        prompt_value,
        n=n,
        temperature=temperature,
        stop=stop,
        callbacks=prompt_cb,
    )

    output_models = []
    parser = RagasOutputParser(pydantic_object=self.output_model)
    for i in range(n):
        output_string = resp.generations[0][i].text

        # ~~~~~~~~~PATCH START~~~~~~~~~~
        try:
            # for Qwen model
            output_string = output_string.split("</think>\n\n", 1)[1]
        except Exception:
            output_string = output_string
        # ~~~~~~~~~PATCH END~~~~~~~~~~

        try:
            answer = await parser.parse_output_string(
                output_string=output_string,
                prompt_value=prompt_value,
                llm=llm,
                callbacks=prompt_cb,
                retries_left=retries_left,
            )
            processed_output = self.process_output(answer, data)  # type: ignore
            output_models.append(processed_output)
        except RagasOutputParserException as e:
            prompt_rm.on_chain_error(error=e)
            logger.error("Prompt %s failed to parse output: %s", self.name, e)
            raise e

        prompt_rm.on_chain_end({"output": output_models})

    return output_models


def patch_pydantic_parsing_for_qwen_llm_output():
    # Replace the original method with your wrapper
    PydanticPrompt.generate_multiple = wrapped
    print("[Patch] PydanticPrompt.generate_multiple patched successfully.")


def remove_patch():
    # Replace the original method with your wrapper
    PydanticPrompt.generate_multiple = _original_generate_multiple
    print("[Patch] PydanticPrompt.generate_multiple restored to original successfully.")
