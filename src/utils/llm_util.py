import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Iterable, Literal, overload

import ollama
from ollama._types import ChatResponse, GenerateResponse

from test_pipeline.utils.general_util import (
    encode_image_to_base64,
    get_images_recursively,
    validate_all_files_or_all_dirs,
    validate_image_path_list,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def image_validation_n_encoding(image_paths: Iterable[str | Path] = []) -> list[str]:
    if isinstance(image_paths, list):
        input_type: str = validate_all_files_or_all_dirs(image_paths)

        if input_type == "directories":
            image_paths = []
            for path in image_paths:
                image_paths.extend(get_images_recursively(path))
        elif input_type == "files":
            validated: list[Path] = validate_image_path_list(image_paths)
            if not all(validated):
                raise ValueError(
                    "Invalid image paths provided. All files have to be of type: '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'"
                )

        b64_images: list[str] = []

        # LLMs require base64 encoded strings for images
        for image_path in image_paths:
            b64_string: str = encode_image_to_base64(image_path)
            b64_images.append(b64_string)
    else:
        raise ValueError("Images must be a list of strings.")

    return b64_images


class LLMClient:
    def __init__(self, model: str) -> None:
        # If host is provided, use Client() (remote)
        self.client = ollama.Client()
        self.model: str = model

    @overload
    def chat(
        self,
        messages: list[dict[str, Any]] = [],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> Iterator[ChatResponse]: ...
    @overload
    def chat(
        self,
        messages: list[dict[str, Any]] = [],
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    def chat(
        self,
        messages: list[dict[str, Any]] = [],
        *,
        stream: bool = False,
        image_paths: list[str] | list[Path] = [],
        **kwargs: Any,
    ) -> ChatResponse | Iterator[ChatResponse]:
        """
        Docstring for chat

        Args:
            messages: A list of chat messages for the LLM to take into account. The system prompt should be the first message. Followed by user messages.
            stream: If True it streams the output, instead of outputting a single chunk.
            image_paths: A list of image paths or folder path(s) containing images. N.B. all subfolders will be searched for images.
            kwargs: any other Ollama.Client.chat arguments
        """
        # Handle images parameter properly
        if image_paths:
            # For Ollama API, images should encoded in binary strings and added to the message dict
            b64_images: list[str] = image_validation_n_encoding(image_paths)
            messages[-1].update({"images": b64_images})
            logging.info("Inference started.")

            # This is used to suppress type checking errors for ollama.Client methods.
            # The ollama library's typing is not fully compatible with Pylance's static analysis,
            # but the code works correctly at runtime.

            return self.client.chat(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )

            # This is used to suppress type checking errors for ollama.Client methods.
            # The ollama library's typing is not fully compatible with Pylance's static analysis,
            # but the code works correctly at runtime.

        return self.client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=self.model, messages=messages, stream=stream, **kwargs
        )

    @overload
    def generate(
        self,
        prompt: str,
        system: str,
        stream: Literal[True],
        image_paths: list[str] | list[Path] = [],
        **kwargs: Any,
    ) -> Iterator[GenerateResponse]: ...
    @overload
    def generate(
        self,
        prompt: str,
        system: str,
        stream: Literal[False] = ...,
        image_paths: list[str] | list[Path] = [],
        **kwargs: Any,
    ) -> GenerateResponse: ...

    def generate(
        self,
        prompt: str,
        system: str,
        stream: bool = False,
        image_paths: list[str] | list[Path] = [],
        **kwargs: Any,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        """Generate a response from the LLM.
        Args:
            prompt (str): The user's prompt
            system (str): The system's prompt
            stream (bool): Whether to stream the response
            **kwargs: Additional arguments to pass to the LLM
        Returns:
            GenerateResponse
        """

        # Handle images parameter properly
        if image_paths:
            # For Ollama API, images should encoded in binary strings and added to the message dict
            b64_images: list[str] = image_validation_n_encoding(image_paths)
            logging.info("Inference started.")

            return self.client.generate(
                model=self.model,
                prompt=prompt,
                system=system,
                images=b64_images,
                stream=stream,
                **kwargs,
            )

        return self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            stream=stream,
            **kwargs,
        )


class ChatSession:
    def __init__(self, llm: LLMClient, system_prompt: str | None = None):
        self.llm: LLMClient = llm
        self.messages: list[dict[str, Any]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def ask(
        self,
        text: str,
        image_paths: list[str] | list[Path] = [],
        **kwargs: dict[str, str],
    ) -> str:
        """
        Sends a user message and returns the assistant's reply.
        """
        self.messages.append({"role": "user", "content": text})
        response: ChatResponse = self.llm.chat(
            self.messages, stream=False, image_paths=image_paths, **kwargs
        )

        reply: str = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def ask_stream(
        self,
        text: str,
        image_paths: list[str] | list[Path] = [],
        **kwargs: dict[str, str],
    ):
        """
        Sends a user message and streams the assistant's reply.
        """
        self.messages.append({"role": "user", "content": text})
        response: Iterator[ChatResponse] = self.llm.chat(
            self.messages, stream=True, image_paths=image_paths, **kwargs
        )
        collected: Literal[""] = ""
        for chunk in response:
            token = chunk["message"]["content"]
            collected += token
            yield token

        self.messages.append({"role": "assistant", "content": collected})

    def reset(self, keep_system_prompt: bool = True):
        """
        Clears the history. Optionally keeps the system prompt.
        """
        if (
            keep_system_prompt
            and self.messages
            and self.messages[0]["role"] == "system"
        ):
            sys_prompt = self.messages[0]

            self.messages = [sys_prompt]
        else:
            self.messages = []


if __name__ == "__main__":
    llm = LLMClient(model="mistral-small3.2:latest")
    chat = ChatSession(llm, system_prompt="You are an expert document interpreter.")
    image: list[str] = [
        r"test_pipeline\ncc_data\All SOIs\Images\500mm Square RTM Tools\page_1.png",
        r"test_pipeline\ncc_data\All SOIs\Images\500mm Square RTM Tools\page_2.png",
    ]

    # Non-streaming (returns text directly)
    reply = chat.ask("Explain this image.", image_paths=image)
    print(reply)

    # Streaming (returns generator)
    for token in chat.ask_stream("Explain this image again", image_paths=image):
        print(token, end="", flush=True)

    # ===============================================================
    example_json_prompt = {
        "role": "user",
        "content": (
            "Provide a natural language description of the below JSON object. \n"
            f"Example Input:{...}"
            f"Example Output: {...}"
            f"Input to describe: {...}"
        ),
    }
