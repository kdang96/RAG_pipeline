import json
from pathlib import Path
from typing import Any, Generator


def read_jsonl(path: Path) -> Generator[Any, Any, Any]:
    """Load JSON and raise a descriptive exception on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)                  
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read {path}") from exc
    

def write_jsonl(chunks: list[dict[str, Any]], output_path: Path | str) -> None:
    """
    Persist a list of chunk dictionaries as JSON Lines.
    """

    ROOT = Path(__file__).resolve().parents[3]
    full_output_path = ROOT / output_path
    with open(full_output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")


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

