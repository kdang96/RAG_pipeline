import base64
import csv
import io
import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Generator, Tuple

import typer
from PIL import Image

if sys.platform == "win32":
    import win32com.client as win32

app = typer.Typer()


@app.command()
def read_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON and raise a descriptive exception on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read {path}") from exc
    
    
@app.command()
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




def save_json(file_name: str, data: list) -> None:
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


@app.command()
def get_subfolder_paths(dir_path: str) -> list[str]:
    return [str(path) for path in Path(dir_path).rglob("*/") if path.is_dir()]


# encodes images to binary for the model to undertand
def encode_image_to_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


def get_images_recursively(folder: str | Path):
    """
    Get all images within a folder, including subfolders.

    :param folder: Folder path
    :type folder: str | Path
    """
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    folder = Path(folder)

    return [p for p in folder.rglob("*") if p.suffix.lower() in image_exts]

def validate_file_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file()


def validate_image_path(path: str | Path) -> bool:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    path = Path(path)
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def validate_image_path_list(paths: list[str | Path]) -> list[Path]:
    validated = []
    for p in paths:
        validated.append(validate_image_path(p))
    return validated


def validate_all_files_or_all_dirs(items: list[str | Path]) -> str:
    if not all(isinstance(x, str | Path) for x in items):
        raise TypeError("All items must be strings representing paths.")

    paths = [Path(x) for x in items]

    # Check existence
    for p in paths:
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")

    # Classification flags
    all_files = all(p.is_file() for p in paths)
    all_dirs = all(p.is_dir() for p in paths)

    # Mixed case error
    if not (all_files or all_dirs):
        raise ValueError(
            "Paths must be ALL files or ALL directories. Mixed types detected."
        )

    # Return what they are
    return "files" if all_files else "directories"


def get_file_paths(dir_path: str | Path, extension: str) ->  list[Path]:
    """Return a list of all *extension* files under *dir_path* (recursively)."""
    directory = Path(dir_path)
    json_paths = list(directory.glob(extension))
    return json_paths


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


def validate_docx(path: str) -> Path:
    p = Path(path)

    if not p.exists():
        raise typer.BadParameter("File does not exist.")

    if not p.is_file():
        raise typer.BadParameter("Path must be a file, not a directory.")

    if p.suffix.lower() != ".docx":
        raise typer.BadParameter("File must have .docx extension.")

    return p


# minimises the retrieved RAG context for faster LLM inference by converting it to CSV
def minimise_soi_context_size(data: list[dict]) -> str:
    all_entity_keys = list(data[0]["entity"].keys())

    # Build CSV header
    header = [x for x in all_entity_keys]

    # Build CSV rows
    rows = []
    for item in data:
        # row = [item['id']]
        # append entity values in the same order as header
        row = [
            item["entity"].get(k, "") for k in all_entity_keys if k != "doc_number_PK"
        ]
        rows.append(row)

    # Write CSV to string
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    csv_string = output.getvalue().strip()

    # print(csv_string)
    return csv_string


# minimises the retrieved RAG context for faster LLM inference by converting it to CSV
def minimise_context_size(data: list[dict]) -> str:
    all_entity_keys = list(data[0]["entity"].keys())
    all_entity_keys = [x for x in all_entity_keys if x != "wo_number"]
    # Build CSV header
    header = ["wo_number"] + all_entity_keys

    # Build CSV rows
    rows = []
    for item in data:
        row = [item["id"]]
        # append entity values in the same order as header
        row += [item["entity"].get(k, "") for k in all_entity_keys if k != "wo_number"]
        rows.append(row)

    # Write CSV to string
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    csv_string = output.getvalue().strip()

    print(csv_string)
    return csv_string


def save_text(text: str, save_to_path: str):
    with open(save_to_path, "w", encoding="utf-8") as file:
        file.write(text)


def read_text(file_name: str):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()


def find_docx_files(directory: str | Path):
    # Create a Path object for the directory
    path = Path(directory)
    # Find all .docx files recursively
    docx_files = path.rglob("*.docx")
    # Return list of full paths as strings
    return [file.as_posix() for file in docx_files]


def find_all_files(directory: str):
    path = Path(directory)
    files = [str(f) for f in path.iterdir() if f.is_file()]

    return files


def find_all_non_docx_word_files(directory):
    path = Path(directory)
    extensions = ["*.doc", "*.docm", "*.dot", "*.dotm", "*.dotx"]

    docx_paths = []
    for ext in extensions:
        ext_paths = [str(file) for file in path.rglob(ext)]
        docx_paths.extend(ext_paths)

    return docx_paths


def convert_all_to_docx(directory, output_dir=None):
    path = Path(directory)
    files = find_all_non_docx_word_files(path)

    if output_dir is None:
        output_dir = path  # save alongside originals
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False  # run in background

    converted = []
    try:
        for f in files:
            input_path = Path(f)
            output_path = output_dir / (input_path.stem + ".docx")

            doc = word.Documents.Open(str(input_path))
            doc.SaveAs(str(output_path), FileFormat=16)  # 16 = docx
            doc.Close()

            converted.append(output_path)

    finally:
        word.Quit()

    return converted


def create_eval_report(data: dict):
    curr_datetime = datetime.now().strftime(r"%H%M_%d%m%y")
    save_json(f"test_pipeline/eval_reports/eval_report_{curr_datetime}.json", data)


def crop_image_to_relevant_area(image_path: str, crop_box: Tuple[int]):
    """
    Crop an image and convert to base64 string.

    Args:
        image_path: Path to the input image.
        crop_box: (left, upper, right, lower) coordinates for cropping. e.g., (115, 300, 1540, 2250)

    Returns:
        str: Base64-encoded string of the cropped image.
    """
    # Open image
    img: Any = Image.open(image_path)

    # Crop image (removes distracting numbers)
    cropped = img.crop(crop_box)

    cropped.show()

    # Convert cropped image to bytes
    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")

    # Encode to base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_base64


if __name__ == "__main__":
    # convert_all_to_docx(
    #     r"C:\kdangelov\Code\LLM_Atlas\test_pipeline\ncc_data\All SOIs\OneDrive_1_26-09-2025",
    #     r"C:\kdangelov\Code\LLM_Atlas\test_pipeline\ncc_data\All SOIs\temp",
    # )

    app()
