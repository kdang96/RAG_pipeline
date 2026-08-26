from utils.general_util import format_entities_for_llm, read_jsonl, write_jsonl


def test_write_then_read_roundtrip(tmp_path):
    rows = [{"chunk_id": 0, "chunk": "hello"}, {"chunk_id": 1, "chunk": "world"}]
    out = tmp_path / "sub" / "chunks.jsonl"

    write_jsonl(rows, out)

    # Regression: the file must land exactly where the caller asked, not
    # somewhere resolved against the source tree.
    assert out.exists()
    assert list(read_jsonl(out)) == rows


def test_write_jsonl_preserves_unicode(tmp_path):
    out = tmp_path / "c.jsonl"
    write_jsonl([{"chunk": "Baden-Württemberg"}], out)
    assert "Baden-Württemberg" in out.read_text(encoding="utf-8")


def test_format_entities_rounds_floats():
    text = format_entities_for_llm([{"score": 0.123456789, "name": "x"}])
    assert "score: 0.12346" in text
    assert "name: x" in text


def test_format_entities_separates_records():
    text = format_entities_for_llm([{"a": 1}, {"b": 2}])
    assert "a: 1" in text
    assert "b: 2" in text
