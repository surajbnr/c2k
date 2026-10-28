"""§6.2 chunking. Regression: a paper whose text contains a tokenizer
special-token string (e.g. a code-model paper quoting FIM tokens) used to
crash `c2k chunk` for the whole corpus."""

from __future__ import annotations

from c2k.chunk import chunk_text


def test_text_containing_special_token_strings_is_chunked_as_plain_text() -> None:
    text = "Prefix tokens like <|fim_middle|> and <|endoftext|> appear in the paper. " * 40
    chunks = chunk_text("doc1", text, tokens=100, overlap=20)
    assert chunks, "expected at least one chunk"
    assert any("<|fim_middle|>" in text[c.char_start : c.char_end] for c in chunks)
