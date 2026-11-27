"""Render a chat message with Jinja and mark the rendered content as untrusted.

From bindings/python, run `make test-data`, then `python examples/untrusted_spans_chat.py`.
"""

from pathlib import Path

from jinja2 import Template

from tokenizers import Tokenizer

HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
EOT = "<|eot_id|>"
BODY = Template("{{ content | trim }}")


def render_chat(content: str) -> tuple[str, list[tuple[int, int]]]:
    body = BODY.render(content=content)
    start = len(HEADER)
    return f"{HEADER}{body}{EOT}", [(start, start + len(body))] if body else []


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data" / "llama-3-tokenizer.json"
    spec = Tokenizer.from_file(path)
    label = "[note-demo]"
    spec.add_tokens([label])
    tokenizer = spec.compile()
    text, spans = render_chat(f"  café {label}  ")

    plain = tokenizer.encode(text, add_special_tokens=False)
    protected = tokenizer.encode(text, add_special_tokens=False, untrusted_spans=spans)

    assert tokenizer.token_to_id(label) in plain.ids
    assert tokenizer.token_to_id(label) not in protected.ids
    assert protected.ids[-1] == tokenizer.token_to_id(EOT)
    assert tokenizer.decode(protected.ids, skip_special_tokens=False) == text
    print("The added label stayed plain text, the template marker kept its ID, and the text round-tripped.")
