"""Regenerate the golden files: the exact-output contract of the bindings.

A golden file records everything encode and decode produce for one tokenizer
on a fixed set of inputs — ids, tokens, offsets, masks, word ids, decoded
strings for short samples, plus ids digests over the data/ fixture corpora
for breadth. test_golden.py diffs the current build against these files, so
any behavioral deviation from the released wheel surfaces, including ones
nobody thought to assert.

Goldens are never hand-edited. They come from the released tokenizers wheel —
the reference the 1.0 rewrite must match: `make golden-regen` installs it
into .release/ and runs this script with PYTHONPATH pointing there (the
release shares our package name; same trick as benches/bench_vs_release.py).

Inputs: every model in tk-encode/examples/bench_models.json, the corpora
under data/fixtures (a short excerpt captured verbatim, the rest as a capped
digest), and the edge-case strings below. Regenerating needs the data
fetched: `make -C ../../tokenizers bench-models fixtures`.

Output format: one JSON-lines file per model under goldens/ — meta line,
then one line per sample/pair/digest, so a behavior change diffs line by line.
"""

import hashlib
import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "tokenizers" / "data"
BENCH_MODELS = REPO / "tokenizers" / "tk-encode" / "examples" / "bench_models.json"
GOLDENS = Path(__file__).parent / "goldens"

EXCERPT_CHARS = 160
# Fixture files are ~5 MB each; the first 200k chars already exercise the
# whole distribution, and keep regeneration under a minute.
DIGEST_CAP_CHARS = 200_000

EDGE_CASES = {
    "empty": "",
    "spaces": "   ",
    "hello": "Hello world",
    "punctuation": "Hello, world!! How's it going? (fine; thanks...)",
    "whitespace-mix": "line one\nline two\r\n\tindented\n  trailing  ",
    "accents": "café naïve résumé — déjà vu",
    # The same letters in decomposed form (base letter + combining accent):
    # ids must come out identical to the composed spelling above once the
    # tokenizer normalizes, or reveal that it does not.
    "accents-decomposed": "cafe\u0301 nai\u0308ve re\u0301sume\u0301",
    "emoji": "🤗 emoji, families 👨‍👩‍👧‍👦, flags 🇫🇷 and skin tones 👍🏽",
    "cjk": "漢字とひらがなとカタカナが混ざった文章です。",
    "korean": "한국어 텍스트 조각",
    "rtl": "مرحبا بالعالم — שלום עולם",
    "numbers": "1234567890, 3.14159, 1,000,000th",
    "code": "def f(x):\n    return x**2  # squared\nprint(f'{f(3)=}')",
    "long-word": "Donaudampfschifffahrtsgesellschaftskapitänsmützenabzeichen",
    "url-email": "https://example.com/a/b?q=1&r=2#frag user.name+tag@example.co.uk",
    # no-break, thin and ideographic spaces
    "unicode-spaces": "a\u00a0b\u2009c\u3000d",
    "math": "∑ᵢ xᵢ² ≤ ∫₀^∞ e⁻ˣ dx ≈ 1",
}

PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Question in English?", "Ответ на русском языке, с цифрами 123."),
]


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def sample_record(tok: Tokenizer, source: str, text: str) -> dict:
    enc = tok.encode(text)
    return {
        "kind": "sample",
        "source": source,
        "text": text,
        "ids": enc.ids,
        "ids_no_specials": tok.encode(text, add_special_tokens=False).ids,
        "tokens": enc.tokens,
        "offsets": [list(pair) for pair in enc.offsets],
        "type_ids": enc.type_ids,
        "special_tokens_mask": enc.special_tokens_mask,
        "word_ids": enc.word_ids,
        "decoded": tok.decode(enc.ids, skip_special_tokens=True),
        "decoded_with_specials": tok.decode(enc.ids, skip_special_tokens=False),
    }


def pair_record(tok: Tokenizer, text: str, pair: str) -> dict:
    enc = tok.encode(text, pair)
    return {
        "kind": "pair",
        "text": text,
        "pair": pair,
        "ids": enc.ids,
        "tokens": enc.tokens,
        "type_ids": enc.type_ids,
        "sequence_ids": enc.sequence_ids,
        "special_tokens_mask": enc.special_tokens_mask,
        "offsets": [list(pair) for pair in enc.offsets],
    }


def digest_record(tok: Tokenizer, relative: str, text: str) -> dict:
    ids = tok.encode(text).ids
    return {
        "kind": "digest",
        "file": relative,
        "cap_chars": DIGEST_CAP_CHARS,
        "text_sha256": text_digest(text),
        "n_tokens": len(ids),
        "ids_sha256": ids_digest(ids),
    }


def fixture_files() -> list[Path]:
    files = sorted((DATA / "fixtures" / "lang").glob("*.txt")) + sorted(
        (DATA / "fixtures" / "modalities").glob("*.txt")
    )
    if not files:
        sys.exit(f"no fixture corpora under {DATA / 'fixtures'} — run `make -C {REPO / 'tokenizers'} fixtures`")
    return files


def main():
    if not tokenizers.__version__.startswith("0."):
        sys.exit(
            f"goldens must come from the released 0.x wheel, not {tokenizers.__version__} — run `make golden-regen`"
        )

    fixtures = fixture_files()
    GOLDENS.mkdir(exist_ok=True)
    for model in json.loads(BENCH_MODELS.read_text()):
        tokenizer_file = DATA / model["file"]
        if not tokenizer_file.is_file():
            sys.exit(f"{tokenizer_file} missing — run `make -C {REPO / 'tokenizers'} bench-models`")
        tok = Tokenizer.from_file(str(tokenizer_file))

        records = [
            {
                "kind": "meta",
                "model": model["name"],
                "tokenizer_file": model["file"],
                "tokenizers_version": tokenizers.__version__,
            }
        ]
        records += [sample_record(tok, f"edge:{name}", text) for name, text in EDGE_CASES.items()]
        records += [
            sample_record(tok, f"{f.relative_to(DATA).as_posix()}[:{EXCERPT_CHARS}]", f.read_text()[:EXCERPT_CHARS])
            for f in fixtures
        ]
        records += [pair_record(tok, text, pair) for text, pair in PAIRS]
        records += [
            digest_record(tok, f.relative_to(DATA).as_posix(), f.read_text()[:DIGEST_CAP_CHARS]) for f in fixtures
        ]

        path = GOLDENS / f"{model['name']}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records) + "\n")
        print(f"{path.name}: {len(records) - 1} records from tokenizers {tokenizers.__version__}")


if __name__ == "__main__":
    main()
