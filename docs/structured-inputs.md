# Structured special-token inputs on rc0

`PipelineTokenizer::encode_segments` encodes a producer's sequence of ordinary text
and trusted special-token spellings. This is opt-in; `encode` keeps its existing
behavior. Template renderers can use the Rust API directly or emit the JSON schema
below. The caller must decide which segments are trusted: parsing JSON does not
establish trust, and user-controlled text must not be allowed to set `special: "yes"`.

```rust,no_run
use tokenizers::{from_json_file, segments_from_json, pipeline::EncodeSegment};

fn main() -> tokenizers::Result<()> {
let tokenizer = from_json_file("tokenizer-v2.json")?;
let segments = vec![
    EncodeSegment::special("<s>"), // must be registered by this tokenizer
    EncodeSegment::text("User text, including any literal <s> spelling"),
];
let encodings = tokenizer.encode_segments(&segments, true).wait()?;

let segments = segments_from_json(r#"{
  "version": 1,
  "type": "jinja_render_segments",
  "segments": [
    {"content": "<s>", "special": "yes"},
    {"content": "User text", "special": "no"}
  ]
}"#)?;
let batch = tokenizer.encode_segments_batch(&[segments], true).wait()?;
Ok(())
}
```

The JSON reader lives in `tk-serialize` and is re-exported by `tokenizers`; it adds
no dependency to `tk-encode`. It rejects unknown, duplicate, or missing fields,
wrong types, unsupported versions, and flags other than `"yes"`/`"no"`. `version`
must be the integer literal `1`. The schema is a single sequence, without pair or
pretokenized inputs. An empty `segments` array is valid.

A special segment must exactly match a stored special-token spelling in the
pipeline's raw or normalized added vocabulary. It is emitted directly as one id;
normalization, whitespace stripping, and single-word matching do not loosen this
check. Unknown tokens and tokens registered only as ordinary added tokens are
errors. For normalized registrations, rc0 stores the normalized spelling.

Adjacent ordinary segments are concatenated before normalization and encoding, so
producer-defined text boundaries do not prevent BPE merges. Special-token matching
is disabled for ordinary segments in both raw and normalized passes. Non-special
added-token matching uses a lazily cached vocabulary with special entries removed
before matching. This preserves shorter and overlapping ordinary added tokens. This changes no shared policy.
Ordinary model encoding can still produce any id present in the model vocabulary;
this API controls special-token matching, not a blanket ban on particular ids.

The post-processor runs once on the complete sequence. `add_special_tokens` controls
its template additions; it does not remove explicitly supplied special segments.
Padding is applied by `wait`, or overridden with `wait_with_padding`. Iterating the
handle directly yields unpadded results, just like rc0's existing encode handle.
Batch results preserve input order. Batches use rc0's configured worker pool when
available, with a serial fallback; these handles are fully computed before the
method returns and do not provide asynchronous streaming.

This revision targets `tokenizers-rc0`. It does not restore the removed
`TokenizerImpl`, Python setters, or legacy Python sync/async bindings. Output uses
rc0's ids, type ids and attention masks; offsets and truncation require upstream
pipeline support. Renderer/Transformers integration and Python bindings remain
follow-up work.

To exercise the complete Rust/JSON path locally, run from `tokenizers/`:

```bash
cargo run --example structured_encode -- data/gpt2.json segments.json
```

The example accepts a legacy or canonical tokenizer config and prints the encoded
ids as JSON. `segments.json` uses the schema shown above.
