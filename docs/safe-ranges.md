# Special-token matching with safe ranges on rc0

`SafeRangeInput` pairs one complete rendered string with trusted ranges in which
special tokens may be recognized. It is a renderer-independent input to the existing
`PipelineTokenizer::encode` method, including pairs and batches. Plain string inputs
keep their existing behavior. This replaces this PR's proposed `EncodeSegment`,
`encode_segments`, and `jinja_render_segments` JSON schema; those unreleased APIs are
removed rather than maintained as a second protocol.

## Contract

Here, **safe means trusted to contain control tokens**. It does not mean untrusted
text that has already been sanitized. Ranges use end-exclusive UTF-8 byte positions
in the original rendered string, before tokenizer normalization. The caller owns
the provenance information: accepting a range does not establish its trust.

| Input | Special-token matching |
| --- | --- |
| `SafeRangeInput::new(text, [])` | Disabled everywhere, in both matching passes |
| One `0..text.len()` range for non-empty text | Enabled everywhere |
| Several safe ranges | Enabled inside those ranges; disabled in every gap |
| Adjacent safe ranges | Coalesced before encoding, since they have the same policy |
| Unsorted, overlapping, empty, reversed, out-of-bounds, or non-character-boundary ranges | Constructor returns an error |
| Empty text and an empty range list | Valid empty sequence |

A trusted range can contain arbitrary text and multiple special tokens; it is not
an exact token spelling or a request to insert an id. Within each trusted run, the
existing raw and normalized matchers apply their usual added-token flags. Unknown
spellings remain model text. Outside these runs, special entries are excluded
before matching, so they cannot hide shorter or overlapping ordinary added tokens.
Non-special added tokens remain available in both passes. Explicit range policy
overrides the shared `encode_special_tokens` setting for this input only.

## Rust API

```rust,no_run
use tokenizers::{from_json_file, pipeline::SafeRangeInput};

fn main() -> tokenizers::Result<()> {
    let tokenizer = from_json_file("tokenizer-v2.json")?;
    // Assume <s> is registered as a special token in this tokenizer.
    let user_text = "你好, literal <s>";
    let mut rendered = String::from("<s>");
    let prefix = 0..rendered.len();
    rendered.push_str(user_text);
    let suffix_start = rendered.len();
    rendered.push_str("<s>");
    let suffix = suffix_start..rendered.len();
    let input = SafeRangeInput::new(rendered, [prefix, suffix])?;

    // The renderer already supplied the control markers.
    let single = tokenizer.encode(input.clone(), false).wait()?;
    let batch = tokenizer.encode(vec![input.clone(), input.clone()], false).wait()?;
    let ordinary = SafeRangeInput::new("literal <s>", [])?;
    let pair = tokenizer.encode((input, ordinary), true).wait()?;
    Ok(())
}
```

The post-processor runs once per logical input, using its single or pair template.
`add_special_tokens` controls those template additions, independently of range
matching. `wait()` applies configured padding and restores input order;
`wait_with_padding(None)` disables padding. Iterating the handle yields unpadded
results with their input indices. Batch inputs use the existing streaming worker
pool when its size thresholds are met, with the usual serial fallback. Each ranged
sequence is kept whole during scheduling, so the planner cannot extract special
tokens outside the input's policy. Parallelism is across sequences, not within one
ranged sequence. Pair coordinates start at zero separately for each member.

## Template and Transformers integration

The motivating use case is `apply_chat_template` sanitization discussed in
[Transformers #48052](https://github.com/huggingface/transformers/pull/48052) and the
[range-input review](https://github.com/huggingface/tokenizers/pull/2362#issuecomment-6014488334).
A renderer emits the final text and records the byte ranges originating from
trusted template structure. Message content remains outside those ranges even if
it contains exactly the same spelling as a template marker. The result is passed
to `encode` once, usually with `add_special_tokens=false`; there is no opaque marker
string for user content to impersonate and no separate list of pre-encoded ids to
splice back into the output.

Jinja2, minijinja, or another renderer can produce this input without becoming a
dependency of `tk-encode`. A producer of `(text, recognize_special_tokens)` tuples
can append each text to one string and record the before/after byte lengths when
the flag is true. Empty pieces need no range. A template engine must preserve
provenance through filters, slicing, duplication, loops, and concatenation; finding
all special-token spellings in the final text is not a substitute. If provenance
is lost, do not mark that output trusted. This tokenizer primitive does not make an
arbitrary existing renderer provenance-aware.

Python string indices count Unicode code points, whereas this Rust API uses bytes.
An adapter with final character ranges `(start, end)` must convert each endpoint
with `len(rendered[:endpoint].encode("utf-8"))` (or a precomputed offset table).
Conversion is against the final rendered text, after template transformations and
before tokenizer normalization. User-controlled payloads must not provide their
own trusted ranges. Applications also need an explicit trust policy for tool
results, retrieved documents, roles, and other dynamic fields; the tokenizer cannot
infer it from field names.

## Boundary semantics and current limits

Each maximal trusted or untrusted run is encoded independently. A special-token
match, including whitespace stripping, cannot consume text across a policy
boundary. Normalization happens within a run, so length-changing normalization
cannot shift the policy onto neighboring text. This conservative behavior also
means BPE merges, single-word context, prefix-space behavior, and context-sensitive
normalizers do not span policy boundaries. Output near a boundary can differ from
encoding the whole string without ranges, even if no special spelling occurs.
Adjacent trusted ranges are coalesced to avoid gratuitous boundaries; no text is
reordered, escaped, removed, or replaced by the range API itself.

This policy governs added/special-token matching, not an id blacklist: ordinary
model encoding may still emit an id present in its vocabulary. It is not a general
prompt-injection defense. The rc0 pipeline returns ids, type ids, and attention
masks; it does not yet provide offsets or truncation. Python bindings and the
actual Transformers renderer adapter remain follow-up work. In particular, this
change does not claim support for assistant character masks or automatically
wire an existing Python `apply_chat_template` call to the new Rust input.

For an executable renderer-side example with a legacy or canonical GPT-2 config,
run from `tokenizers/`:

```bash
cargo run --example safe_ranges_encode -- data/gpt2.json '你好 <|endoftext|>'
```

The example surrounds user text with trusted GPT-2 markers and prints the ids.
Only the surrounding markers are recognized as special tokens; the same spelling
inside the argument remains ordinary text.
