# The canonical `tokenizer.json`, version `2.0`

What this crate reads and writes. Anything not drawn here is refused.

A `1.0` file is a *legacy* file. tk-convert turns one into a `2.0` file; this crate never reads one.

## Shape

```text
{                                        ← key order is exact, as written
  "version"        : "2.0"
  "truncation"     : null                  encode-time settings, never read
  "padding"        : null
  "added_tokens"   : [ AddedToken, … ]
  "normalizer"     : Component | null
  "pre_tokenizer"  : Component | null
  "post_processor" : Component | null
  "decoder"        : Component | null
  "model"          : Model
}
```

## Pipeline

```text
             normalizer      pre_tokenizer        model       post_processor
  text  ──▶  rewrite text ──▶ cut into words ──▶ words→ids ──▶ add specials  ──▶ ids

                                             decoder
  ids   ─────────────────────────────────────────────────────────────────────▶  text
```

## Component

Every component, at every depth, is an object tagged with `"type"`.

```text
{ "type": "Split", … }                    a component

{ "type": "Sequence", "<children>": [ … ] }    nests, one key per family:

     normalizer ─▶ "normalizers"      post_processor ─▶ "processors"
  pre_tokenizer ─▶ "pretokenizers"           decoder ─▶ "decoders"
```

| slot | `"type"` |
|---|---|
| **normalizer** | `Sequence` `BertNormalizer` `ByteLevel` `Lowercase` `MetaspaceNormalizer` `NFC` `NFD` `NFKC` `NFKD` `Nmt` `Precompiled` `Prepend` `Replace` `Strip` `StripAccents` |
| **pre_tokenizer** | `Sequence` `BertPreTokenizer` `CharDelimiterSplit` `Digits` `FixedLength` `Punctuation` `Split` `UnicodeScripts` `Whitespace` `WhitespaceSplit` |
| **post_processor** | `TemplateProcessing` |
| **decoder** | `Sequence` `BPEDecoder` `ByteFallback` `ByteLevel` `CTC` `Fuse` `Metaspace` `Replace` `Strip` `WordPiece` |

There is one post-processor. `Sequence`, `BertProcessing`, `ByteLevel` and `RobertaProcessing` were
spellings of a template, and tk-convert rewrites each into the `TemplateProcessing` it named -- so
this reader has no arm for them, and only that pass has to know what the old names meant.

There is no `Metaspace` pre-tokenizer. One is two components — it rewrites text *and* cuts it, which
the pipeline keeps apart — so canonically it is spelled as the two it is:

```text
  "normalizer":    {"type":"MetaspaceNormalizer", "replacement":"▁",
                    "prepend":bool, "drop_whitespace":bool}
  "pre_tokenizer": {"type":"Split", "pattern":{"String":"▁"},
                    "behavior":"MergedWithNext", "invert":false}
```

There is no `ByteLevel` pre-tokenizer either. Byte-level describes how the vocabulary is encoded,
so it is `"byte_level"` on the model; the split it used to imply is a plain `Split` on the GPT-2
regex, or nothing at all where it said `use_regex: false`.

tk-convert rewrites a legacy `Metaspace` into that pair, and a legacy `ByteLevel` into the model
flag plus that `Split`. Spelling it out beats folding it back: the
legacy tag could not say `drop_whitespace` without a wrapping `WhitespaceSplit`, could not sit
beside a byte-level model, and forced the normalizer to be last in the chain so a writer could find
it.

## Model

```text
BPE        { "type", "unk_token":str|null,
             "continuing_subword_prefix":str|null, "end_of_word_suffix":str|null,
             "fuse_unk":bool, "byte_fallback":bool, "ignore_merges":bool,
             "byte_level":bool,
             "vocab": {token: id},
             "merges": [ [left, right], … ] }        ← array index IS the rank

Unigram    { "type", "unk_id":int|null,
             "vocab": [ [token, score], … ],         ← array index IS the id
             "byte_fallback":bool }

WordPiece  { "type", "unk_token":str, "continuing_subword_prefix":str,
             "max_input_chars_per_word":int, "vocab": {token: id} }

WordLevel  { "type", "unk_token":str, "vocab": {token: id} }
```

## TemplateProcessing

```text
{ "type", "single": [ Piece, … ], "pair": [ Piece, … ] }

Piece = { "seq": "A"|"B"  }               one input sequence
      | { "ids": [int, …] }               a run of special tokens, by id

        + "type_id": int                  on either, and only when it is not 0
```

A run carries its own ids, so there is no `special_tokens` table and no placeholder names: the
strings behind the ids are in the vocabulary already. The `1.0` spelling — a `SpecialToken`
wrapper naming a table entry — is one more thing tk-convert rewrites.

## AddedToken

```text
{ "id":0, "content":"<s>", "single_word":false, "lstrip":false,
  "rstrip":false, "normalized":false, "special":true }
```

## Refused

Each is an error naming what to convert.

```text
  "version" other than "2.0"               ─┐
  a "model" with no "type"                  │
  "merges" spelled "a b" not ["a","b"]      ├──▶ tk-convert ──▶ canonical 2.0 ──▶ here
  a Metaspace *pre-tokenizer* (it is two)   │
  a Metaspace with "add_prefix_space"       │
  a ByteLevel pre-tokenizer                ─┘

  a vocabulary named by path ("files")     ─────▶ refused, nothing converts it
```
