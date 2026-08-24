# The canonical `tokenizer.json`

What this crate reads and writes. Anything not drawn here is refused.

## Shape

```text
{                                        ← key order is exact, as written
  "version"        : "1.0"
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
| **pre_tokenizer** | `Sequence` `BertPreTokenizer` `ByteLevel` `CharDelimiterSplit` `Digits` `FixedLength` `Metaspace`† `Punctuation` `Split` `UnicodeScripts` `Whitespace` `WhitespaceSplit` |
| **post_processor** | `Sequence` `BertProcessing` `ByteLevel` `RobertaProcessing` `TemplateProcessing` |
| **decoder** | `Sequence` `BPEDecoder` `ByteFallback` `ByteLevel` `CTC` `Fuse` `Metaspace` `Replace` `Strip` `WordPiece` |

† `Metaspace` is **read-only**: it is the legacy spelling of two components, and it is the one
place where reading and writing are deliberately not symmetric.

A `Metaspace` pre-tokenizer rewrites text *and* cuts it, which the pipeline keeps apart. Reading one
lowers it to the two components it actually is, and each is then written as itself:

```text
  "pre_tokenizer": {"type":"Metaspace", ...}          ← old files, still read
             ─read─▶  MetaspaceNormalizer + Split
             ─write─▶  "normalizer": {"type":"MetaspaceNormalizer", ...}
                       "pre_tokenizer": {"type":"Split", ...}
```

Writing the halves rather than folding them back is a v1 decision. The fold could only emit what the
legacy tag can say — `drop_whitespace` had to come back out as a wrapping `WhitespaceSplit`, a
byte-level model beside a `Metaspace` was a write error, and a `Metaspace` normalizer was required
to be last in the chain so the writer could find it. None of that survives; the cost is that a file
this writer produces does not load in `tokenizers` before v1, which v1 does not promise.

## Model

```text
BPE        { "type", "unk_token":str|null,
             "continuing_subword_prefix":str|null, "end_of_word_suffix":str|null,
             "fuse_unk":bool, "byte_fallback":bool, "ignore_merges":bool,
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

Piece = { "Sequence"    : {"id":"A"|"B",    "type_id":int} }
      | { "SpecialToken": {"ids":[int, …],  "type_id":int} }   ← the ids, not a name
```

A `SpecialToken` may carry `"id"` naming a `"special_tokens"` entry instead, which is what
every file written before this crate does. The ids are read out of that table; nothing else
in it is, and neither is written back.

## AddedToken

```text
{ "id":0, "content":"<s>", "single_word":false, "lstrip":false,
  "rstrip":false, "normalized":false, "special":true }
```

## Refused

Each is an error naming what to convert.

```text
  a "model" with no "type"                 ─┐
  "merges" spelled "a b" not ["a","b"]      ├──▶ tk-convert ──▶ canonical ──▶ here
  a Metaspace spelled "add_prefix_space"   ─┘

  a vocabulary named by path ("files")     ─────▶ refused, nothing converts it
```
