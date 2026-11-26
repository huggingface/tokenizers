//! A tokenizer spelled the way a SentencePiece `.model` file describes it must encode exactly like
//! `SentencePieceProcessor.encode`.
//!
//! The reference is SentencePiece 0.2.1 itself, not this tree: the vocabulary and merges below are
//! a tiny BPE model trained with `--model_type=bpe --normalization_rule_name=identity`, the merges
//! derived from the piece scores the way `convert_slow_tokenizer.build_tokenizer_from_spm_proto`
//! derives them, and every id literal is `SentencePieceProcessor.encode(text)` on that model.
//!
//! What is under test is the piece normalizer, set the way `Metaspace::from_sentencepiece` sets it
//! from SentencePiece's `add_dummy_prefix` and `remove_extra_whitespaces` flags.

use tk_encode::pipeline::{EncodeOptions, Tokenizer};

/// `b_111.model` / `b_110.model`, which differ only in `remove_extra_whitespaces`.
const VOCAB: &str = r#"{
    "<unk>": 0, "▁a": 1, "▁b": 2, "▁c": 3, "▁ab": 4, "▁bc": 5, "▁ca": 6, "▁aa": 7,
    "▁bb": 8, "▁cc": 9, "▁abc": 10, "▁bca": 11, "▁cab": 12, "aa": 13, "ab": 14,
    "bb": 15, "bc": 16, "ca": 17, "cc": 18, "▁": 19, "a": 20, "b": 21, "c": 22
}"#;

const MERGES: &str = r#"[
    ["▁", "a"], ["▁", "b"], ["▁", "c"], ["▁", "ab"], ["▁a", "b"], ["▁", "bc"],
    ["▁b", "c"], ["▁", "ca"], ["▁c", "a"], ["▁", "aa"], ["▁a", "a"], ["▁", "bb"],
    ["▁b", "b"], ["▁", "cc"], ["▁c", "c"], ["▁a", "bc"], ["▁ab", "c"], ["▁b", "ca"],
    ["▁bc", "a"], ["▁c", "ab"], ["▁ca", "b"], ["a", "a"], ["a", "b"], ["b", "b"],
    ["b", "c"], ["c", "a"], ["c", "c"]
]"#;

/// `<s>` is id 23, one past the vocabulary, so it only ever comes from the added-token path.
fn tokenizer(add_dummy_prefix: bool, remove_extra_whitespaces: bool) -> Tokenizer {
    let prepend = if add_dummy_prefix { "always" } else { "never" };
    let json = format!(
        r#"{{
            "version": "2.0",
            "normalizer": null,
            "piece_normalizer": {{"type": "Metaspace", "replacement": "▁", "prepend": "{prepend}",
                "whitespace_match": "only_space",
                "collapse_contiguous": {remove_extra_whitespaces}, "skip_prefix": false,
                "trim_trailing_delimiters": {remove_extra_whitespaces}}},
            "pre_tokenizer": {{"type": "Split", "pattern": {{"String": "▁"}},
                "behavior": "MergedWithNext", "invert": false}},
            "model": {{"type": "BPE", "unk_token": "<unk>", "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": false,
                "ignore_merges": false, "byte_level": false,
                "vocab": {VOCAB}, "merges": {MERGES}}},
            "added_tokens": [
                {{"id": 23, "content": "<s>", "single_word": false, "lstrip": false,
                  "rstrip": false, "normalized": false, "special": true}},
                {{"id": 24, "content": "</s>", "single_word": false, "lstrip": true,
                  "rstrip": true, "normalized": false, "special": true}},
                {{"id": 25, "content": "ac", "single_word": true, "lstrip": false,
                  "rstrip": false, "normalized": false, "special": true}}
            ],
            "post_processor": null,
            "decoder": null
        }}"#
    );
    tk_serialize::from_json(&json).expect("the canonical reader reads it")
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let encodings = tokenizer
        .encode(text, &EncodeOptions::default())
        .wait()
        .unwrap();
    encodings[0].ids().iter().map(|t| t.id()).collect()
}

/// `(add_dummy_prefix, remove_extra_whitespaces, text, SentencePiece's ids)`.
#[rustfmt::skip]
const CASES: &[(bool, bool, &str, &[u32])] = &[
    // A dummy prefix goes in front of a text that already opens with the delimiter, which is what
    // released `Metaspace` refused to do.
    (true, true,  "▁aa",       &[19, 7]),
    (true, true,  "▁",         &[]),
    // ... and the trailing strip takes the delimiter the text itself carried.
    (true, true,  "a▁",        &[1]),
    (true, true,  "aa bb cc",  &[7, 8, 9]),
    (true, true,  "ab bc",     &[4, 5]),
    (true, true,  "abc",       &[10]),
    (true, true,  "aa",        &[7]),
    (true, true,  "",          &[]),
    // Only spaces count: a tab and a newline survive, and have no piece of their own.
    (true, true,  "aa\tbb  cc", &[7, 0, 15, 9]),
    (true, true,  "aa\nbb",    &[7, 0, 15]),
    // `remove_extra_whitespaces`: the spaces around a piece go, and a run of them is one delimiter.
    (true, true,  " aa bb",    &[7, 8]),
    (true, true,  "   aa   ",  &[7]),
    (true, true,  "  ",        &[]),
    (true, true,  "a  b",      &[1, 2]),

    // The same model with `remove_extra_whitespaces` off: every space is a delimiter of its own.
    (true, false, " aa bb",    &[19, 7, 8]),
    (true, false, "   aa   ",  &[19, 19, 19, 7, 19, 19, 19]),
    (true, false, "  ",        &[19, 19, 19]),
    (true, false, "a  b",      &[1, 19, 2]),
    (true, false, "a▁",        &[1, 19]),
    (true, false, "▁",         &[19, 19]),
    (true, false, "aa\tbb  cc", &[7, 0, 15, 19, 9]),

    // An added token cuts the text into pieces, and each piece is normalized on its own: the
    // expected ids are SentencePiece on each chunk, not on the whole text.
    (true, true,  "aa<s>bb",       &[7, 23, 8]),
    (true, true,  "<s>",           &[23]),
    (true, true,  "<s>aa",         &[23, 7]),
    (true, true,  " aa<s> bb",     &[7, 23, 8]),
    // The prefix lands on the piece after the added token too, even though that piece already
    // opens with the delimiter.
    (true, true,  "▁aa<s>▁bb",     &[19, 7, 23, 19, 8]),

    // `lstrip`/`rstrip` swallow the spaces around the token before any piece is cut, so the ids
    // are SentencePiece on the trimmed chunks -- with `remove_extra_whitespaces` off, where the
    // strip is the only thing removing them.
    (true, false, "aa </s> bb",    &[7, 24, 8]),
    (true, true,  "aa </s> bb",    &[7, 24, 8]),
    // `single_word` sees the delimiter as a word boundary, and a letter as a letter.
    (true, true,  "ac </s>",       &[25, 24]),
    (true, true,  "cac </s>",      &[6, 22, 24]),
    // With `remove_extra_whitespaces` off, `lstrip` is the only thing taking that space: the
    // untrimmed chunk would have kept a delimiter of its own.
    (true, false, "cac </s>",      &[6, 22, 24]),
];

#[test]
fn encode_matches_sentencepiece() {
    let mut cache = [None, None];
    for (add_dummy_prefix, remove_extra_whitespaces, text, want) in CASES {
        let tokenizer = cache[usize::from(*remove_extra_whitespaces)]
            .get_or_insert_with(|| tokenizer(*add_dummy_prefix, *remove_extra_whitespaces));
        assert_eq!(
            encode(tokenizer, text),
            *want,
            "add_dummy_prefix={add_dummy_prefix} \
             remove_extra_whitespaces={remove_extra_whitespaces} {text:?}"
        );
    }
}
