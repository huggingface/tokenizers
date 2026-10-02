//! Training must give the same `tokenizer.json` as the latest *released* `tokenizers`.
//!
//! Each test starts both libraries from the same untrained pipeline, gives each trainer the same
//! arguments, trains both on the same files, and compares the two `tokenizer.json` files. Both files
//! are read and written back by `tk_serialize` first, so a setting spelled two ways compares equal.
//!
//! Released WordPiece and Unigram training is not reproducible run to run. Those tests demand the
//! rest of the file exactly and the vocab only as closely as two release runs agree.
//!
//! The corpus is two files, about 7.5 MB:
//!
//!   make train-oracle

#![cfg(feature = "oracle")]

use std::collections::{BTreeSet, HashSet};

use serde_json::Value;
use tk_train::{
    BpeTrainer, ProgressFormat, TokenizerTrainerBuilder, TrainerWrapper, UnigramTrainer,
    WordLevelTrainer, WordPieceTrainer,
};
use tokenizers_release as release;
use tokenizers_release::models::TrainerWrapper as ReleaseTrainer;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

fn corpus() -> Vec<String> {
    ["big.txt", "unigram_wagahaiwa_nekodearu.txt"]
        .map(|name| format!("{DATA}/{name}"))
        .to_vec()
}

fn canonical(json: &str) -> Value {
    let tokenizer = tk_serialize::from_json(&tk_convert::canonicalize_str(json).unwrap()).unwrap();
    serde_json::from_str(&tk_serialize::to_json(&tokenizer).unwrap()).unwrap()
}

/// An untrained `tokenizer.json`. `model` still holds a token or a few, because v1 cannot load a
/// model with an empty vocab. Training replaces that vocab on both sides.
fn untrained(normalizer: Value, pre_tokenizer: Value, decoder: Value, model: Value) -> String {
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": decoder,
        "model": model,
    })
    .to_string()
}

fn train_release(pipeline: &str, trainer: impl Into<ReleaseTrainer>) -> Value {
    let mut tokenizer: release::Tokenizer = pipeline.parse().unwrap();
    tokenizer
        .train_from_files(&mut trainer.into(), corpus())
        .unwrap();
    canonical(&tokenizer.to_string(false).unwrap())
}

fn train_v1(
    pipeline: &str,
    configure: impl FnOnce(
        TokenizerTrainerBuilder<TrainerWrapper>,
    ) -> TokenizerTrainerBuilder<TrainerWrapper>,
) -> Value {
    let stages = tk_serialize::from_json(&tk_convert::canonicalize_str(pipeline).unwrap()).unwrap();
    let builder = TokenizerTrainerBuilder::from_tokenizer(&stages)
        .unwrap()
        .progress(ProgressFormat::Silent);
    let trained = configure(builder)
        .build()
        .unwrap()
        .train_files(corpus())
        .unwrap();
    canonical(&tk_serialize::to_json(&trained).unwrap())
}

fn release_specials(tokens: &[&str]) -> Vec<release::AddedToken> {
    tokens
        .iter()
        .map(|token| release::AddedToken::from(*token, true))
        .collect()
}

fn assert_same_tokenizer(v1: &Value, release: &Value) {
    assert_same_outside_vocab(v1, release);
    assert_same_entries(
        &v1["model"]["vocab"],
        &release["model"]["vocab"],
        "model.vocab",
    );
}

/// Compares key by key, so a failure names the part of the file that differs.
fn assert_same_outside_vocab(v1: &Value, release: &Value) {
    for key in keys(v1, release) {
        if key != "model" {
            assert_eq!(v1[&key], release[&key], "{key} differs");
        }
    }
    let (v1, release) = (&v1["model"], &release["model"]);
    for key in keys(v1, release) {
        match key.as_str() {
            "vocab" => {}
            "merges" => assert_same_entries(&v1[&key], &release[&key], "model.merges"),
            _ => assert_eq!(v1[&key], release[&key], "model.{key} differs"),
        }
    }
}

fn keys(a: &Value, b: &Value) -> BTreeSet<String> {
    let a = a.as_object().unwrap().keys();
    a.chain(b.as_object().unwrap().keys()).cloned().collect()
}

/// Names the first entry that differs: printing two 8000-entry vocabs buries it.
fn assert_same_entries(v1: &Value, release: &Value, what: &str) {
    let (v1, release) = (entries(v1), entries(release));
    if let Some(i) = (0..v1.len().min(release.len())).find(|&i| v1[i] != release[i]) {
        panic!(
            "{what} differs first at entry {i}: v1 {} vs release {}",
            v1[i], release[i]
        );
    }
    assert_eq!(v1.len(), release.len(), "{what} lengths differ");
}

/// A vocab map as `[token, id]` pairs, or a list as it is.
fn entries(value: &Value) -> Vec<Value> {
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(token, id)| serde_json::json!([token, id]))
            .collect(),
        Value::Array(entries) => entries.clone(),
        other => panic!("not a vocab: {other}"),
    }
}

/// The tokens of a vocab, without their ids or scores.
fn tokens(vocab: &Value) -> HashSet<String> {
    let tokens = entries(vocab).into_iter();
    tokens
        .map(|entry| entry[0].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn byte_level_bpe() {
    let alphabet = release::pre_tokenizers::byte_level::ByteLevel::alphabet();
    // v1 refuses a byte-level model that cannot spell every byte.
    let bytes: serde_json::Map<String, Value> = alphabet
        .iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), id.into()))
        .collect();
    let pipeline = untrained(
        Value::Null,
        serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
        serde_json::json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
        serde_json::json!({"type": "BPE", "vocab": bytes, "merges": []}),
    );
    let specials = ["<|endoftext|>"];

    let release = train_release(
        &pipeline,
        release::models::bpe::BpeTrainer::builder()
            .vocab_size(8000)
            .min_frequency(2)
            .special_tokens(release_specials(&specials))
            .initial_alphabet(alphabet.iter().copied().collect())
            .show_progress(false)
            .build(),
    );
    let v1 = train_v1(&pipeline, |builder| {
        builder
            .trainer(TrainerWrapper::BpeTrainer(
                BpeTrainer::builder()
                    .min_frequency(2)
                    .initial_alphabet(alphabet)
                    .build(),
            ))
            .vocab_size(8000)
            .special_tokens(specials)
    });

    assert_same_tokenizer(&v1, &release);
}

#[test]
fn metaspace_bpe() {
    let metaspace = serde_json::json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    let pipeline = untrained(
        Value::Null,
        metaspace.clone(),
        metaspace,
        serde_json::json!({"type": "BPE", "unk_token": "<unk>", "fuse_unk": true, "vocab": {"<unk>": 0}, "merges": []}),
    );
    let specials = ["<unk>", "<s>", "</s>"];

    let release = train_release(
        &pipeline,
        release::models::bpe::BpeTrainer::builder()
            .vocab_size(8000)
            .special_tokens(release_specials(&specials))
            .max_token_length(Some(16))
            .show_progress(false)
            .build(),
    );
    let v1 = train_v1(&pipeline, |builder| {
        builder
            .trainer(TrainerWrapper::BpeTrainer(
                BpeTrainer::builder()
                    .max_token_length(Some(16))
                    .fuse_unk(true)
                    .build(),
            ))
            .vocab_size(8000)
            .special_tokens(specials)
    });

    assert_same_tokenizer(&v1, &release);
}

#[test]
fn bert_wordpiece() {
    let pipeline = untrained(
        serde_json::json!({"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}),
        serde_json::json!({"type": "BertPreTokenizer"}),
        serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
        serde_json::json!({"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0}}),
    );
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

    let release = train_release(
        &pipeline,
        release::models::wordpiece::WordPieceTrainer::builder()
            .vocab_size(8000)
            .special_tokens(release_specials(&specials))
            .show_progress(false)
            .build(),
    );
    let v1 = train_v1(&pipeline, |builder| {
        builder
            .trainer(TrainerWrapper::WordPieceTrainer(
                WordPieceTrainer::builder().build(),
            ))
            .vocab_size(8000)
            .special_tokens(specials)
    });

    assert_same_outside_vocab(&v1, &release);
    // Two release runs share 7999 or 8000 of these 8000 tokens.
    let (v1, release) = (
        tokens(&v1["model"]["vocab"]),
        tokens(&release["model"]["vocab"]),
    );
    assert_eq!(v1.len(), release.len());
    let shared = v1.intersection(&release).count();
    assert!(
        shared * 1000 >= release.len() * 999,
        "{shared} of {} shared",
        release.len()
    );
}

#[test]
fn sentencepiece_unigram() {
    let metaspace = serde_json::json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    let pipeline = untrained(
        serde_json::json!({"type": "Sequence", "normalizers": [{"type": "Nmt"}, {"type": "NFKC"}]}),
        metaspace.clone(),
        metaspace,
        serde_json::json!({"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0]]}),
    );
    let specials = ["<pad>", "</s>", "<unk>"];

    let release = train_release(
        &pipeline,
        release::models::unigram::UnigramTrainer::builder()
            .vocab_size(8000)
            .unk_token(Some("<unk>".into()))
            .special_tokens(release_specials(&specials))
            .show_progress(false)
            .build()
            .unwrap(),
    );
    let v1 = train_v1(&pipeline, |builder| {
        builder
            .trainer(TrainerWrapper::UnigramTrainer(
                UnigramTrainer::builder().build(),
            ))
            .vocab_size(8000)
            .unk_token("<unk>")
            .special_tokens(specials)
    });

    assert_same_outside_vocab(&v1, &release);
    assert_eq!(
        tokens(&v1["model"]["vocab"]),
        tokens(&release["model"]["vocab"])
    );
}

#[test]
fn whitespace_wordlevel() {
    let pipeline = untrained(
        serde_json::json!({"type": "Lowercase"}),
        serde_json::json!({"type": "Whitespace"}),
        Value::Null,
        serde_json::json!({"type": "WordLevel", "unk_token": "[UNK]", "vocab": {"[UNK]": 0}}),
    );
    let specials = ["[UNK]", "[PAD]"];

    let release = train_release(
        &pipeline,
        release::models::wordlevel::WordLevelTrainer::builder()
            .vocab_size(8000)
            .min_frequency(2)
            .special_tokens(release_specials(&specials))
            .show_progress(false)
            .build()
            .unwrap(),
    );
    let v1 = train_v1(&pipeline, |builder| {
        builder
            .trainer(TrainerWrapper::WordLevelTrainer(
                WordLevelTrainer::builder().min_frequency(2).build(),
            ))
            .vocab_size(8000)
            .special_tokens(specials)
    });

    assert_same_tokenizer(&v1, &release);
}
