//! Trains with the tk-train API and compares the result against `tests/golden`, which released
//! `tokenizers` 0.23.2 trained with the same pipelines, trainers and corpus (`golden/generate.py`).
//!
//! Each test trains on about 100 MB, so they are ignored by default:
//!
//!   make train-golden
//!
//! Where 0.23.2 gives one answer, the test demands it exactly. Where it does not (WordPiece and
//! Unigram give a different tokenizer on every run), the test demands what every 0.23.2 run agreed
//! on. Where 0.23.2 is wrong, the test demands the correct behaviour instead and says so.

#![cfg(all(
    feature = "bpe",
    feature = "unigram",
    feature = "wordpiece",
    feature = "wordlevel"
))]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;
use tk_encode::pipeline::PipelineTokenizer;
use tk_train::{ProgressFormat, TokenizerTrainerBuilder, TrainerWrapper};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");

/// Every real-text corpus in `data/`, in the order `generate.py` reads them.
fn corpus() -> Vec<PathBuf> {
    let mut files = sorted_files("fixtures/lang", |_| true);
    files.extend(sorted_files("fixtures/modalities", |name| {
        !name.starts_with("added_")
    }));
    files.extend(
        ["big.txt", "unigram_wagahaiwa_nekodearu.txt"].map(|name| Path::new(DATA).join(name)),
    );
    files
}

fn sorted_files(dir: &str, keep: fn(&str) -> bool) -> Vec<PathBuf> {
    let dir = Path::new(DATA).join(dir);
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| keep(name))
        .collect();
    names.sort();
    names.into_iter().map(|name| dir.join(name)).collect()
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let raw = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&tk_convert::canonicalize_str(&raw).unwrap()).unwrap()
}

fn golden(case: &str) -> Value {
    read_json(Path::new(GOLDEN).join(case).join("tokenizer.json"))
}

fn trainer_state(case: &str) -> Value {
    let path = Path::new(GOLDEN).join(case).join("trainer.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// What 0.23.2 kept on the trainer and v1 keeps on the builder. Progress is left out: the goldens
/// were made without it, and it does not change what is trained.
struct BuilderSettings {
    vocab_size: Option<usize>,
    special_tokens: Vec<String>,
    unk_token: Option<String>,
}

/// The golden's trainer state, with the settings v1 keeps on the builder taken out of it.
fn golden_trainer_state(case: &str) -> (Value, BuilderSettings) {
    let mut state = trainer_state(case);
    let special_tokens =
        take(&mut state, "special_tokens", Value::Array(vec![])).map_or(vec![], |tokens| {
            let tokens = tokens.as_array().unwrap().iter();
            tokens
                .map(|token| {
                    token
                        .get("content")
                        .unwrap_or(token)
                        .as_str()
                        .unwrap()
                        .to_string()
                })
                .collect()
        });
    let unk_token =
        take(&mut state, "unk_token", Value::Null).and_then(|unk| unk.as_str().map(String::from));
    let vocab_size = take(&mut state, "vocab_size", Value::Null)
        .and_then(|size| size.as_u64())
        .map(|size| size as usize);
    let settings = BuilderSettings {
        vocab_size,
        special_tokens,
        unk_token,
    };
    (state, settings)
}

/// Replaces the first `key` found in `value`, at any depth, and returns what it held.
fn take(value: &mut Value, key: &str, with: Value) -> Option<Value> {
    let object = value.as_object_mut()?;
    if let Some(found) = object.get_mut(key) {
        return Some(std::mem::replace(found, with));
    }
    object
        .values_mut()
        .find_map(|inner| take(inner, key, with.clone()))
}

/// The stages the golden was trained with, without the tokens training added.
fn golden_stages(mut tokenizer: Value) -> PipelineTokenizer {
    tokenizer["added_tokens"] = Value::Array(vec![]);
    tk_serialize::from_json(&tokenizer.to_string()).unwrap()
}

/// `builder` with the trainer 0.23.2 had for `case`, and the settings that trainer held.
fn with_golden_trainer(
    case: &str,
    builder: TokenizerTrainerBuilder<TrainerWrapper>,
) -> TokenizerTrainerBuilder<TrainerWrapper> {
    let (state, settings) = golden_trainer_state(case);
    let trainer: TrainerWrapper = serde_json::from_value(state).unwrap();
    let mut builder = builder
        .trainer(trainer)
        .special_tokens(settings.special_tokens)
        .progress(ProgressFormat::Indicatif);
    if let Some(vocab_size) = settings.vocab_size {
        builder = builder.vocab_size(vocab_size);
    }
    if let Some(unk_token) = settings.unk_token {
        builder = builder.unk_token(unk_token);
    }
    builder
}

/// A builder configured as 0.23.2 was for `case`, from its trainer state and the golden's stages.
fn golden_builder(case: &str, golden: &Value) -> TokenizerTrainerBuilder<TrainerWrapper> {
    let stages = TokenizerTrainerBuilder::from_tokenizer(&golden_stages(golden.clone())).unwrap();
    with_golden_trainer(case, stages)
}

fn to_value(tokenizer: &PipelineTokenizer) -> Value {
    serde_json::from_str(&tk_serialize::to_json(tokenizer).unwrap()).unwrap()
}

/// `token -> id` for a BPE, WordPiece or WordLevel model.
fn vocab(tokenizer: &Value) -> HashMap<String, u64> {
    let vocab = tokenizer["model"]["vocab"].as_object().unwrap();
    vocab
        .iter()
        .map(|(token, id)| (token.clone(), id.as_u64().unwrap()))
        .collect()
}

/// `piece -> score` for a Unigram model.
fn pieces(tokenizer: &Value) -> HashMap<String, f64> {
    let vocab = tokenizer["model"]["vocab"].as_array().unwrap();
    vocab
        .iter()
        .map(|entry| {
            (
                entry[0].as_str().unwrap().to_string(),
                entry[1].as_f64().unwrap(),
            )
        })
        .collect()
}

fn merges(tokenizer: &Value) -> Vec<(String, String)> {
    let merges = tokenizer["model"]["merges"].as_array().unwrap();
    merges
        .iter()
        .map(|merge| match merge {
            Value::String(pair) => {
                let (a, b) = pair.split_once(' ').unwrap();
                (a.to_string(), b.to_string())
            }
            pair => (
                pair[0].as_str().unwrap().to_string(),
                pair[1].as_str().unwrap().to_string(),
            ),
        })
        .collect()
}

fn added_tokens(tokenizer: &Value) -> HashMap<String, u64> {
    let tokens = tokenizer["added_tokens"].as_array().unwrap();
    tokens
        .iter()
        .map(|t| {
            (
                t["content"].as_str().unwrap().to_string(),
                t["id"].as_u64().unwrap(),
            )
        })
        .collect()
}

/// 0.23.2 lets an added token keep the id it had before training, which can belong to another
/// token of the trained vocab or sit past it. Each added token must instead carry its own id.
fn assert_added_tokens_point_at_themselves(tokenizer: &Value) {
    let vocab = vocab(tokenizer);
    let taken: HashSet<u64> = vocab.values().copied().collect();
    for (content, id) in added_tokens(tokenizer) {
        match vocab.get(&content) {
            Some(&model_id) => assert_eq!(id, model_id, "{content:?} is {model_id} in the vocab"),
            None => assert!(
                !taken.contains(&id),
                "{content:?} reuses the vocab's id {id}"
            ),
        }
    }
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn bpe_byte_level_matches_release_exactly() {
    let golden = golden("bpe_byte_level");
    let trainer = golden_builder("bpe_byte_level", &golden).build().unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    assert_eq!(vocab(&trained), vocab(&golden));
    assert_eq!(merges(&trained), merges(&golden));
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn bpe_sentencepiece_adds_byte_tokens_and_keeps_release_merges() {
    let mut golden = golden("bpe_sentencepiece");
    // 0.23.2 never adds the <0xNN> byte tokens byte fallback needs (huggingface/tokenizers#1407),
    // and v1 refuses to load a byte-fallback model without them.
    golden["model"]["byte_fallback"] = Value::Bool(false);
    let trainer = golden_builder("bpe_sentencepiece", &golden)
        .build()
        .unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    let vocab = vocab(&trained);
    for byte in 0..=255u8 {
        assert!(
            vocab.contains_key(&format!("<0x{byte:02X}>")),
            "missing <0x{byte:02X}>"
        );
    }
    // The byte tokens never occur in a word, so they only take vocab slots away from the last merges.
    let (trained, golden) = (merges(&trained), merges(&golden));
    assert_eq!(trained.len(), golden.len() - 256);
    assert_eq!(trained[..], golden[..trained.len()]);
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn wordpiece_bert_shares_release_vocab() {
    let golden = golden("wordpiece_bert");
    let trainer = golden_builder("wordpiece_bert", &golden).build().unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    // Two 0.23.2 runs share 99.99% of this vocab.
    let (trained_vocab, golden_vocab) = (vocab(&trained), vocab(&golden));
    let shared = trained_vocab
        .keys()
        .filter(|token| golden_vocab.contains_key(*token))
        .count();
    assert_eq!(trained_vocab.len(), golden_vocab.len());
    assert!(
        shared as f64 / golden_vocab.len() as f64 >= 0.999,
        "{shared} of {} shared",
        golden_vocab.len()
    );
    assert_added_tokens_point_at_themselves(&trained);
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn unigram_sentencepiece_keeps_release_pieces() {
    let golden = golden("unigram_sentencepiece");
    let trainer = golden_builder("unigram_sentencepiece", &golden)
        .build()
        .unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    // Every 0.23.2 run gives this piece set, with scores up to 0.0062 apart.
    let (trained, golden) = (pieces(&trained), pieces(&golden));
    assert_eq!(
        trained.keys().collect::<HashSet<_>>(),
        golden.keys().collect::<HashSet<_>>()
    );
    for (piece, score) in &golden {
        assert!(
            (trained[piece] - score).abs() <= 0.01,
            "{piece:?}: {} vs {score}",
            trained[piece]
        );
    }
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn wordlevel_keeps_tokens_added_before_training() {
    let golden = golden("wordlevel_added_tokens");
    // Fails until the builder takes added tokens that are not special: `HuggingFace` and
    // `Tokenizers` were added as normalized, non-special tokens.
    let trainer = golden_builder("wordlevel_added_tokens", &golden)
        .special_tokens(["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>"])
        .build()
        .unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    assert_eq!(vocab(&trained), vocab(&golden));
    // 0.23.2 drops the tokens whose ids the trainer's special tokens take.
    let added = added_tokens(&trained);
    for content in ["<|im_start|>", "<|im_end|>", "HuggingFace", "Tokenizers"] {
        assert!(added.contains_key(content), "{content:?} is gone");
    }
    assert_added_tokens_point_at_themselves(&trained);
}

#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn retrain_gpt2_matches_release_model() {
    let golden = golden("retrain_gpt2");
    let gpt2 =
        tk_serialize::from_json(&read_json(Path::new(DATA).join("gpt2.json")).to_string()).unwrap();
    let gpt2 = TokenizerTrainerBuilder::from_tokenizer(&gpt2).unwrap();
    let trainer = with_golden_trainer("retrain_gpt2", gpt2).build().unwrap();
    let trained = to_value(&trainer.train_files(corpus()).unwrap());

    assert_eq!(vocab(&trained), vocab(&golden));
    assert_eq!(merges(&trained), merges(&golden));
    assert_added_tokens_point_at_themselves(&trained);
}

/// Trains on the `fixtures/lang` corpora, with the last 500 lines of each as its dev set, and
/// expects 0.23.2's vocab and merges exactly.
#[cfg(feature = "parity-aware-bpe")]
#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn parity_bpe_matches_release_exactly() {
    let mut state = trainer_state("parity_bpe");
    // 0.23.2's Python state for this trainer is hand-written, not serde: it spells the variant in
    // lowercase.
    let variant = state["variant"].as_str().unwrap();
    state["variant"] = Value::String(variant[..1].to_uppercase() + &variant[1..]);
    let _trainer: tk_train::ParityBpeTrainer = serde_json::from_value(state).unwrap();

    todo!("ParityBpe has no way to train through TokenizerTrainer yet (decision A6)");
}
