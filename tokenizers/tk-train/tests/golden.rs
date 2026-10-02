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
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
use tk_train::{TokenizerBlueprint, TokenizerTrainer, TokenizerTrainerBuilder, TrainerWrapper};

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

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .split('\n')
        .map(String::from)
        .collect()
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let raw = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&tk_convert::canonicalize_str(&raw).unwrap()).unwrap()
}

fn golden(case: &str) -> Value {
    read_json(Path::new(GOLDEN).join(case).join("tokenizer.json"))
}

/// The golden's trainer state, with its special tokens and unknown token taken out.
///
/// 0.23.2 kept both on the trainer. v1 keeps them on the builder, which refuses a trainer that
/// still has them.
fn golden_trainer_state(case: &str) -> (Value, Vec<String>, Option<String>) {
    let path = Path::new(GOLDEN).join(case).join("trainer.json");
    let mut state: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let specials =
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
    let unk =
        take(&mut state, "unk_token", Value::Null).and_then(|unk| unk.as_str().map(String::from));
    (state, specials, unk)
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

/// The blueprint the golden was trained with: its stages, without the tokens training added.
fn golden_blueprint(mut tokenizer: Value) -> TokenizerBlueprint {
    tokenizer["added_tokens"] = Value::Array(vec![]);
    let tokenizer = tk_serialize::from_json(&tokenizer.to_string()).unwrap();
    TokenizerBlueprint::from(&tokenizer)
}

/// A builder configured as 0.23.2 was for `case`, from its trainer state and the golden's stages.
fn golden_builder(case: &str, golden: &Value) -> TokenizerTrainerBuilder {
    let (state, specials, unk) = golden_trainer_state(case);
    let trainer: TrainerWrapper = serde_json::from_value(state).unwrap();
    let builder = TokenizerTrainer::builder(trainer)
        .blueprint(golden_blueprint(golden.clone()))
        .special_tokens(specials);
    match unk {
        Some(unk) => builder.unk_token(unk),
        None => builder,
    }
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
    let trained = to_value(&trainer.train_files(&corpus()).unwrap());

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
    let trained = to_value(&trainer.train_files(&corpus()).unwrap());

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
    let trained = to_value(&trainer.train_files(&corpus()).unwrap());

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
    let trained = to_value(&trainer.train_files(&corpus()).unwrap());

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
    let trainer = golden_builder("wordlevel_added_tokens", &golden)
        .special_tokens(["<|im_start|>", "<|im_end|>"])
        .added_tokens(vec![
            AddedToken::from("HuggingFace", false).normalized(true),
            AddedToken::from("Tokenizers", false).normalized(true),
        ])
        .build()
        .unwrap();
    let trained = to_value(&trainer.train_files(&corpus()).unwrap());

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
    let (state, specials, _) = golden_trainer_state("retrain_gpt2");
    let trainer: TrainerWrapper = serde_json::from_value(state).unwrap();
    let trainer = TokenizerTrainer::builder(trainer)
        .blueprint(TokenizerBlueprint::from(&gpt2))
        .special_tokens(specials)
        .build()
        .unwrap();
    let sequences = corpus()
        .iter()
        .flat_map(|path| lines(path))
        .collect::<Vec<_>>();
    let trained = to_value(&trainer.train(sequences.iter()).unwrap());

    assert_eq!(vocab(&trained), vocab(&golden));
    assert_eq!(merges(&trained), merges(&golden));
    assert_added_tokens_point_at_themselves(&trained);
}

#[cfg(feature = "parity-aware-bpe")]
#[test]
#[ignore = "trains on ~100 MB: make train-golden"]
fn parity_bpe_matches_release_exactly() {
    const DEV_LINES: usize = 500;
    let golden = golden("parity_bpe");
    let (mut state, specials, _) = golden_trainer_state("parity_bpe");
    // 0.23.2's Python state for this trainer is hand-written, not serde: it spells the variant in
    // lowercase.
    let variant = state["variant"].as_str().unwrap();
    state["variant"] = Value::String(variant[..1].to_uppercase() + &variant[1..]);
    let trainer: tk_train::ParityBpeTrainer = serde_json::from_value(state).unwrap();
    let trainer = TokenizerTrainer::builder(trainer)
        .blueprint(golden_blueprint(golden.clone()))
        .special_tokens(specials)
        .build()
        .unwrap();

    let languages: Vec<Vec<String>> = corpus()
        .iter()
        .filter(|path| path.parent().unwrap().ends_with("lang"))
        .map(|path| lines(path))
        .collect();
    let train = languages
        .iter()
        .map(|lines| lines[..lines.len() - DEV_LINES].iter())
        .collect();
    let dev = languages
        .iter()
        .map(|lines| lines[lines.len() - DEV_LINES..].iter())
        .collect();
    let trained = to_value(&trainer.train_languages(train, Some(dev)).unwrap());

    assert_eq!(vocab(&trained), vocab(&golden));
    assert_eq!(merges(&trained), merges(&golden));
}
