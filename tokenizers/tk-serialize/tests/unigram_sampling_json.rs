#![cfg(all(feature = "unigram", feature = "serialize"))]

use std::collections::{BTreeMap, HashSet};
use tk_encode::models::unigram::Unigram;
use tk_encode::pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn tokenizer(alpha: Option<f64>, nbest_size: Option<usize>) -> PipelineTokenizer {
    let mut model = Unigram::from(
        vec![
            ("<unk>".into(), 0.0),
            ("a".into(), -1.0),
            ("b".into(), -1.0),
            ("ab".into(), -2.0),
        ],
        Some(0),
        false,
    )
    .unwrap();
    model.alpha = alpha;
    model.nbest_size = nbest_size;
    PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::Unigram(model),
        Default::default(),
        None,
        BTreeMap::new(),
        None,
        None,
    )
}

fn model(tokenizer: &PipelineTokenizer) -> &Unigram {
    let PipelineModel::Unigram(model) = tokenizer.get_model() else {
        panic!("expected a Unigram model");
    };
    model
}

fn round_trip(tokenizer: &PipelineTokenizer) -> PipelineTokenizer {
    tk_serialize::from_json(&tk_serialize::to_json(tokenizer).unwrap()).unwrap()
}

fn ids(tokenizer: &PipelineTokenizer) -> Vec<u32> {
    tokenizer
        .encode("ab", &EncodeOptions::no_specials())
        .wait()
        .unwrap()[0]
        .ids()
        .iter()
        .map(|token| token.id())
        .collect()
}

#[test]
fn all_optional_sampling_settings_round_trip() {
    for alpha in [None, Some(0.0), Some(0.25), Some(1.0)] {
        for nbest in [None, Some(0), Some(1), Some(8)] {
            let before = tokenizer(alpha, nbest);
            let after = round_trip(&before);
            assert_eq!(model(&after), model(&before));
        }
    }
}

#[test]
fn temperature_bits_survive_repeated_json_round_trips() {
    for alpha in [0.1, 0.12345678901234567, 1e-20, 0.9999999999999999] {
        let before = tokenizer(Some(alpha), Some(2));
        let after = round_trip(&round_trip(&before));
        assert_eq!(model(&after).alpha.unwrap().to_bits(), alpha.to_bits());
    }
}

#[test]
fn explicit_fields_are_read_and_written_as_numbers() {
    let mut config: serde_json::Value =
        serde_json::from_str(&tk_serialize::to_json(&tokenizer(None, None)).unwrap()).unwrap();
    config["model"]["alpha"] = serde_json::json!(0.5);
    config["model"]["nbest_size"] = serde_json::json!(12);
    let loaded = tk_serialize::from_json(&config.to_string()).unwrap();
    assert_eq!(model(&loaded).alpha, Some(0.5));
    assert_eq!(model(&loaded).nbest_size, Some(12));
    let written: serde_json::Value =
        serde_json::from_str(&tk_serialize::to_json(&loaded).unwrap()).unwrap();
    assert_eq!(written["model"]["alpha"], 0.5);
    assert_eq!(written["model"]["nbest_size"], 12);
}

#[test]
fn missing_and_null_options_keep_deterministic_defaults() {
    let original = tokenizer(None, None);
    let mut config: serde_json::Value =
        serde_json::from_str(&tk_serialize::to_json(&original).unwrap()).unwrap();
    assert!(config["model"].get("alpha").is_none());
    assert!(config["model"].get("nbest_size").is_none());
    let expected = ids(&original);
    for nulls in [false, true] {
        if nulls {
            config["model"]["alpha"] = serde_json::Value::Null;
            config["model"]["nbest_size"] = serde_json::Value::Null;
        }
        let loaded = tk_serialize::from_json(&config.to_string()).unwrap();
        assert_eq!(model(&loaded).alpha, None);
        assert_eq!(model(&loaded).nbest_size, None);
        for _ in 0..16 {
            assert_eq!(ids(&loaded), expected);
        }
    }
}

#[test]
fn explicit_zero_temperature_still_disables_sampling() {
    let before = tokenizer(Some(0.0), Some(0));
    let after = round_trip(&before);
    assert_eq!(model(&after).alpha, Some(0.0));
    assert_eq!(model(&after).nbest_size, Some(0));
    for _ in 0..16 {
        assert_eq!(ids(&after), ids(&before));
    }
}

#[test]
fn pipeline_keeps_sampling_after_serialization() {
    for nbest in [None, Some(0), Some(2)] {
        let before = tokenizer(Some(1.0), nbest);
        let after = round_trip(&before);
        let observed: HashSet<_> = (0..256).map(|_| ids(&after)).collect();
        // Both paths have score -2; a single path after all draws is vanishingly unlikely.
        assert_eq!(observed, HashSet::from([vec![1, 2], vec![3]]));
    }
}

#[test]
fn single_best_sampling_remains_restricted_to_one_path() {
    let before = tokenizer(Some(1.0), Some(1));
    let after = round_trip(&before);
    assert_eq!(model(&after).nbest_size, Some(1));
    let expected = ids(&before);
    for _ in 0..16 {
        assert_eq!(ids(&after), expected);
    }
}

#[test]
fn malformed_sampling_fields_are_rejected() {
    let original = tk_serialize::to_json(&tokenizer(None, None)).unwrap();
    for (key, value) in [
        ("alpha", serde_json::json!("warm")),
        ("alpha", serde_json::json!(true)),
        ("nbest_size", serde_json::json!(-1)),
        ("nbest_size", serde_json::json!(1.5)),
        ("nbest_size", serde_json::json!("2")),
    ] {
        let mut config: serde_json::Value = serde_json::from_str(&original).unwrap();
        config["model"][key] = value;
        let error = tk_serialize::from_json(&config.to_string())
            .err()
            .expect("invalid sampling configuration must be rejected");
        assert!(error.to_string().contains(key));
    }
}

#[test]
fn nonfinite_temperature_is_not_silently_dropped() {
    for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(tk_serialize::to_json(&tokenizer(Some(alpha), None)).is_err());
    }
}
