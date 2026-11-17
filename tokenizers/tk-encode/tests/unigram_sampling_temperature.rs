#![cfg(feature = "unigram")]

use std::collections::HashMap;

use tk_encode::models::unigram::Unigram;
use tk_encode::pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn model(alpha: Option<f64>, score_per_character: f64, max_piece: usize) -> Unigram {
    let mut vocab = vec![("<unk>".to_owned(), -100.0)];
    vocab.extend((1..=max_piece).map(|n| ("a".repeat(n), score_per_character * n as f64)));
    let mut model = Unigram::from(vocab, Some(0), false).unwrap();
    model.alpha = alpha;
    model
}

fn assert_uniform_segmentations(alpha: f64) {
    // Every segmentation of four characters has score -40, so all eight paths
    // have equal probability for any temperature. Enumerate them independently
    // of the lattice's forward/backward recursion.
    let paths = [
        vec!["aaaa"],
        vec!["a", "aaa"],
        vec!["aa", "aa"],
        vec!["aaa", "a"],
        vec!["a", "a", "aa"],
        vec!["a", "aa", "a"],
        vec!["aa", "a", "a"],
        vec!["a", "a", "a", "a"],
    ];
    let mut counts: HashMap<Vec<String>, usize> = paths
        .into_iter()
        .map(|path| (path.into_iter().map(str::to_owned).collect(), 0))
        .collect();
    let tokenizer = model(Some(alpha), -10.0, 4);
    let draws = 10_000;
    for _ in 0..draws {
        let pieces = tokenizer.encode("aaaa").unwrap();
        *counts.get_mut(&pieces).expect("a valid segmentation") += 1;
    }
    for (path, count) in counts {
        let frequency = count as f64 / draws as f64;
        assert!(
            (frequency - 0.125).abs() < 0.03,
            "alpha={alpha}, path={path:?}, frequency={frequency}"
        );
    }
}

#[test]
fn sampling_low_temperature_preserves_equal_path_scores() {
    assert_uniform_segmentations(0.1);
}

#[test]
fn sampling_half_temperature_preserves_equal_path_scores() {
    assert_uniform_segmentations(0.5);
}

#[test]
fn sampling_unit_temperature_preserves_equal_path_scores() {
    assert_uniform_segmentations(1.0);
}

#[test]
fn sampling_a_single_low_probability_path_does_not_underflow() {
    for alpha in [0.1, 0.5, 1.0] {
        let tokenizer = model(Some(alpha), -1000.0, 1);
        let text = "a".repeat(32);
        assert_eq!(tokenizer.encode(&text).unwrap(), vec!["a"; 32]);
    }
}

#[test]
fn deterministic_and_empty_encoding_are_unchanged() {
    for alpha in [None, Some(0.0), Some(0.5), Some(1.0)] {
        let tokenizer = model(alpha, -10.0, 1);
        assert_eq!(tokenizer.encode("aaaa").unwrap(), vec!["a"; 4]);
        assert!(tokenizer.encode("").unwrap().is_empty());
    }
}

#[test]
fn pipeline_sampling_keeps_low_probability_words_across_calls() {
    let tokenizer = PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::Unigram(model(Some(0.5), -1000.0, 1)),
        Default::default(),
        None,
        Default::default(),
        None,
        None,
    );
    let text = "a".repeat(32);
    for _ in 0..3 {
        let encodings = tokenizer
            .encode(vec![text.as_str(), "aa"], &EncodeOptions::default())
            .wait()
            .unwrap();
        assert_eq!(encodings[0].ids(), vec![1; 32]);
        assert_eq!(encodings[1].ids(), [1, 1]);
    }
}
