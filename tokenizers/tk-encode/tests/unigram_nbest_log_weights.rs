#![cfg(feature = "unigram")]

use tk_encode::models::unigram::Unigram;
use tk_encode::pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn model(prefix_score: f64, a_score: f64, aa_score: f64, alpha: f64, nbest: usize) -> Unigram {
    let mut tokenizer = Unigram::from(
        vec![
            ("<unk>".to_owned(), -100.0),
            ("z".to_owned(), prefix_score),
            ("a".to_owned(), a_score),
            ("aa".to_owned(), aa_score),
        ],
        Some(0),
        false,
    )
    .unwrap();
    tokenizer.alpha = Some(alpha);
    tokenizer.nbest_size = Some(nbest);
    tokenizer
}

fn assert_merged_frequency(tokenizer: &Unigram, expected: f64) {
    let draws = 10_000;
    let mut merged = 0;
    for _ in 0..draws {
        let pieces = tokenizer.encode("zaa").unwrap();
        if pieces == ["z", "aa"] {
            merged += 1;
        } else {
            assert_eq!(pieces, ["z", "a", "a"]);
        }
    }
    let frequency = merged as f64 / draws as f64;
    assert!(
        (frequency - expected).abs() < 0.03,
        "expected {expected}, observed {frequency}"
    );
}

#[test]
fn nbest_equal_path_scores_sample_both_paths_after_common_offset() {
    for prefix_score in [0.0, -2000.0, -10_000.0] {
        for alpha in [0.5, 1.0] {
            // The common prefix cannot change the relative probability of
            // [z, aa] and [z, a, a], whose total scores are equal.
            assert_merged_frequency(&model(prefix_score, -1.0, -2.0, alpha, 2), 0.5);
        }
    }
}

#[test]
fn nbest_unequal_path_scores_keep_their_relative_weights() {
    for prefix_score in [0.0, -2000.0] {
        for alpha in [0.5, 1.0] {
            // Before the common prefix, [aa] scores -1 and [a, a] scores -4.
            let expected = 1.0 / (1.0 + f64::exp(-3.0 * alpha));
            assert_merged_frequency(&model(prefix_score, -2.0, -1.0, alpha, 2), expected);
        }
    }
}

#[test]
fn pipeline_nbest_sampling_does_not_collapse_after_a_low_probability_prefix() {
    let tokenizer = PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::Unigram(model(-2000.0, -1.0, -2.0, 0.5, 2)),
        Default::default(),
        None,
        Default::default(),
        None,
        None,
    );
    let mut seen_merged = false;
    let mut seen_split = false;
    for _ in 0..64 {
        let encodings = tokenizer
            .encode("zaa", &EncodeOptions::default())
            .wait()
            .unwrap();
        match encodings[0].ids() {
            ids if ids == [1, 3] => seen_merged = true,
            ids if ids == [1, 2, 2] => seen_split = true,
            ids => panic!("unexpected segmentation: {ids:?}"),
        }
    }
    assert!(seen_merged && seen_split);
}

#[test]
fn single_best_and_disabled_sampling_are_unchanged() {
    for (alpha, nbest) in [(0.5, 1), (1.0, 1), (0.0, 2)] {
        let tokenizer = model(-2000.0, -2.0, -1.0, alpha, nbest);
        for _ in 0..3 {
            assert_eq!(tokenizer.encode("zaa").unwrap(), ["z", "aa"]);
            assert!(tokenizer.encode("").unwrap().is_empty());
        }
    }
}
