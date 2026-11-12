#![cfg(feature = "unigram")]

use std::collections::BTreeMap;
use tk_encode::models::unigram::Unigram;
use tk_encode::pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn model(nbest: Option<usize>, alpha: Option<f64>) -> Unigram {
    let vocab = [
        ("a", -1.0),
        ("b", -1.0),
        ("ab", -0.1),
        ("é", -1.0),
        ("🙂", -1.0),
    ]
    .into_iter()
    .map(|(piece, score)| (piece.to_owned(), score))
    .collect();
    let mut model = Unigram::from(vocab, None, false).unwrap();
    model.alpha = alpha;
    model.nbest_size = nbest;
    model
}

fn check_covered_input(nbest: Option<usize>, text: &str) {
    let model = model(nbest, Some(1.0));
    for _ in 0..32 {
        let tokens = model.tokenize(text).unwrap();
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.value.as_str())
                .collect::<String>(),
            text
        );
        let mut offset = 0;
        for token in tokens {
            assert_eq!(model.id_to_token(token.id).unwrap(), token.value);
            assert_eq!(token.offsets, (offset, offset + token.value.len()));
            offset += token.value.len();
        }
        assert_eq!(offset, text.len());
    }
}

#[test]
fn full_lattice_samples_known_ascii_without_an_unknown_id() {
    check_covered_input(None, "abab");
}

#[test]
fn full_lattice_samples_known_unicode_without_an_unknown_id() {
    check_covered_input(None, "éab🙂");
}

#[test]
fn nbest_samples_known_tokens_without_an_unknown_id() {
    check_covered_input(Some(8), "éabab🙂");
}

#[test]
fn single_best_sample_preserves_the_expected_segmentation() {
    assert_eq!(
        model(Some(1), Some(1.0)).encode("abab").unwrap(),
        ["ab", "ab"]
    );
}

#[test]
fn pipeline_sampling_without_an_unknown_id_emits_complete_input() {
    for nbest in [None, Some(1), Some(8)] {
        let tokenizer = PipelineTokenizer::from_parts(
            AddedVocabulary::new(),
            Vec::new(),
            PipelinePreTokenizer::None,
            PipelineModel::Unigram(model(nbest, Some(1.0))),
            Default::default(),
            None,
            BTreeMap::new(),
            None,
            None,
        );
        for _ in 0..16 {
            let encodings = tokenizer
                .encode("éabab🙂", &EncodeOptions::no_specials())
                .wait()
                .unwrap();
            let pieces: String = encodings[0]
                .ids()
                .iter()
                .map(|token| tokenizer.get_model().id_to_token(token.id()).unwrap())
                .collect();
            assert_eq!(pieces, "éabab🙂");
        }
    }
}

#[test]
fn sampling_after_deterministic_encoding_does_not_require_an_unknown_id() {
    let mut model = model(None, None);
    assert_eq!(model.encode("abab").unwrap(), ["ab", "ab"]);
    model.alpha = Some(1.0);
    assert_eq!(model.encode("abab").unwrap().concat(), "abab");
}

#[test]
fn deterministic_and_empty_input_controls_remain_unchanged() {
    for alpha in [None, Some(0.0), Some(1.0)] {
        let model = model(None, alpha);
        assert!(model.encode("").unwrap().is_empty());
        if alpha != Some(1.0) {
            assert_eq!(model.encode("abab").unwrap(), ["ab", "ab"]);
            assert!(model.encode("x").is_err());
        }
    }
}

#[test]
fn configured_unknown_tokens_still_fuse_adjacent_unknown_characters() {
    let mut model = Unigram::from(
        vec![("<unk>".into(), 0.0), ("a".into(), -1.0)],
        Some(0),
        false,
    )
    .unwrap();
    model.alpha = Some(1.0);
    for nbest in [None, Some(1), Some(8)] {
        model.nbest_size = nbest;
        let tokens = model.tokenize("xya").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(
            (tokens[0].id, tokens[0].value.as_str(), tokens[0].offsets),
            (0, "xy", (0, 2))
        );
        assert_eq!(
            (tokens[1].id, tokens[1].value.as_str(), tokens[1].offsets),
            (1, "a", (2, 3))
        );
    }
}
