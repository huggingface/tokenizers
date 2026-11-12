use super::*;
use std::collections::BTreeMap;

fn trainer() -> UnigramTrainer {
    UnigramTrainer::builder()
        .vocab_size(3)
        .max_piece_length(1)
        .n_sub_iterations(1)
        .unk_token(Some("<unk>".into()))
        .show_progress(false)
        .build()
        .unwrap()
}

fn seed_scores(sentences: &[Sentence]) -> BTreeMap<String, f64> {
    trainer()
        .make_seed_sentence_pieces(sentences, &None)
        .into_iter()
        .collect()
}

#[test]
fn character_counts_above_u32_keep_weighted_seed_probabilities() {
    for (repeated, other) in [("aa", "b"), ("éé", "字")] {
        let sentences = vec![
            (repeated.into(), 3_000_000_000),
            (other.into(), 2_000_000_000),
        ];
        let scores = seed_scores(&sentences);
        let first = repeated.chars().next().unwrap().to_string();
        assert_eq!(scores.len(), 2);
        assert!((scores[&first] - 0.75_f64.ln()).abs() < 1e-12);
        assert!((scores[other] - 0.25_f64.ln()).abs() < 1e-12);
    }
}

#[test]
fn small_character_counts_keep_weighted_seed_probabilities() {
    let scores = seed_scores(&[("aa".into(), 3), ("b".into(), 2)]);
    assert!((scores["a"] - 0.75_f64.ln()).abs() < 1e-12);
    assert!((scores["b"] - 0.25_f64.ln()).abs() < 1e-12);
}

fn check_e_step(count_a: u32, count_b: u32) {
    let model = Unigram::from(
        vec![
            ("<unk>".into(), 0.0),
            ("a".into(), 0.25_f64.ln()),
            ("b".into(), 0.75_f64.ln()),
        ],
        Some(0),
        false,
    )
    .unwrap();
    let (objective, tokens, expected) =
        trainer().run_e_step(&model, &[("a".into(), count_a), ("b".into(), count_b)]);
    let (a, b) = (f64::from(count_a), f64::from(count_b));
    let reference = -(a * 0.25_f64.ln() + b * 0.75_f64.ln()) / (a + b);
    assert!((objective - reference).abs() < 1e-12);
    assert_eq!(tokens, 2);
    assert_eq!(expected, [0.0, a, b]);
}

#[test]
fn corpus_counts_above_u32_keep_the_normalized_objective() {
    for (a, b) in [(u32::MAX, 1), (3_000_000_000, 3_000_000_000)] {
        check_e_step(a, b);
    }
}

#[test]
fn small_corpus_counts_keep_the_normalized_objective() {
    check_e_step(3, 2);
}

#[test]
fn public_training_and_trainer_serialization_accept_large_aggregate_counts() {
    for (repeated, other) in [("aa", "b"), ("éé", "字")] {
        let words = [
            (repeated.to_string(), 3_000_000_000),
            (other.to_string(), 2_000_000_000),
        ]
        .into_iter()
        .collect();
        let original = UnigramTrainer::builder()
            .vocab_size(3)
            .max_piece_length(1)
            .n_sub_iterations(1)
            .unk_token(Some("<unk>".into()))
            .show_progress(false)
            .words(words)
            .build()
            .unwrap();
        let restored: UnigramTrainer =
            serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
        for weighted in [&original, &restored] {
            let mut model = Unigram::default();
            weighted.train(&mut model).unwrap();
            let scores: BTreeMap<_, _> = model.iter().cloned().collect();
            let first = repeated.chars().next().unwrap().to_string();
            assert_eq!(scores.len(), 3);
            // The trainer's digamma prior differs from log-frequency by O(1/count).
            assert!((scores[&first] - 0.75_f64.ln()).abs() < 1e-8);
            assert!((scores[other] - 0.25_f64.ln()).abs() < 1e-8);
            let text = format!("{repeated}{other}");
            let expected: Vec<_> = text.chars().map(|c| c.to_string()).collect();
            assert_eq!(model.encode(&text).unwrap(), expected);
        }
    }
}

#[test]
fn small_weighted_training_matches_the_actual_expanded_corpus() {
    let mut expanded = trainer();
    expanded
        .feed(["aa", "aa", "aa", "b", "b"].into_iter(), |s| Ok(vec![s.into()]))
        .unwrap();
    let mut reference = Unigram::default();
    expanded.train(&mut reference).unwrap();
    let mut weighted = Unigram::default();
    trainer()
        .do_train(vec![("aa".into(), 3), ("b".into(), 2)], &mut weighted)
        .unwrap();
    let scores: BTreeMap<_, _> = weighted.iter().cloned().collect();
    let reference: BTreeMap<_, _> = reference.iter().cloned().collect();
    assert_eq!(scores, reference);
}
