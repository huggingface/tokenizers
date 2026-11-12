use super::*;
use ahash::RandomState;

fn required(chars: &[&str], seed: u64) -> AHashSet<String> {
    let mut set = AHashSet::with_hasher(RandomState::with_seeds(seed, 17, 31, 47));
    set.extend(chars.iter().map(|s| (*s).to_owned()));
    set
}

fn trained_pieces(pieces: &[(&str, f64)]) -> Unigram {
    let mut vocab = vec![("<UNK>".to_owned(), f64::NAN)];
    vocab.extend(pieces.iter().map(|(token, score)| ((*token).to_owned(), *score)));
    Unigram::from(vocab, Some(0), false).unwrap()
}

#[test]
fn equal_score_character_ids_do_not_depend_on_hash_seed() {
    let trainer = UnigramTrainer::builder().show_progress(false).build().unwrap();
    for seed in 0..32 {
        let model = trainer
            .finalize(
                trained_pieces(&[("界", -2.0), ("b", -2.0), ("é", -2.0), ("a", -2.0)]),
                required(&["é", "界", "a", "b"], seed),
            )
            .unwrap();
        for (id, token) in ["a", "b", "é", "界"].into_iter().enumerate() {
            assert_eq!(model.token_to_id(token), Some(id as u32), "seed={seed}");
        }
    }
}

#[test]
fn missing_character_penalties_do_not_depend_on_hash_seed() {
    let trainer = UnigramTrainer::builder().show_progress(false).build().unwrap();
    for seed in 0..32 {
        let model = trainer
            .finalize(
                trained_pieces(&[("a", -2.0), ("z", -3.0)]),
                required(&["é", "c", "a", "b"], seed),
            )
            .unwrap();
        let expected = vec![
            ("a".to_owned(), -2.0),
            ("é".to_owned(), -3.0 + 0.0002),
            ("c".to_owned(), -3.0 + 0.0001),
            ("b".to_owned(), -3.0),
            ("z".to_owned(), -3.0),
        ];
        assert_eq!(model.vocab(), expected, "seed={seed}");
    }
}

#[test]
fn special_token_order_and_unknown_id_are_preserved() {
    let trainer = UnigramTrainer::builder()
        .show_progress(false)
        .special_tokens(vec![
            AddedToken::from("[SEP]", true),
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
        ])
        .unk_token(Some("[UNK]".to_owned()))
        .build()
        .unwrap();
    for seed in 0..16 {
        let model = trainer
            .finalize(
                trained_pieces(&[("c", -1.0), ("b", -1.0), ("a", -1.0)]),
                required(&["c", "a", "b"], seed),
            )
            .unwrap();
        assert_eq!(model.unk_id(), Some(1));
        let tokens: Vec<_> = model.iter().map(|(token, _)| token.as_str()).collect();
        assert_eq!(tokens, ["[SEP]", "[UNK]", "[CLS]", "a", "b", "c"]);
    }
}

#[test]
fn distinct_scores_keep_the_existing_probability_order() {
    let trainer = UnigramTrainer::builder().show_progress(false).build().unwrap();
    for seed in 0..16 {
        let model = trainer
            .finalize(
                trained_pieces(&[("a", -2.0), ("b", -3.0), ("c", -1.0)]),
                required(&["b", "c", "a"], seed),
            )
            .unwrap();
        assert_eq!(
            model.vocab(),
            [("c".to_owned(), -1.0), ("a".to_owned(), -2.0), ("b".to_owned(), -3.0)]
        );
    }
}

#[test]
fn native_training_assigns_repeatable_ids_to_equal_frequency_characters() {
    let trainer = UnigramTrainer::builder()
        .show_progress(false)
        .vocab_size(32)
        .build()
        .unwrap();
    for _ in 0..24 {
        let mut model = Unigram::default();
        trainer
            .do_train(
                ["a", "b", "c", "é", "界"].into_iter().map(|s| (s.to_owned(), 4)).collect(),
                &mut model,
            )
            .unwrap();
        let encoded = model.tokenize("abcé界").unwrap();
        assert_eq!(encoded.iter().map(|t| t.id).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert_eq!(encoded.iter().map(|t| t.offsets).collect::<Vec<_>>(), [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8)]);
    }
}

#[test]
fn unseen_initial_alphabet_scores_and_ids_survive_retraining() {
    let trainer = UnigramTrainer::builder()
        .show_progress(false)
        .vocab_size(32)
        .initial_alphabet(['x', 'é', '界'].into_iter().collect())
        .build()
        .unwrap();
    let mut reference = None;
    for _ in 0..24 {
        let mut model = Unigram::default();
        trainer.do_train(vec![("a".to_owned(), 8)], &mut model).unwrap();
        let vocab = model.vocab().to_vec();
        if let Some(expected) = &reference {
            assert_eq!(&vocab, expected);
        } else {
            reference = Some(vocab);
        }
        let tokens = model.tokenize("axé界").unwrap();
        assert_eq!(tokens.iter().map(|t| t.value.as_str()).collect::<String>(), "axé界");
    }
}
