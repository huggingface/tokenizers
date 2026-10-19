//! Reserved identities, active aliases, cohort semantics, and attempt restart.
use super::*;

#[test]
fn cohort_cohort_words_include_stale_addresses_and_zero_weights() {
    // The suffix creates an active "aa" ID before AA -> aa. Across words,
    // left and right birth chains for that identity interleave spatially.
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .end_of_word_suffix("a".into())
        .show_progress(false)
        .build();
    check(&trainer, &counts(&[("aaaaa", 1), ("aaaaaaa", 1)]));
    let words = (0..128)
        .map(|i| {
            (
                format!("{i}{}baaba", "aaaaaaaaabcd".repeat(16)).into(),
                [0, 1, 7, 13][i % 4],
            )
        })
        .collect();
    for (prefix, suffix, limit) in [
        (None, Some("a"), None),
        (Some("ab"), None, Some(9)),
        (Some("##"), Some("</w>"), Some(3)),
    ] {
        let mut trainer = BpeTrainer::builder()
            .vocab_size(80)
            .min_frequency(2)
            .show_progress(false)
            .max_token_length(limit)
            .build();
        trainer.continuing_subword_prefix = prefix.map(str::to_owned);
        trainer.end_of_word_suffix = suffix.map(str::to_owned);
        check(&trainer, &words);
    }
}

#[test]
fn suffix_identity_reuse_has_literal_mainline_merge_choices() {
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    let words = counts(&[("baaba", 1)]);
    let mut trace = Vec::new();
    let (_, merges, _) = train(
        &trainer,
        WordCountsView::from_map(&words),
        2,
        Some(&mut |pair, count, id| trace.push((pair, count, id))),
    )
    .unwrap();
    assert_eq!(&trace[..2], &[((0, 0), 1, 2), ((1, 2), 2, 3)]);
    assert_eq!(
        &merges[..2],
        &[("a".into(), "a".into()), ("b".into(), "aa".into())]
    );
    assert_eq!(trainer.do_train(&words).unwrap().1, merges);
}

#[test]
fn active_id_reuse_rebuilds_without_publishing_speculative_rules() {
    let mut trainer = BpeTrainer::builder()
        .vocab_size(48)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    for (late, limited) in [(false, false), (true, false), (false, true)] {
        trainer.limit_alphabet = limited.then_some(2);
        trainer.initial_alphabet = if limited {
            ['a', 'b'].into()
        } else {
            Default::default()
        };
        let mut words = counts(&[("baaba", 1)]);
        if late {
            words.insert("xyxyxy".into(), 100);
        }
        let execution = execution::Execution::new(2).unwrap();
        let (outcome, trace, retained_alphabet) = execution.pool.install(|| {
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            let mut training = Training {
                trainer: &trainer,
                execution: &execution,
                merge_options: merge::MergeOptions::default(),
                progress: &progress,
                policy: IdentityPolicy::FirstActivationOnly,
                retained_alphabet: None,
                trace: Vec::new(),
            };
            let outcome = training
                .attempt(WordCountsView::from_map(&words), &mut None)
                .unwrap();
            (outcome, training.trace, training.retained_alphabet)
        });
        assert!(matches!(outcome, AttemptOutcome::RestartForReuse));
        assert_eq!(trace.is_empty(), !late);
        if limited {
            assert_eq!(retained_alphabet.as_deref(), Some(['a', 'b'].as_slice()));
        }
        // Exact oracle traces expose duplicate publication on a late restart.
        // They also cover a first-rule collision and rebuilt vocabulary IDs.
        check_with_workers(&trainer, &words, &[1, 4]);
    }
}

#[test]
fn affix_first_activations_preserve_reserved_ids_and_model_order() {
    let words = counts(&[
        (&"xabcdab中abab".repeat(40), 7),
        (&"abababaaaa中文".repeat(20), 11),
        ("zeroaaaa🙂", 0),
        ("", 1),
    ]);
    let trainer = BpeTrainer::builder()
        .vocab_size(80)
        .min_frequency(2)
        .show_progress(false)
        .max_token_length(Some(7))
        .continuing_subword_prefix("##".into())
        .end_of_word_suffix("</w>".into())
        .special_tokens(vec![AddedToken::from("##ab", true)])
        .build();
    let execution = execution::Execution::new(2).unwrap();
    let (outcome, trace) = execution.pool.install(|| {
        let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
        let mut training = Training {
            trainer: &trainer,
            execution: &execution,
            merge_options: merge::MergeOptions::default(),
            progress: &progress,
            policy: IdentityPolicy::FirstActivationOnly,
            retained_alphabet: None,
            trace: Vec::new(),
        };
        let outcome = training
            .attempt(WordCountsView::from_map(&words), &mut None)
            .unwrap();
        (outcome, training.trace)
    });
    let AttemptOutcome::Complete((vocab, _, _)) = outcome else {
        panic!("first activations do not require ID-reuse execution");
    };
    assert_eq!(vocab["##ab"], 0);
    assert!(trace.iter().any(|&(_, _, id)| id == 0));
    check_with_workers(&trainer, &words, &[1, 4]);
}
