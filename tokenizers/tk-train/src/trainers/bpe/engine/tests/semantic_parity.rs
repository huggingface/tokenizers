//! Weighted rule choices, ties, overlaps, filtering, affixes, and reference traces.
use super::*;

#[test]
fn weighted_ties_unicode_aa_and_reserved_id_activations() {
    let words = counts(&[
        ("", 1),
        ("a", 10),
        ("aaaaaaa", 3),
        ("abababab", 4),
        ("abcabc", 4),
        ("测试测试", 7),
        ("ééé", 2),
        ("baab", 0),
    ]);
    for special in [
        vec![],
        ["ab", "aba", "ab", "aa", "aaaa"]
            .map(|s| AddedToken::from(s, true))
            .to_vec(),
    ] {
        let trainer = BpeTrainer::builder()
            .vocab_size(100)
            .special_tokens(special)
            .show_progress(false)
            .build();
        check(&trainer, &words);
        check(&trainer, &counts(&[(&"a".repeat(1024), 3)]));
    }
}

#[test]
fn affixes_aliases_and_strict_length_boundaries() {
    let words = counts(&[
        ("aaaaaaa", 11),
        ("abab", 7),
        ("baaba", 4),
        ("测试测试", 3),
        ("ccc", 1),
    ]);
    // Named cases retain the distinct birth gates, identity collisions, and
    // empty-affix behavior without repeating every unrelated combination.
    for (_name, prefix, suffix, limit) in [
        ("plain", None, None, None),
        ("zero birth gate", None, None, Some(0)),
        ("unit birth gate", None, None, Some(1)),
        ("exact pair gate", None, None, Some(2)),
        ("short birth gate", None, None, Some(3)),
        ("middle birth gate", None, None, Some(5)),
        ("long birth gate", None, None, Some(16)),
        ("prefix", Some("##"), None, None),
        ("suffix", None, Some("</w>"), None),
        ("both affixes", Some("##"), Some("</w>"), None),
        ("prefix identity reuse", Some("a"), None, Some(3)),
        ("suffix unequal spans", None, Some("a"), Some(5)),
        ("both identity reuse", Some("a"), Some("a"), Some(3)),
        ("decorated pair gate", Some("##"), Some("</w>"), Some(2)),
        ("decorated long gate", Some("##"), Some("</w>"), Some(16)),
        ("empty prefix", Some(""), Some("</w>"), Some(3)),
        ("empty suffix", Some("##"), Some(""), Some(3)),
        ("both empty", Some(""), Some(""), None),
    ] {
        let mut trainer = BpeTrainer::builder()
            .vocab_size(70)
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("aa", true)])
            .build();
        trainer.continuing_subword_prefix = prefix.map(str::to_owned);
        trainer.end_of_word_suffix = suffix.map(str::to_owned);
        trainer.max_token_length = limit;
        check(&trainer, &words);
    }
}

#[test]
fn pruning_waits_for_all_birth_producers_and_alphabet_filtering() {
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .show_progress(false)
        .build();
    check(&trainer, &counts(&[("xabp", 1), ("xabq", 1)]));
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .limit_alphabet(3)
        .initial_alphabet(['a', '测'].into())
        .show_progress(false)
        .build();
    let words = counts(&[
        ("aaaaaaa", 11),
        ("abab", 7),
        ("caba", 4),
        ("测试测试", 3),
        ("ccc", 1),
    ]);
    check(&trainer, &words);
    // The mainline oracle shares alphabet construction. A literal expectation
    // independently checks the frequency limit and forced characters.
    let mut alphabet_only = trainer;
    alphabet_only.vocab_size = 3;
    let (vocab, merges, _) =
        train(&alphabet_only, WordCountsView::from_map(&words), 1, None).unwrap();
    assert_eq!(
        vocab,
        [("a".into(), 0), ("b".into(), 1), ("测".into(), 2)]
            .into_iter()
            .collect::<AHashMap<_, _>>()
    );
    assert!(merges.is_empty());
}
