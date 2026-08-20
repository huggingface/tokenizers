//! Tests for the config-shaped [`BPE`]: its builder, its options, its file loaders, and its parity
//! with the runtime engine.
//!
//! The parity module at the bottom is the half of `tk-encode`'s old BPE test file that could not
//! stay there. Those tests compare [`PipelineBPE`] against this `BPE` as a reference implementation,
//! and only this crate can see both types — the pipeline's *own* expectations stayed behind, spelled
//! out as literal id lists.
use super::*;
use std::io::Write;
use tempfile::NamedTempFile;
use tk_encode::models::bpe::Vocab;

#[test]
fn cache_is_per_bpe_instance() {
    // Two BPE instances with different merges must tokenize the same
    // input differently even when they share a thread, i.e. the BPE
    // thread-local cache must not leak entries across instances.
    let vocab_a: Vocab = [
        ("h", 0u32),
        ("e", 1),
        ("l", 2),
        ("o", 3),
        ("he", 4),
        ("hel", 5),
        ("hell", 6),
        ("hello", 7),
    ]
    .iter()
    .map(|(s, i)| ((*s).into(), *i))
    .collect();
    let merges_a: Merges = vec![
        ("h".into(), "e".into()),
        ("he".into(), "l".into()),
        ("hel".into(), "l".into()),
        ("hell".into(), "o".into()),
    ];
    let bpe_a = BpeBuilder::default()
        .vocab_and_merges(vocab_a, merges_a)
        .build()
        .unwrap();

    let vocab_b: Vocab = [("h", 0u32), ("e", 1), ("l", 2), ("o", 3)]
        .iter()
        .map(|(s, i)| ((*s).into(), *i))
        .collect();
    let bpe_b = BpeBuilder::default()
        .vocab_and_merges(vocab_b, vec![])
        .build()
        .unwrap();

    // Interleave the two models so any cross-instance cache pollution
    // is visible on the second lookup.
    let ids_a: Vec<u32> = bpe_a
        .tokenize("hello")
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    let ids_b: Vec<u32> = bpe_b
        .tokenize("hello")
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    let ids_a2: Vec<u32> = bpe_a
        .tokenize("hello")
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    let ids_b2: Vec<u32> = bpe_b
        .tokenize("hello")
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();

    assert_eq!(ids_a, vec![7u32], "bpe_a must merge to [hello]");
    assert_eq!(ids_b, vec![0u32, 1, 2, 2, 3], "bpe_b has no merges");
    assert_eq!(ids_a2, ids_a, "bpe_a second call must match first");
    assert_eq!(ids_b2, ids_b, "bpe_b second call must match first");
}

#[test]
fn unk_not_fused() {
    let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
        .iter()
        .cloned()
        .collect();
    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, vec![])
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let tokens = bpe.tokenize("c").unwrap();
    assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

    let tokens = bpe.tokenize("cc").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::new(0u32, "<unk>".into(), (0, 1)),
            Token::new(0u32, "<unk>".into(), (1, 2)),
        ]
    );

    let tokens = bpe.tokenize("accb").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::new(1u32, "a".into(), (0, 1)),
            Token::new(0u32, "<unk>".into(), (1, 2)),
            Token::new(0u32, "<unk>".into(), (2, 3)),
            Token::new(2u32, "b".into(), (3, 4)),
        ]
    );
}

#[test]
fn unk_get_fused() {
    let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
        .iter()
        .cloned()
        .collect();
    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, vec![])
        .unk_token("<unk>".to_string())
        .fuse_unk(true)
        .build()
        .unwrap();
    let tokens = bpe.tokenize("c").unwrap();
    assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

    let tokens = bpe.tokenize("cc").unwrap();
    assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 2)),]);

    let tokens = bpe.tokenize("accb").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::new(1u32, "a".into(), (0, 1)),
            Token::new(0u32, "<unk>".into(), (1, 3)),
            Token::new(2u32, "b".into(), (3, 4)),
        ]
    );
}

#[test]
// Test tokenization. With dropout set to 0 tokenization is deterministic,
// so we know exactly what the result should be.
//
// To test this, we'll build a simple model to tokenize the word 'unrelated'.
fn tokenize_with_and_without_dropout() {
    let vocab: Vocab = [
        ("u".into(), 0),
        ("n".into(), 1),
        ("r".into(), 2),
        ("e".into(), 3),
        ("l".into(), 4),
        ("a".into(), 5),
        ("t".into(), 6),
        ("d".into(), 7),
        ("re".into(), 8),
        ("at".into(), 9),
        ("ed".into(), 10),
        ("un".into(), 11),
        ("ated".into(), 12),
        ("rel".into(), 13),
        ("related".into(), 14),
        ("unrelated".into(), 15),
    ]
    .iter()
    .cloned()
    .collect();
    let merges: Merges = vec![
        ("r".to_string(), "e".to_string()),
        ("a".to_string(), "t".to_string()),
        ("e".to_string(), "d".to_string()),
        ("u".to_string(), "n".to_string()),
        ("at".to_string(), "ed".to_string()),
        ("re".to_string(), "l".to_string()),
        ("rel".to_string(), "ated".to_string()),
        ("un".to_string(), "related".to_string()),
    ];
    let mut bpe = BPE::new(vocab, merges);

    // With no dropout:
    let tokens = bpe.tokenize("unrelated").unwrap();
    assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

    // With dropout = 0.0 (equivalent to dropout == none)
    bpe.dropout = Some(0.0);
    let tokens = bpe.tokenize("unrelated").unwrap();
    assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

    // Now set dropout to 1.0. Result should be no merges performed.
    bpe.dropout = Some(1.0);
    let tokens = bpe.tokenize("unrelated").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::new(0u32, "u".into(), (0, 1)),
            Token::new(1u32, "n".into(), (1, 2)),
            Token::new(2u32, "r".into(), (2, 3)),
            Token::new(3u32, "e".into(), (3, 4)),
            Token::new(4u32, "l".into(), (4, 5)),
            Token::new(5u32, "a".into(), (5, 6)),
            Token::new(6u32, "t".into(), (6, 7)),
            Token::new(3u32, "e".into(), (7, 8)),
            Token::new(7u32, "d".into(), (8, 9)),
        ]
    );

    // Now try with dropout between 0 and 1.
    bpe.dropout = Some(0.5);
    let tokens = bpe.tokenize("unrelated").unwrap();
    assert!(!tokens.is_empty() && tokens.len() <= 9);
}

#[test]
// Ensure `BPE::from_file` works as expected.
fn bpe_from_file() {
    // Set up vocab file.
    let mut vocab_file = NamedTempFile::new().unwrap();
    vocab_file
        .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
        .unwrap();

    // Set up merges file.
    let mut merges_file = NamedTempFile::new().unwrap();
    merges_file.write_all(b"#version: 0.2\na b").unwrap();

    // Make sure we can instantiate a BPE model from the files.
    let builder = BPE::from_file(
        vocab_file.path().to_str().unwrap(),
        merges_file.path().to_str().unwrap(),
    );
    let bpe = builder.build().unwrap();

    // Check merges.
    assert_eq!(bpe.merges.get(&(0, 1)).unwrap(), &(0u32, 3u32));

    // Check vocab.
    assert_eq!(bpe.vocab.token_to_id("a").unwrap(), 0u32);
    assert_eq!(bpe.vocab.token_to_id("b").unwrap(), 1u32);
    assert_eq!(bpe.vocab.token_to_id("c").unwrap(), 2u32);
    assert_eq!(bpe.vocab.token_to_id("ab").unwrap(), 3u32);
}

#[test]
// Ensure BPEBuilder with dropout = 0.0 doesn't error
fn bpe_with_dropout_0() {
    let bpe = BPE::builder().dropout(0.0).build().unwrap();
    assert_eq!(bpe.dropout, Some(0.0));
}

#[test]
fn bpe_with_dropout_out_of_range_errors() {
    // The range check is the builder's, and it is separate from the pipeline's refusal to *run*
    // dropout: 0.5 builds fine here and is rejected on lowering.
    let err = BPE::builder().dropout(1.5).build().unwrap_err();
    assert!(matches!(
        err.downcast_ref::<Error>(),
        Some(Error::InvalidDropout)
    ));
}

#[test]
fn bpe_with_continuing_subword_prefix() {
    let vocab: Vocab = vec![
        ("a".to_string(), 0),
        ("##b".to_string(), 1),
        ("##c".to_string(), 2),
        ("ab".to_string(), 3),
        ("abc".to_string(), 4),
    ]
    .into_iter()
    .collect();

    let merges = vec![
        ("a".to_string(), "##b".to_string()),
        ("ab".to_string(), "##c".to_string()),
    ];

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("[UNK]".to_string())
        .continuing_subword_prefix("##".to_string())
        .build()
        .unwrap();

    let res = bpe.tokenize("ab");
    assert_eq!(
        res.unwrap(),
        vec![Token {
            id: 3,
            value: "ab".to_string(),
            offsets: (0, 2)
        }]
    );
    let res = bpe.tokenize("abc");
    assert_eq!(
        res.unwrap(),
        vec![Token {
            id: 4,
            value: "abc".to_string(),
            offsets: (0, 3)
        }]
    );
}

#[test]
// Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
fn bpe_from_file_merge_token_oov() {
    // Set up vocab file.
    let mut vocab_file = NamedTempFile::new().unwrap();
    vocab_file
        .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
        .unwrap();

    // Set up merges file.
    let mut merges_file = NamedTempFile::new().unwrap();
    merges_file.write_all(b"#version: 0.2\na b\na d").unwrap();

    // Ensure the result of BPE::from_file is a MergeTokenOutOfVocabulary error.
    match BPE::from_file(
        vocab_file.path().to_str().unwrap(),
        merges_file.path().to_str().unwrap(),
    )
    .build()
    {
        Ok(_) => unreachable!(),
        Err(err) => match err.downcast_ref::<Error>() {
            Some(Error::MergeTokenOutOfVocabulary(token)) => {
                assert_eq!(*token, String::from("d"))
            }
            _ => unreachable!(),
        },
    }
}

#[test]
// Ensure `BadMerges` error is returned when there is an invalid line in the
// merges.txt file.
fn bpe_from_file_bad_merges() {
    // Set up vocab file.
    let mut vocab_file = NamedTempFile::new().unwrap();
    vocab_file
        .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
        .unwrap();

    // Set up merges file with a bad line.
    let mut merges_file = NamedTempFile::new().unwrap();
    merges_file.write_all(b"#version: 0.2\na b\nc").unwrap();

    // Ensure the result of BPE::from_file is a BadMerges error.
    match BPE::from_file(
        vocab_file.path().to_str().unwrap(),
        merges_file.path().to_str().unwrap(),
    )
    .build()
    {
        Ok(_) => unreachable!(),
        Err(err) => match err.downcast_ref::<Error>() {
            Some(Error::BadMerges(line)) => assert_eq!(*line, 2),
            _ => unreachable!(),
        },
    }
}

#[test]
fn bpe_byte_fallback() {
    // 0x61 == 'a' in bytes
    let vocab: Vocab = [("<unk>".into(), 0), ("<0x61>".into(), 1)]
        .iter()
        .cloned()
        .collect();
    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, vec![])
        .unk_token("<unk>".to_string())
        .byte_fallback(true)
        .build()
        .unwrap();
    let tokens = bpe.tokenize("c").unwrap();
    assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

    let tokens = bpe.tokenize("a").unwrap();
    assert_eq!(tokens, vec![Token::new(1u32, "<0x61>".into(), (0, 1)),]);
}

#[test]
fn bpe_byte_fallback_newline() {
    // 0x0A == '\n' in bytes
    let vocab: Vocab = [("<unk>".into(), 0), ("<0x0A>".into(), 1)]
        .iter()
        .cloned()
        .collect();
    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, vec![])
        .unk_token("<unk>".to_string())
        .byte_fallback(true)
        .build()
        .unwrap();
    let tokens = bpe.tokenize("\n").unwrap();
    assert_eq!(tokens, vec![Token::new(1u32, "<0x0A>".into(), (0, 1)),]);
}

#[test]
fn ignore_merges() {
    let vocab: Vocab = [
        (".:.:".into(), 0),
        ("Ġbelirtilen".into(), 1),
        (".".into(), 2),
        (":".into(), 3),
        ("bel".into(), 4),
        ("irtilen".into(), 5),
        ("Ġ".into(), 6),
        (".:".into(), 7),
        ("belirtilen".into(), 8),
        (".:.".into(), 9),
        ("be".into(), 10),
        ("l".into(), 11),
        ("ir".into(), 12),
        ("ti".into(), 13),
        ("en".into(), 14),
        ("irtil".into(), 15),
        ("irti".into(), 16),
        ("i".into(), 17),
        ("r".into(), 18),
        ("t".into(), 19),
        ("b".into(), 20),
        ("e".into(), 21),
        ("n".into(), 22),
    ]
    .iter()
    .cloned()
    .collect();
    let mut bpe = BpeBuilder::default()
        .vocab_and_merges(
            vocab,
            vec![
                (".".into(), ":".into()),
                ("b".into(), "e".into()),
                ("be".into(), "l".into()),
                ("i".into(), "r".into()),
                ("t".into(), "i".into()),
                ("ir".into(), "ti".into()),
                ("e".into(), "n".into()),
                ("irti".into(), "l".into()),
            ],
        )
        .ignore_merges(true)
        .build()
        .unwrap();
    let tokens = bpe.tokenize(".:.:").unwrap();
    assert_eq!(tokens, vec![Token::new(0u32, ".:.:".into(), (0, 4))]);

    let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
    assert_eq!(
        tokens,
        vec![Token::new(1u32, "Ġbelirtilen".into(), (0, 12))]
    );

    bpe.ignore_merges = false;

    let tokens = bpe.tokenize(".:.:").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::new(7u32, ".:".into(), (0, 2)),
            Token::new(7u32, ".:".into(), (2, 4))
        ]
    );

    let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
    assert_eq!(
        tokens,
        vec![
            Token {
                id: 6,
                value: "Ġ".into(),
                offsets: (0, 2)
            },
            Token {
                id: 4,
                value: "bel".into(),
                offsets: (2, 5)
            },
            Token {
                id: 15,
                value: "irtil".into(),
                offsets: (5, 10)
            },
            Token {
                id: 14,
                value: "en".into(),
                offsets: (10, 12)
            }
        ]
    )
}

/// The two engines, run against each other.
///
/// `BPE` is the reference implementation and `PipelineBPE` is what actually encodes, so every one of
/// these builds the same configuration both ways and demands the same ids. This is the only place in
/// the workspace that can: the reference lives here and the engine lives in `tk-encode`.
mod legacy_parity {
    use super::*;
    use tk_encode::models::bpe::{PipelineBPE, PipelineBpeOptions};
    use tk_encode::pipeline::Model as PipelineModelTrait;
    use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;

    const HELLO_VOCAB: &[(&str, u32)] = &[
        ("h", 0),
        ("e", 1),
        ("l", 2),
        ("o", 3),
        ("he", 4),
        ("hel", 5),
        ("hell", 6),
        ("hello", 7),
    ];
    const HELLO_MERGES: &[(&str, &str)] = &[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")];

    fn v(pairs: &[(&str, u32)]) -> Vocab {
        pairs.iter().map(|&(s, i)| (s.into(), i)).collect()
    }

    fn m(pairs: &[(&str, &str)]) -> Merges {
        pairs.iter().map(|&(a, b)| (a.into(), b.into())).collect()
    }

    fn hello_builder() -> BpeBuilder {
        BpeBuilder::default().vocab_and_merges(v(HELLO_VOCAB), m(HELLO_MERGES))
    }

    fn pipeline_ids(model: &PipelineBPE, sequence: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut scratch = model.init_scratch();
        PipelineModelTrait::tokenize_pipeline(model, sequence, &mut scratch, &mut out).unwrap();
        out.iter().map(|t| t.id()).collect()
    }

    fn reference_ids(model: &BPE, sequence: &str) -> Vec<u32> {
        model
            .tokenize(sequence)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect()
    }

    /// Build both engines from one `BPE` and check them against each other over `inputs`.
    fn assert_parity(bpe: BPE, with_byte_level: bool, inputs: &[&str]) {
        let reference = bpe.clone();
        let pipeline = into_pipeline(bpe, with_byte_level).unwrap();
        for input in inputs {
            assert_eq!(
                pipeline_ids(&pipeline, input),
                reference_ids(&reference, input),
                "{input:?}"
            );
        }
    }

    #[test]
    fn merges_match() {
        assert_parity(
            hello_builder().build().unwrap(),
            false,
            &["hello", "hell", "helo", "oleh"],
        );
    }

    /// An unknown character with no unk token simply vanishes, which can make two previously
    /// non-adjacent symbols mergeable. Both engines have to agree on that.
    fn hello_with_unk(fuse_unk: bool) -> BPE {
        let mut vocab = v(HELLO_VOCAB);
        vocab.insert("<unk>".into(), 8);
        BpeBuilder::default()
            .vocab_and_merges(vocab, m(HELLO_MERGES))
            .unk_token("<unk>".into())
            .fuse_unk(fuse_unk)
            .build()
            .unwrap()
    }

    #[test]
    fn dropped_unknown_char_matches() {
        assert_parity(hello_builder().build().unwrap(), false, &["hxe"]);
    }

    #[test]
    fn unk_matches() {
        assert_parity(
            hello_with_unk(false),
            false,
            &["hxe", "xh", "hxxe", "xx", "hello"],
        );
    }

    #[test]
    fn fused_unk_matches() {
        assert_parity(
            hello_with_unk(true),
            false,
            &["hxxe", "xxh", "xxxx", "xhx", "hello"],
        );
    }

    fn byte_fallback_vocab() -> Vocab {
        let mut vocab = v(&[("h", 300), ("e", 301), ("<unk>", 400)]);
        vocab.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), u32::from(b))));
        vocab
    }

    #[test]
    fn byte_fallback_matches() {
        assert_parity(
            BpeBuilder::default()
                .vocab_and_merges(byte_fallback_vocab(), vec![])
                .byte_fallback(true)
                .build()
                .unwrap(),
            false,
            &["hé", "🤗", "he"],
        );
    }

    #[test]
    fn byte_fallback_over_unk_matches() {
        assert_parity(
            BpeBuilder::default()
                .vocab_and_merges(byte_fallback_vocab(), vec![])
                .byte_fallback(true)
                .unk_token("<unk>".into())
                .build()
                .unwrap(),
            false,
            &["é", "hé"],
        );
    }

    #[test]
    fn ignore_merges_matches() {
        assert_parity(
            hello_builder().ignore_merges(true).build().unwrap(),
            false,
            &["hello", "helo"],
        );
    }

    /// The scratch pool hands the SAME scratch to successive encodes, so a leak between calls would
    /// corrupt every encode after the first. Drive several inputs — including repeats and an empty
    /// string — through one reused scratch and check each still matches the reference.
    #[test]
    fn reused_scratch_matches_reference() {
        let bpe = hello_builder().build().unwrap();
        let reference = bpe.clone();
        let model = into_pipeline(bpe, false).unwrap();
        let mut scratch = model.init_scratch();
        for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
            let mut out = Vec::new();
            PipelineModelTrait::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
            let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
            assert_eq!(got, reference_ids(&reference, input), "{input:?}");
        }
    }

    fn projected(s: &str) -> String {
        s.bytes().map(|b| BYTES_CHAR_LOOKUP[b as usize]).collect()
    }

    /// A gpt2-shaped miniature. The end-to-end invariant for a byte-level model: *raw* input through
    /// the pipeline must equal *projected* input through the reference, because the pipeline decodes
    /// the vocabulary at load and the reference never does.
    #[test]
    fn byte_level_matches_projected_reference() {
        let mut vocab: Vocab = (0..=255u8)
            .map(|b| (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)))
            .collect();
        vocab.extend([("he", 300u32), (" he", 301)].map(|(s, i)| (projected(s), i)));
        let merges: Merges = [("h", "e"), (" ", "he")]
            .iter()
            .map(|&(a, b)| (projected(a), projected(b)))
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        let reference = bpe.clone();
        let pipeline = into_pipeline(bpe, true).unwrap();
        for input in [" he", "é", "\x00\x7f", "hé llo"] {
            assert_eq!(
                pipeline_ids(&pipeline, input),
                reference_ids(&reference, &projected(input)),
                "{input:?}"
            );
        }
    }

    /// Lowering rejects what the engine cannot run, and the builder is what produced the value it
    /// rejects — so both halves of the dropout rule are visible from here.
    #[test]
    fn lowering_rejects_dropout() {
        let bpe = hello_builder().dropout(0.5).build().unwrap();
        assert!(into_pipeline(bpe, false).is_err());
        let bpe = hello_builder().dropout(0.0).build().unwrap();
        assert!(into_pipeline(bpe, false).is_ok());
    }

    /// `cache_capacity(0)` is the one option whose meaning is carried by an `Option` on the runtime
    /// side and by a `0` on the config side, so lowering it must still encode identically. (Whether
    /// the cache is *there* is checked inside `tk-encode`, which can see the scratch.)
    #[test]
    fn cache_capacity_survives_lowering() {
        assert_parity(
            hello_builder().cache_capacity(0).build().unwrap(),
            false,
            &["hello", "hello", "helo", "hxe"],
        );
    }

    /// The two constructors are one implementation: entering from raw vocab-and-merges and entering
    /// from the tables the builder resolved has to give the same model.
    #[test]
    fn both_constructors_agree() {
        let from_config = into_pipeline(hello_builder().build().unwrap(), false).unwrap();
        let from_parts = PipelineBPE::from_vocab_and_merges(
            v(HELLO_VOCAB),
            m(HELLO_MERGES),
            PipelineBpeOptions::default(),
        )
        .unwrap();
        for input in ["hello", "hell", "helo", "oleh", "hxe", ""] {
            assert_eq!(
                pipeline_ids(&from_config, input),
                pipeline_ids(&from_parts, input),
                "{input:?}"
            );
        }
    }
}
