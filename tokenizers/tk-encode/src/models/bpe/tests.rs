//! Tests for the `BPE` config/loading surface and the pipeline [`PipelineBPE`] encoder.
use super::*;
use crate::models::OrderedVocabIter;
use crate::pipeline;
use std::io::Write;

use tempfile::NamedTempFile;

#[test]
fn test_ordered_vocab_iter() {
    let vocab_r: VocabR = [
        (0, "a".into()),
        (1, "b".into()),
        (2, "c".into()),
        (3, "ab".into()),
    ]
    .iter()
    .cloned()
    .collect();
    let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
    let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
    assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
}

#[test]
// Ensure `BPE::from_file` works as expected.
fn test_bpe_from_file() {
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
fn test_bpe_with_dropout_0() {
    let bpe = BPE::builder().dropout(0.0).build().unwrap();
    assert_eq!(bpe.dropout, Some(0.0));
}

#[test]
// Ensure `BPE::from_file` works as expected.
fn test_bpe_with_continuing_subword_prefix() {
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

    assert_eq!(bpe.vocab.token_to_id("##b").unwrap(), 1u32);
    assert_eq!(bpe.merges.get(&(0, 1)).unwrap(), &(0u32, 3u32));
}

#[test]
// Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
fn test_bpe_from_file_merge_token_oov() {
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
fn test_bpe_from_file_bad_merges() {
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
fn test_ignore_merges_vocab_and_merges_load() {
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
    let bpe = BpeBuilder::default()
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

    assert!(bpe.ignore_merges);
    assert_eq!(bpe.vocab.token_to_id(".:.:").unwrap(), 0u32);
    assert_eq!(bpe.vocab.token_to_id("Ġbelirtilen").unwrap(), 1u32);
}

mod pipeline_bpe {
    use super::*;
    use crate::{pipeline::Model as PipelineModel, utils::byte_level::BYTES_CHAR_LOOKUP};

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
        pipeline::Model::tokenize_pipeline(model, sequence, &mut scratch, &mut out).unwrap();
        out.iter().map(|t| t.id()).collect()
    }

    #[test]
    fn applies_merges() {
        let bpe = hello_builder().build().unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        for (input, want) in [
            ("hello", vec![7]),
            ("hell", vec![6]),
            ("helo", vec![5, 3]),
            ("oleh", vec![3, 2, 1, 0]),
        ] {
            assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
        }
    }

    // Each model owns its word cache. Two models that disagree on the merges must keep
    // disagreeing when interleaved on one thread: a cache keyed on the word alone, shared
    // across instances, would answer the second model from the first model's entry.
    #[test]
    fn cache_is_per_model_instance() {
        let merged = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        let unmerged = PipelineBPE::from_bpe(
            BpeBuilder::default()
                .vocab_and_merges(v(HELLO_VOCAB), vec![])
                .build()
                .unwrap(),
            false,
        )
        .unwrap();

        for _ in 0..2 {
            assert_eq!(pipeline_ids(&merged, "hello"), vec![7]);
            assert_eq!(pipeline_ids(&unmerged, "hello"), vec![0, 1, 2, 2, 3]);
        }
    }

    // dropout > 0 is rejected by `from_bpe`, so the only merge behavior the pipeline can be
    // held to is the deterministic one: every applicable merge is taken.
    #[test]
    fn merges_to_the_longest_vocab_entry() {
        let vocab = v(&[
            ("u", 0),
            ("n", 1),
            ("r", 2),
            ("e", 3),
            ("l", 4),
            ("a", 5),
            ("t", 6),
            ("d", 7),
            ("re", 8),
            ("at", 9),
            ("ed", 10),
            ("un", 11),
            ("ated", 12),
            ("rel", 13),
            ("related", 14),
            ("unrelated", 15),
        ]);
        let merges = m(&[
            ("r", "e"),
            ("a", "t"),
            ("e", "d"),
            ("u", "n"),
            ("at", "ed"),
            ("re", "l"),
            ("rel", "ated"),
            ("un", "related"),
        ]);
        let pipeline = PipelineBPE::from_bpe(BPE::new(vocab, merges), false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "unrelated"), vec![15]);
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        let pipeline = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        assert!(pipeline_ids(&pipeline, "").is_empty());
    }

    // The scratch pool hands the SAME scratch to successive encodes. A bug leaking
    // state between calls (an undrained merge queue, a stale word buffer) would
    // corrupt every encode after the first. Drive several inputs — including
    // repeats and an empty string — through one reused scratch and check each still
    // matches the fresh-scratch reference. This is the invariant the pool relies on.
    #[test]
    fn reused_scratch_matches_fresh() {
        let model = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        let mut scratch = model.init_scratch();
        for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
            let mut out = Vec::new();
            pipeline::Model::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
            let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
            // `pipeline_ids` builds a scratch of its own, so it is the fresh-scratch run
            assert_eq!(got, pipeline_ids(&model, input), "{input:?}");
        }
    }

    // A cache may forget a word, but it must never change one. Run every word twice
    // through one scratch, the second time answered from the cache, against a model
    // built with no cache at all.
    #[test]
    fn cached_ids_match_uncached() {
        let cached = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        let uncached =
            PipelineBPE::from_bpe(hello_builder().cache_capacity(0).build().unwrap(), false)
                .unwrap();

        let mut scratch = cached.init_scratch();
        assert!(scratch.word_cache.is_some(), "nothing is being cached");
        for _ in 0..2 {
            for word in [
                "hello",
                "hell",
                "o",
                "hellohello",
                "hello-a-word-past-fifteen-bytes",
                "hxe",
            ] {
                let mut out = Vec::new();
                pipeline::Model::tokenize_pipeline(&cached, word, &mut scratch, &mut out).unwrap();
                let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
                assert_eq!(got, pipeline_ids(&uncached, word), "{word:?}");
            }
        }
    }

    #[test]
    fn unknown_char_without_unk_is_dropped() {
        let pipeline = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        // 'x' vanishes, making 'h' and 'e' adjacent, so the (h,e) merge applies
        assert_eq!(pipeline_ids(&pipeline, "hxe"), vec![4]);
    }

    #[test]
    fn unk_replaces_unknown_chars() {
        let mut vocab = v(HELLO_VOCAB);
        vocab.insert("<unk>".into(), 8);
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, m(HELLO_MERGES))
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        for (input, want) in [
            ("hxe", vec![0, 8, 1]),
            ("xh", vec![8, 0]),
            ("hxxe", vec![0, 8, 8, 1]),
            ("xx", vec![8, 8]),
        ] {
            assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
        }
    }

    #[test]
    fn fused_unk_collapses_runs() {
        let mut vocab = v(HELLO_VOCAB);
        vocab.insert("<unk>".into(), 8);
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, m(HELLO_MERGES))
            .unk_token("<unk>".into())
            .fuse_unk(true)
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        for (input, want) in [
            ("hxxe", vec![0, 8, 1]),
            ("xxh", vec![8, 0]),
            ("xxxx", vec![8]),
            ("xhx", vec![8, 0, 8]),
        ] {
            assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
        }
    }

    // A merge-free vocab, so the only thing on trial is how unknown characters are
    // emitted between known ones.
    fn abc_unk_builder(fuse_unk: bool) -> BpeBuilder {
        BpeBuilder::default()
            .vocab_and_merges(v(&[("<unk>", 0), ("a", 1), ("b", 2)]), vec![])
            .unk_token("<unk>".into())
            .fuse_unk(fuse_unk)
    }

    #[test]
    fn unfused_unk_is_emitted_per_character() {
        let pipeline =
            PipelineBPE::from_bpe(abc_unk_builder(false).build().unwrap(), false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "c"), vec![0]);
        assert_eq!(pipeline_ids(&pipeline, "cc"), vec![0, 0]);
        assert_eq!(pipeline_ids(&pipeline, "accb"), vec![1, 0, 0, 2]);
    }

    #[test]
    fn fused_unk_between_known_chars_is_one_token() {
        let pipeline =
            PipelineBPE::from_bpe(abc_unk_builder(true).build().unwrap(), false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "c"), vec![0]);
        assert_eq!(pipeline_ids(&pipeline, "cc"), vec![0]);
        assert_eq!(pipeline_ids(&pipeline, "accb"), vec![1, 0, 2]);
    }

    // The legacy model accepted a partial `<0xNN>` vocab and quietly took the unk for any byte
    // it was missing. The pipeline builds a 256-entry byte table up front instead, so an
    // incomplete byte vocab is a build error rather than a silent unk at encode time.
    #[test]
    fn byte_fallback_covers_control_bytes() {
        let bpe = BpeBuilder::default()
            .vocab_and_merges(byte_fallback_vocab(), vec![])
            .byte_fallback(true)
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "\n"), vec![0x0A]);
    }

    // `continuing_subword_prefix` decorates every character after the first before the
    // vocab lookup, so "abc" resolves through a/##b/##c and merges to the whole word.
    #[test]
    fn continuing_subword_prefix_merges_to_whole_word() {
        let bpe = BPE::builder()
            .vocab_and_merges(
                v(&[("a", 0), ("##b", 1), ("##c", 2), ("ab", 3), ("abc", 4)]),
                m(&[("a", "##b"), ("ab", "##c")]),
            )
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "ab"), vec![3]);
        assert_eq!(pipeline_ids(&pipeline, "abc"), vec![4]);
    }

    fn byte_fallback_vocab() -> Vocab {
        let mut vocab = v(&[("h", 300), ("e", 301), ("<unk>", 400)]);
        vocab.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), u32::from(b))));
        vocab
    }

    #[test]
    fn byte_fallback_encodes_missing_chars_as_byte_tokens() {
        let bpe = BpeBuilder::default()
            .vocab_and_merges(byte_fallback_vocab(), vec![])
            .byte_fallback(true)
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        // 'é' is not in the vocab: falls back to its UTF-8 bytes C3 A9
        assert_eq!(pipeline_ids(&pipeline, "hé"), vec![300, 0xC3, 0xA9]);
        assert_eq!(pipeline_ids(&pipeline, "🤗"), vec![0xF0, 0x9F, 0xA4, 0x97]);
        assert_eq!(pipeline_ids(&pipeline, "he"), vec![300, 301]);
    }

    #[test]
    fn byte_fallback_wins_over_unk() {
        let bpe = BpeBuilder::default()
            .vocab_and_merges(byte_fallback_vocab(), vec![])
            .byte_fallback(true)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
        assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
    }

    #[test]
    fn ignore_merges_prefers_whole_word() {
        let pipeline =
            PipelineBPE::from_bpe(hello_builder().ignore_merges(true).build().unwrap(), false)
                .unwrap();
        // direct vocab hit bypasses the merge loop; a miss falls through to it
        assert_eq!(pipeline_ids(&pipeline, "hello"), vec![7]);
        assert_eq!(pipeline_ids(&pipeline, "helo"), vec![5, 3]);
        // without the flag the same words go through the merge loop and land on the same ids
        let merging = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
        assert_eq!(pipeline_ids(&merging, "hello"), vec![7]);
        assert_eq!(pipeline_ids(&merging, "helo"), vec![5, 3]);
    }

    #[test]
    fn rejects_unsupported_configs() {
        // no merges: BpeBuilder::build underflows on merges whose right token
        // is shorter than continuing_subword_prefix (pre-existing, unrelated)
        let build = |f: fn(BpeBuilder) -> BpeBuilder| {
            f(BpeBuilder::default().vocab_and_merges(v(HELLO_VOCAB), vec![]))
                .build()
                .unwrap()
        };
        assert!(PipelineBPE::from_bpe(build(|b| b.dropout(0.5)), false).is_err());
        // affixes are supported: `convert_affixed` decorates each character before the lookup
        assert!(
            PipelineBPE::from_bpe(build(|b| b.continuing_subword_prefix("##".into())), false)
                .is_ok()
        );
        assert!(
            PipelineBPE::from_bpe(build(|b| b.end_of_word_suffix("</w>".into())), false).is_ok()
        );
        // no-op values must not be rejected: gpt2's tokenizer.json serializes
        // prefix/suffix as "" and the reference treats dropout 0.0 as disabled
        assert!(
            PipelineBPE::from_bpe(
                build(|b| {
                    b.continuing_subword_prefix(String::new())
                        .end_of_word_suffix(String::new())
                        .dropout(0.0)
                }),
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_unk_token_missing_from_vocab() {
        let bpe = hello_builder().unk_token("<unk>".into()).build().unwrap();
        assert!(PipelineBPE::from_bpe(bpe, false).is_err());
    }

    #[test]
    fn byte_fallback_with_missing_codes_errors() {
        // Incomplete <0xNN> coverage must be a build error, not a panic.
        let bpe = hello_builder().byte_fallback(true).build().unwrap();
        assert!(PipelineBPE::from_bpe(bpe, false).is_err());
    }

    fn projected(s: &str) -> String {
        s.bytes().map(|b| BYTES_CHAR_LOOKUP[b as usize]).collect()
    }

    /// A gpt2-shaped miniature: the 256 projected single-byte tokens
    /// (id == byte value) plus `extra` tokens and merges, given in raw
    /// space and projected here — like a real byte-level tokenizer.json,
    /// whose vocab is stored in the projected alphabet.
    fn byte_level_bpe(extra: &[(&str, u32)], merges: &[(&str, &str)], ignore_merges: bool) -> BPE {
        let mut vocab: Vocab = (0..=255u8)
            .map(|b| (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)))
            .collect();
        vocab.extend(extra.iter().map(|&(s, i)| (projected(s), i)));
        let merges: Merges = merges
            .iter()
            .map(|&(a, b)| (projected(a), projected(b)))
            .collect();
        BpeBuilder::default()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(ignore_merges)
            .build()
            .unwrap()
    }

    #[test]
    fn byte_level_merges_raw_bytes() {
        let bpe = byte_level_bpe(
            &[("he", 300), (" he", 301)],
            &[("h", "e"), (" ", "he")],
            false,
        );
        let pipeline = PipelineBPE::from_bpe(bpe, true).unwrap();
        assert_eq!(pipeline_ids(&pipeline, " he"), vec![301]);
        // single bytes hit the un-projected single-byte tokens (id == byte value)
        assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
        // with no merge available every byte stands alone, still keyed by byte value —
        // the pipeline takes raw bytes where the vocab is stored projected
        for input in ["\x00\x7f", "hé llo"] {
            assert_eq!(
                pipeline_ids(&pipeline, input),
                input.bytes().map(u32::from).collect::<Vec<_>>(),
                "{input:?}"
            );
        }
    }

    #[test]
    fn byte_level_ignore_merges_whole_word() {
        let bpe = byte_level_bpe(&[(" hello", 300)], &[], true);
        let pipeline = PipelineBPE::from_bpe(bpe, true).unwrap();
        assert_eq!(pipeline_ids(&pipeline, " hello"), vec![300]);
        // not in vocab → falls through to single-byte atoms
        assert_eq!(
            pipeline_ids(&pipeline, "zz"),
            vec![u32::from(b'z'), u32::from(b'z')]
        );
    }

    #[test]
    fn byte_level_requires_full_byte_coverage() {
        // An ASCII-only vocab covers no control/high bytes: building the
        // byte-level pipeline must be a build error, not a panic.
        let bpe = hello_builder().build().unwrap();
        assert!(PipelineBPE::from_bpe(bpe, true).is_err());
    }
}
