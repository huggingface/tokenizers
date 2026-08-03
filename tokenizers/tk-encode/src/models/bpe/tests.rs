//! Tests for both BPE models: the legacy [`BPE`] and the pipeline [`PipelineBPE`].
use super::*;
use crate::pipeline;
use crate::models::OrderedVocabIter;
use std::io::Write;
use crate::tokenizer::{Model, Result, Token};

    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_is_per_bpe_instance() {
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
    fn test_unk_not_fused() {
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
    fn test_unk_get_fused() {
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
    fn test_tokenize_with_and_without_dropout() {
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
    fn test_bpe_byte_fallback() {
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
    fn test_bpe_byte_fallback_newline() {
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
    fn test_ignore_merges() {
        // 0x0A == '\n' in bytes
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

    mod pipeline_bpe {
        use super::*;
        use crate::{
            Model, pipeline::Model as PipelineModel, utils::byte_level::BYTES_CHAR_LOOKUP,
        };

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
        const HELLO_MERGES: &[(&str, &str)] =
            &[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")];

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
            out.iter().map(|t| t.id).collect()
        }

        fn reference_ids(model: &BPE, sequence: &str) -> Vec<u32> {
            model
                .tokenize(sequence)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect()
        }

        #[test]
        fn applies_merges() {
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hello", vec![7]),
                ("hell", vec![6]),
                ("helo", vec![5, 3]),
                ("oleh", vec![3, 2, 1, 0]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
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
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let model = PipelineBPE::from_bpe(bpe, false).unwrap();
            let mut scratch = model.init_scratch();
            for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
                let mut out = Vec::new();
                pipeline::Model::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
                let got: Vec<u32> = out.iter().map(|t| t.id).collect();
                assert_eq!(got, reference_ids(&reference, input), "{input:?}");
            }
        }

        #[test]
        fn unknown_char_without_unk_is_dropped() {
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // 'x' vanishes, making 'h' and 'e' adjacent, so the (h,e) merge
            // applies — mirrors the reference model.
            assert_eq!(pipeline_ids(&pipeline, "hxe"), vec![4]);
            assert_eq!(
                pipeline_ids(&pipeline, "hxe"),
                reference_ids(&reference, "hxe")
            );
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
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hxe", vec![0, 8, 1]),
                ("xh", vec![8, 0]),
                ("hxxe", vec![0, 8, 8, 1]),
                ("xx", vec![8, 8]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
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
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hxxe", vec![0, 8, 1]),
                ("xxh", vec![8, 0]),
                ("xxxx", vec![8]),
                ("xhx", vec![8, 0, 8]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
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
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // 'é' is not in the vocab: falls back to its UTF-8 bytes C3 A9
            assert_eq!(pipeline_ids(&pipeline, "hé"), vec![300, 0xC3, 0xA9]);
            assert_eq!(pipeline_ids(&pipeline, "🤗"), vec![0xF0, 0x9F, 0xA4, 0x97]);
            for input in ["hé", "🤗", "he"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        #[test]
        fn byte_fallback_wins_over_unk() {
            let bpe = BpeBuilder::default()
                .vocab_and_merges(byte_fallback_vocab(), vec![])
                .byte_fallback(true)
                .unk_token("<unk>".into())
                .build()
                .unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
            assert_eq!(pipeline_ids(&pipeline, "é"), reference_ids(&reference, "é"));
        }

        #[test]
        fn ignore_merges_prefers_whole_word() {
            let bpe = hello_builder().ignore_merges(true).build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // direct vocab hit bypasses the merge loop; a miss falls through to it
            assert_eq!(pipeline_ids(&pipeline, "hello"), vec![7]);
            assert_eq!(pipeline_ids(&pipeline, "helo"), vec![5, 3]);
            for input in ["hello", "helo"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
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
        fn byte_level_bpe(
            extra: &[(&str, u32)],
            merges: &[(&str, &str)],
            ignore_merges: bool,
        ) -> BPE {
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
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, true).unwrap();
            assert_eq!(pipeline_ids(&pipeline, " he"), vec![301]);
            // single bytes hit the un-projected single-byte tokens (id == byte value)
            assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
            // the end-to-end invariant: raw input through the pipeline must equal
            // projected input through the reference model
            for input in [" he", "é", "\x00\x7f", "hé llo"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, &projected(input)),
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
