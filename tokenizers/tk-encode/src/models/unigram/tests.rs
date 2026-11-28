use crate::pipeline::Tokenize;

use super::*;

#[test]
fn encodes_a_vocab_holding_every_utf8_byte() {
    let chars = crate::models::chars_covering_every_utf8_byte();
    let mut pieces = vec![("<unk>".to_string(), 0.0)];
    pieces.extend(chars.iter().map(|c| (c.to_string(), -1.0)));
    let model = Unigram::from(pieces, Some(0), false).unwrap();
    let mut scratch = model.init_scratch();

    for (c, id) in chars.iter().zip(1u32..) {
        let token = c.to_string();
        let mut output = Vec::new();
        model.tokenize(&token, &mut scratch, &mut output).unwrap();
        let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
        assert_eq!(ids, [id], "{c:?}");
        assert_eq!(model.token_to_id(&token), Some(id), "{c:?}");
    }
}

#[test]
fn test_populate_nodes_unk() {
    let pieces = vec![("<unk>".to_string(), 0.0)];
    let model = Unigram::from(pieces, Some(0), false).unwrap();

    let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
    model.populate_nodes(&mut lattice);

    assert_eq!(lattice.begin_nodes[0].len(), 1);
    assert_eq!(lattice.begin_nodes[1].len(), 1);
    assert_eq!(lattice.begin_nodes[2].len(), 1);
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][0]].id, 0);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][0]].id, 0);
    assert_eq!(lattice.nodes[lattice.begin_nodes[2][0]].id, 0);
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][0]].node_id, 2);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][0]].node_id, 3);
    assert_eq!(lattice.nodes[lattice.begin_nodes[2][0]].node_id, 4);
}

#[test]
fn unknown_score_tracks_mutated_min_score_in_both_encoding_paths() {
    let pieces = vec![("<unk>".to_string(), 0.0), ("ab".to_string(), 1.0)];
    let mut model = Unigram::from(pieces, Some(0), false).unwrap();
    model.min_score = 100.0;
    model.set_fuse_unk(false);

    for optimized in [true, false] {
        model.set_optimized(optimized);
        assert_eq!(model.encode("ab").unwrap(), ["a", "b"]);
    }

    let mut lattice = Lattice::from("x", model.bos_id, model.eos_id);
    model.populate_nodes(&mut lattice);
    let mut expected = vec![0.0; model.len()];
    let log_z = lattice.populate_marginal(1.0, &mut expected);
    assert_eq!(log_z, model.min_score - 10.0);
}

#[test]
fn test_populate_nodes() {
    let pieces = vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), 0.1),
        ("b".to_string(), 0.2),
        ("ab".to_string(), 0.3),
        ("bc".to_string(), 0.4),
    ];
    let model = Unigram::from(pieces, Some(0), false).unwrap();

    let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
    model.populate_nodes(&mut lattice);

    assert_eq!(lattice.begin_nodes[0].len(), 2); // a, ab
    assert_eq!(lattice.begin_nodes[1].len(), 2); // b, bc
    assert_eq!(lattice.begin_nodes[2].len(), 1); // c(unk)

    // Id is the vocabulary id from Unigram model
    // node_id is simply the rank of the given node in the lattice.
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][0]].id, 1);
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][1]].id, 3);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][0]].id, 2);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][1]].id, 4);
    assert_eq!(lattice.nodes[lattice.begin_nodes[2][0]].id, 0);
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][0]].node_id, 2);
    assert_eq!(lattice.nodes[lattice.begin_nodes[0][1]].node_id, 3);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][0]].node_id, 4);
    assert_eq!(lattice.nodes[lattice.begin_nodes[1][1]].node_id, 5);
    assert_eq!(lattice.nodes[lattice.begin_nodes[2][0]].node_id, 6);
}

#[test]
fn test_encode() {
    let sentencepieces = vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), 0.0),
        ("b".to_string(), 0.0),
        ("c".to_string(), 0.0),
        ("d".to_string(), 0.0),
        ("cd".to_string(), 1.0),
        ("ab".to_string(), 2.0),
        ("abc".to_string(), 5.0),
        ("abcd".to_string(), 10.0),
    ];

    let model = Unigram::from(sentencepieces, Some(0), false).unwrap();
    let result = model.encode("abcd").unwrap();
    assert_eq!(result, vec!["abcd"]);
}

#[test]
fn test_encode2() {
    let sentencepieces = vec![
        ("<unk>".to_string(), 0.0),
        ("ab".to_string(), 0.0),
        ("cd".to_string(), -0.1),
        ("abc".to_string(), -0.2),
        ("a".to_string(), -0.3),
        ("b".to_string(), -0.4),
        ("c".to_string(), -0.5),
        ("ABC".to_string(), -0.5),
        ("abcdabcd".to_string(), 20.0), // User defined just max the scores.
        ("q".to_string(), 20.5),
        ("r".to_string(), 20.5),
        ("qr".to_string(), -0.5),
    ];

    let mut model = Unigram::from(sentencepieces, Some(0), false).unwrap();

    for is_optimized in &[true, false] {
        model.set_optimized(*is_optimized);
        println!("IsOptimized {is_optimized:?}");
        assert_eq!(model.encode("abc").unwrap(), vec!["abc"]);
        assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

        model.set_fuse_unk(false);
        assert_eq!(model.encode("AB").unwrap(), vec!["A", "B"]);
        model.set_fuse_unk(true);
        assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

        assert_eq!(model.encode("abcd").unwrap(), vec!["ab", "cd"]);
        assert_eq!(model.encode("abcc").unwrap(), vec!["abc", "c"]);
        assert_eq!(
            model.encode("xabcabaabcdd").unwrap(),
            vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
        );
        model.set_fuse_unk(false);
        assert_eq!(
            model.encode("xyz東京").unwrap(),
            vec!["x", "y", "z", "東", "京"]
        );
        model.set_fuse_unk(true);
        assert_eq!(model.encode("xyz東京").unwrap(), vec!["xyz東京"]);

        // User encoded in original version
        assert_eq!(model.encode("ABC").unwrap(), vec!["ABC"]);
        assert_eq!(model.encode("abABCcd").unwrap(), vec!["ab", "ABC", "cd"]);
        assert_eq!(
            model.encode("ababcdabcdcd").unwrap(),
            vec!["ab", "abcdabcd", "cd"]
        );
        assert_eq!(model.encode("abqrcd").unwrap(), vec!["ab", "q", "r", "cd"]);
    }
}

#[test]
fn test_unigram_bytefallback() {
    // In [97]: processor.encode_as_pieces("⅐⅛⅑ ")
    // Out[97]: ['▁', '<0xE2>', '<0x85>', '<0x90>', '⅛', '<0xE2>', '<0x85>', '<0x91>', '▁']
    let sentencepieces = vec![
        ("<unk>".to_string(), 0.0),
        ("<0xC3>".to_string(), -0.01),
        ("<0xA9>".to_string(), -0.03),
    ];
    let unigram = Unigram::from(sentencepieces, Some(0), true).unwrap();
    let mut scratch = Tokenize::init_scratch(&unigram);
    assert_eq!(pipeline_ids(&unigram, "?é", &mut scratch)[0], 0);
}

/// Ids 0..=8 are `<unk>`, `a`, `b`, `c`, `d`, `cd`, `ab`, `abc`, `abcd`.
fn abcd_vocab() -> Vocab {
    vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), 0.0),
        ("b".to_string(), 0.0),
        ("c".to_string(), 0.0),
        ("d".to_string(), 0.0),
        ("cd".to_string(), 1.0),
        ("ab".to_string(), 2.0),
        ("abc".to_string(), 5.0),
        ("abcd".to_string(), 10.0),
    ]
}

fn pipeline_ids(model: &Unigram, sequence: &str, scratch: &mut UnigramScratch) -> Vec<u32> {
    let mut output = vec![];
    Tokenize::tokenize(model, sequence, scratch, &mut output).unwrap();
    output.iter().map(|token| token.id()).collect()
}

#[test]
fn pipeline_remembers_what_a_sequence_encoded_to() {
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    let mut scratch = Tokenize::init_scratch(&model);

    let ids = pipeline_ids(&model, "abcd", &mut scratch);

    assert_eq!(scratch.word_cache.lookup(b"abcd").hit(), Some(&ids[..]));
}

#[test]
fn cache_hits_agree_with_a_cold_run() {
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    let long = "abcd".repeat(400);
    let corpus = [
        "abcdacdxx",
        "ab",
        // The same sequence again, so this one is served from the cache.
        "abcdacdxx",
        // Out of the vocabulary, and multibyte.
        "東京",
        // 1600 bytes, past the longest word the cache will store.
        long.as_str(),
        "abcdacdxx",
    ];

    let mut warm_scratch = Tokenize::init_scratch(&model);
    let warm = corpus.map(|sequence| pipeline_ids(&model, sequence, &mut warm_scratch));
    let cold = corpus.map(|sequence| {
        let mut scratch = Tokenize::init_scratch(&model);
        pipeline_ids(&model, sequence, &mut scratch)
    });

    assert_eq!(warm, cold);
}

#[test]
fn caches_only_the_ids_this_sequence_produced() {
    // Every sequence the pipeline hands the model appends to one output buffer,
    // so a sequence has to remember its own ids, not everything the buffer holds.
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    let mut scratch = Tokenize::init_scratch(&model);
    let mut output = vec![];
    Tokenize::tokenize(&model, "ab", &mut scratch, &mut output).unwrap();
    Tokenize::tokenize(&model, "cd", &mut scratch, &mut output).unwrap();

    let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
    assert_eq!(ids, [6, 5]);
    assert_eq!(scratch.word_cache.lookup(b"cd").hit(), Some(&[5u32][..]));
}

#[test]
fn byte_fallback_ids_survive_the_cache() {
    // A piece the vocabulary has no id for becomes one id per byte. The cache
    // stores what came out, so a hit has to replay all of them.
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("<0xC3>".to_string(), -0.01),
        ("<0xA9>".to_string(), -0.03),
    ];
    let model = Unigram::from(vocab, Some(0), true).unwrap();
    let mut scratch = Tokenize::init_scratch(&model);

    let ids = pipeline_ids(&model, "é", &mut scratch);

    assert_eq!(ids, [1, 2]);
    assert_eq!(pipeline_ids(&model, "é", &mut scratch), ids);
}

#[test]
fn sampling_is_never_cached() {
    // A sampled tokenization is one draw out of many. Remembering it would turn
    // every later call on the same text into that same draw.
    let mut model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    model.alpha = Some(0.5);
    let mut scratch = Tokenize::init_scratch(&model);

    pipeline_ids(&model, "abcd", &mut scratch);

    assert_eq!(scratch.word_cache.lookup(b"abcd").hit(), None);
}

/// The pipeline takes the ids straight off the best path. The lattice path builds a `String`
/// per piece and looks each one back up, so the two must agree on every sequence: same pieces,
/// same fusion of unknown runs, same byte fallback.
#[test]
fn the_ids_path_agrees_with_the_pieces_path() {
    let multibyte = vec![
        ("<unk>".to_string(), 0.0),
        ("é".to_string(), -2.0),
        ("é東".to_string(), -1.0),
        ("東".to_string(), -2.0),
        ("🙂".to_string(), -1.0),
        ("<0xE2>".to_string(), -0.5),
        ("<0x85>".to_string(), -0.5),
        ("<0x90>".to_string(), -0.5),
    ];
    let corpora = [
        "abcdacdxx",
        "ab",
        "",
        "a",
        "abc",
        // Unknown characters, alone and in a run.
        "zzz",
        // 2, 3 and 4 byte characters, mixed with unknowns.
        "é東🙂жé",
        "東京",
        "⅐⅛",
        "abc é東 🙂 xyz",
        // Longer than any cache or path table wants to grow.
        "abcdacdxx東京é🙂",
    ];
    let long = "abcd".repeat(200);

    for byte_fallback in [false, true] {
        for fuse_unk in [false, true] {
            let mut optimized = Unigram::from(multibyte.clone(), Some(0), byte_fallback).unwrap();
            let mut unoptimized = optimized.clone();
            optimized.set_fuse_unk(fuse_unk);
            unoptimized.set_fuse_unk(fuse_unk);
            unoptimized.set_optimized(false);

            let mut optimized_scratch = Tokenize::init_scratch(&optimized);
            let mut unoptimized_scratch = Tokenize::init_scratch(&unoptimized);
            for sequence in corpora.iter().chain(std::iter::once(&long.as_str())) {
                let got = pipeline_ids(&optimized, sequence, &mut optimized_scratch);
                let want = pipeline_ids(&unoptimized, sequence, &mut unoptimized_scratch);
                assert_eq!(
                    got, want,
                    "{sequence:?} with byte_fallback={byte_fallback} fuse_unk={fuse_unk}"
                );
            }
        }
    }
}

/// Unknown runs are fused before the vocabulary sees them, so a fused run that *is* a token
/// encodes to that token's id rather than to `<unk>`. Reachable when every score is above the
/// unknown penalty, so the fused span outscores the vocabulary entry it is named after.
#[test]
fn a_fused_unknown_run_is_looked_up_as_one_piece() {
    let vocab = vec![("<unk>".to_string(), 100.0), ("ab".to_string(), 100.0)];
    let model = Unigram::from(vocab, Some(0), false).unwrap();
    let mut unoptimized = model.clone();
    unoptimized.set_optimized(false);
    let mut scratch = Tokenize::init_scratch(&model);
    let mut lattice_scratch = Tokenize::init_scratch(&unoptimized);

    // Two unknown characters (2 × (100 - 10)) outscore the `ab` entry (100), so the best
    // path is a fused unknown run, and the fused span is `ab` itself.
    assert_eq!(pipeline_ids(&model, "ab", &mut scratch), [1]);
    assert_eq!(pipeline_ids(&unoptimized, "ab", &mut lattice_scratch), [1]);
    assert_eq!(model.encode("ab").unwrap(), ["ab"]);
}

/// A model without an unknown id still encodes text its vocabulary covers. The error belongs to
/// the input the model cannot cover, not to the model.
#[test]
fn a_model_without_an_unknown_id_encodes_covered_text() {
    let vocab = vec![
        ("a".to_string(), 0.0),
        ("b".to_string(), 0.0),
        ("ab".to_string(), 1.0),
    ];
    let model = Unigram::from(vocab, None, false).unwrap();
    let mut scratch = Tokenize::init_scratch(&model);

    assert_eq!(pipeline_ids(&model, "ab", &mut scratch), [2]);
    assert_eq!(model.encode("ab").unwrap(), ["ab"]);
    let mut output = vec![];
    assert!(Tokenize::tokenize(&model, "c", &mut scratch, &mut output).is_err());
}

/// The pre-rewrite contract: `encode`, then look every piece back up in the vocabulary,
/// with byte fallback for pieces that miss. Kept here as a differential oracle for the
/// id-direct optimized path.
fn string_path_ids(model: &Unigram, sequence: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    for piece in model.encode(sequence).unwrap() {
        match model.token_to_id(&piece) {
            Some(id) => ids.push(id),
            None => {
                if model.byte_fallback() {
                    let bytes: Option<Vec<_>> = piece
                        .bytes()
                        .map(|byte| model.token_to_id(&format!("<0x{byte:02X}>")))
                        .collect();
                    if let Some(bytes) = bytes {
                        ids.extend(bytes);
                        continue;
                    }
                }
                ids.push(model.unk_id().unwrap() as u32);
            }
        }
    }
    ids
}

#[test]
fn id_direct_path_matches_the_string_path_on_unknown_and_duplicate_edges() {
    let cases: Vec<(Vocab, Vec<&str>)> = vec![
        (abcd_vocab(), vec!["", "a", "abcd", "xxab", "東京", "abcc"]),
        (
            // The unknown path scores higher than the known "ab", so Viterbi fuses
            // the two unknowns and the fused text still resolves to a real token.
            vec![("<unk>".to_string(), 100.0), ("ab".to_string(), 150.0)],
            vec!["ab", "aab", "abab"],
        ),
        (
            // A fused unknown run expands through byte fallback, not just one byte.
            vec![
                ("<unk>".to_string(), 0.0),
                ("<0xC3>".to_string(), -0.01),
                ("<0xA9>".to_string(), -0.03),
            ],
            vec!["é", "éé", "aé"],
        ),
        (
            // Duplicate tokens resolve to the latest occurrence in both paths.
            vec![
                ("<unk>".to_string(), 0.0),
                ("a".to_string(), 1.0),
                ("a".to_string(), 2.0),
                ("ab".to_string(), 3.0),
            ],
            vec!["a", "ab", "aa", "b"],
        ),
    ];

    for (vocab, sequences) in cases {
        let byte_fallback = vocab.iter().any(|(piece, _)| piece.starts_with("<0x"));
        let model = Unigram::from(vocab, Some(0), byte_fallback).unwrap();
        for sequence in sequences {
            let mut scratch = Tokenize::init_scratch(&model);
            let actual = pipeline_ids(&model, sequence, &mut scratch);
            assert_eq!(actual, string_path_ids(&model, sequence), "{sequence:?}");
        }
    }
}
