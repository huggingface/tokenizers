use crate::pipeline::Model;
use crate::tokenizer::Token;

use super::*;
#[test]
fn test_populate_nodes_unk() {
    let pieces = vec![("<unk>".to_string(), 0.0)];
    let model = Unigram::from(pieces, Some(0), false).unwrap();

    let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
    model.populate_nodes(&mut lattice);

    assert_eq!(lattice.begin_nodes[0].len(), 1);
    assert_eq!(lattice.begin_nodes[1].len(), 1);
    assert_eq!(lattice.begin_nodes[2].len(), 1);
    assert_eq!(lattice.begin_nodes[0][0].borrow().id, 0);
    assert_eq!(lattice.begin_nodes[1][0].borrow().id, 0);
    assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
    assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
    assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 3);
    assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 4);
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
    assert_eq!(lattice.begin_nodes[0][0].borrow().id, 1);
    assert_eq!(lattice.begin_nodes[0][1].borrow().id, 3);
    assert_eq!(lattice.begin_nodes[1][0].borrow().id, 2);
    assert_eq!(lattice.begin_nodes[1][1].borrow().id, 4);
    assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
    assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
    assert_eq!(lattice.begin_nodes[0][1].borrow().node_id, 3);
    assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 4);
    assert_eq!(lattice.begin_nodes[1][1].borrow().node_id, 5);
    assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 6);
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
    let tokens: Vec<Token> = unigram.tokenize("é").unwrap();
    assert_eq!(
        tokens,
        [
            Token {
                id: 1,
                value: "<0xC3>".to_string(),
                offsets: (0, 2)
            },
            Token {
                id: 2,
                value: "<0xA9>".to_string(),
                offsets: (0, 2)
            }
        ]
    );

    let tokens = unigram.tokenize("?é").unwrap();
    assert_eq!(tokens[0].id, 0);
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
    Model::tokenize_pipeline(model, sequence, scratch, &mut output).unwrap();
    output.iter().map(|token| token.id()).collect()
}

#[test]
fn pipeline_remembers_what_a_sequence_encoded_to() {
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    let mut scratch = Model::init_scratch(&model);

    let ids = pipeline_ids(&model, "abcd", &mut scratch);

    let cache = scratch
        .word_cache
        .as_mut()
        .expect("Unigram encodes with a cache");
    assert_eq!(cache.lookup(b"abcd").hit(), Some(&ids[..]));
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

    let mut warm_scratch = Model::init_scratch(&model);
    let warm = corpus.map(|sequence| pipeline_ids(&model, sequence, &mut warm_scratch));
    let cold = corpus.map(|sequence| {
        let mut scratch = Model::init_scratch(&model);
        pipeline_ids(&model, sequence, &mut scratch)
    });

    assert_eq!(warm, cold);
}

#[test]
fn caches_only_the_ids_this_sequence_produced() {
    // Every sequence the pipeline hands the model appends to one output buffer,
    // so a sequence has to remember its own ids, not everything the buffer holds.
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    let mut scratch = Model::init_scratch(&model);
    let mut output = vec![];
    Model::tokenize_pipeline(&model, "ab", &mut scratch, &mut output).unwrap();
    Model::tokenize_pipeline(&model, "cd", &mut scratch, &mut output).unwrap();

    let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
    assert_eq!(ids, [6, 5]);
    let cache = scratch.word_cache.as_mut().unwrap();
    assert_eq!(cache.lookup(b"cd").hit(), Some(&[5u32][..]));
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
    let mut scratch = Model::init_scratch(&model);

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
    let mut scratch = Model::init_scratch(&model);

    pipeline_ids(&model, "abcd", &mut scratch);

    let cache = scratch.word_cache.as_mut().unwrap();
    assert_eq!(cache.lookup(b"abcd").hit(), None);
}

#[test]
fn a_capacity_of_zero_turns_the_cache_off() {
    let mut model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    model.resize_cache(0);
    let mut scratch = Model::init_scratch(&model);

    assert_eq!(pipeline_ids(&model, "abcd", &mut scratch), [8]);
    assert!(scratch.word_cache.is_none());
}
