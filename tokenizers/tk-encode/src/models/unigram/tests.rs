use crate::pipeline::Model;

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
    let model = Unigram::from(abcd_vocab(), Some(0), false).unwrap();
    assert_eq!(pipeline_pieces(&model, "abcd"), ["abcd"]);
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

    for is_optimized in [true, false] {
        model.set_optimized(is_optimized);
        assert_eq!(pipeline_pieces(&model, "abc"), ["abc"]);
        assert_eq!(pipeline_pieces(&model, "AB"), ["<unk>"]);

        model.set_fuse_unk(false);
        assert_eq!(pipeline_pieces(&model, "AB"), ["<unk>", "<unk>"]);
        model.set_fuse_unk(true);
        assert_eq!(pipeline_pieces(&model, "AB"), ["<unk>"]);

        assert_eq!(pipeline_pieces(&model, "abcd"), ["ab", "cd"]);
        assert_eq!(pipeline_pieces(&model, "abcc"), ["abc", "c"]);
        assert_eq!(
            pipeline_pieces(&model, "xabcabaabcdd"),
            ["<unk>", "abc", "ab", "a", "ab", "cd", "<unk>"]
        );
        model.set_fuse_unk(false);
        assert_eq!(pipeline_pieces(&model, "xyz東京"), ["<unk>"; 5]);
        model.set_fuse_unk(true);
        assert_eq!(pipeline_pieces(&model, "xyz東京"), ["<unk>"]);

        // User encoded in original version
        assert_eq!(pipeline_pieces(&model, "ABC"), ["ABC"]);
        assert_eq!(pipeline_pieces(&model, "abABCcd"), ["ab", "ABC", "cd"]);
        assert_eq!(
            pipeline_pieces(&model, "ababcdabcdcd"),
            ["ab", "abcdabcd", "cd"]
        );
        assert_eq!(pipeline_pieces(&model, "abqrcd"), ["ab", "q", "r", "cd"]);
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
    assert_eq!(pipeline_pieces(&unigram, "é"), ["<0xC3>", "<0xA9>"]);

    // `?` has no byte token, so the fused unknown piece falls back to `<unk>` as a whole.
    assert_eq!(pipeline_pieces(&unigram, "?é"), ["<unk>"]);
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

/// A fresh scratch every call: the tests flip `fuse_unk` and `is_optimized` between calls, and a
/// warm cache would replay what the previous setting produced.
fn pipeline_pieces(model: &Unigram, sequence: &str) -> Vec<String> {
    let mut scratch = Model::init_scratch(model);
    pipeline_ids(model, sequence, &mut scratch)
        .into_iter()
        .map(|id| model.id_to_token(id).unwrap())
        .collect()
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
