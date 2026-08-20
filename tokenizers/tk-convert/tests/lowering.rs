//! The config -> pipeline lowering, tested against the reference `Tokenizer` it lowers from.
//!
//! Moved out of `tk-encode`'s `pipeline` module by the `tk-convert` split: every case here
//! builds a `Tokenizer`, converts it, and compares — so it needs both crates. What stayed behind
//! are the two cases that never name a wrapper (the span-safety sweep over every
//! `PipelinePreTokenizer`, and the special-segment iterator).
//!
//! The three groups are: the conversion *rejections* (configs the pipeline would otherwise encode
//! with silently wrong ids), the post-processor frames, and the parallel-equals-serial differential.

use std::convert::TryFrom;

use tk_convert::models::bpe::BPE;
use tk_convert::pre_tokenizers::Sequence;
use tk_convert::processors::sequence::Sequence as ProcessorSequence;
use tk_convert::{PostProcessorWrapper, PreTokenizerWrapper, Tokenizer};
use tk_encode::models::wordpiece::WordPiece;
use tk_encode::pipeline::{Encoding, PipelineTokenizer};
use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::pre_tokenizers::whitespace::WhitespaceSplit;

// Only the no-regex-backend case names a normalizer wrapper directly.
#[cfg(not(feature = "fancy-regex"))]
use tk_convert::NormalizerWrapper;

#[cfg(feature = "parallelism")]
use tk_encode::pipeline::{Inputs, PARALLEL_MIN_BYTES};
#[cfg(feature = "parallelism")]
use tk_encode::utils::parallelism::set_num_threads;

/// An empty normalizer `Sequence` means "no normalization" (deepseek ships one). It must be
/// dropped on the way in, not carried as a no-op call per segment. A non-empty one is kept.
#[test]
fn empty_normalizer_sequence_is_elided() {
    use tk_convert::normalizers::Sequence as NormSequence;
    use tk_encode::normalizers::utils::Lowercase;

    let vocab = vec![("[UNK]", 0u32), ("hello", 1)];

    let mut tok = wordlevel_tokenizer(vocab.clone(), None);
    tok.with_normalizer(Some(NormSequence::new(vec![])))
        .unwrap();
    assert!(
        !PipelineTokenizer::try_from(&tok).unwrap().has_normalizer(),
        "an empty normalizer Sequence should not survive into the pipeline"
    );

    let mut tok = wordlevel_tokenizer(vocab, None);
    tok.with_normalizer(Some(NormSequence::new(vec![Lowercase.into()])))
        .unwrap();
    assert!(
        PipelineTokenizer::try_from(&tok).unwrap().has_normalizer(),
        "a non-empty normalizer Sequence must still be applied"
    );
}

/// Test the literal only replace and splits can be run without the fancy-regex feature
#[cfg(not(feature = "fancy-regex"))]
#[test]
fn string_pattern_config_loads_and_encodes_with_no_regex_backend() {
    let normalizer: NormalizerWrapper =
        serde_json::from_str(r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#)
            .unwrap();
    let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
        r#"{"type":"Split","pattern":{"String":"▁"},"behavior":"MergedWithPrevious","invert":false}"#,
    )
    .unwrap();

    let mut tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello▁", 1), ("world", 2)], None);
    tok.with_normalizer(Some(normalizer)).unwrap();
    tok.with_pre_tokenizer(Some(pre_tokenizer));

    let encoded = PipelineTokenizer::try_from(&tok)
        .unwrap()
        .encode("hello world", false)
        .wait()
        .unwrap();
    // Not the unk id: both the `Replace` and the `Split` really ran on the literal path.
    assert_eq!(*encoded.first().unwrap().ids(), [1, 2]);
    assert_pipeline_matches_reference(&tok, "hello world");
}

// The three rejections below guard configs the pipeline would otherwise
// encode with silently wrong ids (the byte-level vocab transform only
// applies when ByteLevel is the model's direct input). Each test pins the
// error message so an unrelated failure can't stand in for the guard.

fn conversion_error(tok: &Tokenizer) -> String {
    PipelineTokenizer::try_from(tok).err().unwrap().to_string()
}

#[test]
fn conversion_rejects_nested_sequence() {
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(Sequence::new(vec![PreTokenizerWrapper::Sequence(
        Sequence::new(vec![PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit)]),
    )])));
    let err = conversion_error(&tok);
    assert!(err.contains("Nesting Sequence"), "{}", err);
}

#[test]
fn conversion_rejects_byte_level_not_last_in_sequence() {
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true)),
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
    ])));
    let err = conversion_error(&tok);
    assert!(err.contains("must be the last"), "{}", err);
}

#[test]
fn conversion_rejects_byte_level_with_non_bpe_model() {
    let mut tok = Tokenizer::new(WordPiece::default());
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    let err = conversion_error(&tok);
    assert!(err.contains("not supported with model"), "{}", err);
}

fn wordlevel_tokenizer(
    vocab: Vec<(&str, u32)>,
    post_processor: Option<PostProcessorWrapper>,
) -> Tokenizer {
    use tk_encode::models::wordlevel::WordLevel;
    use tk_encode::pre_tokenizers::whitespace::Whitespace;

    let unk = vocab[0].0.to_string();
    let vocab: ahash::AHashMap<String, u32> =
        vocab.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk)
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok.with_post_processor(post_processor);
    tok
}

#[cfg(feature = "parallelism")]
fn pipeline_ids(pipeline: &PipelineTokenizer, input: &str) -> Vec<u32> {
    pipeline
        .encode(input, false)
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect()
}

// A single `&self` tokenizer is meant to be shared across rayon workers, and the scratch it
// hands each of them carries the pre-token spans of the chunk being encoded. So the workers
// must not be able to reach each other's: every thread encodes a different input here, and
// has to get back the answer that input produces on its own.
//
// The inputs disagree on both how many tokens they produce and which, so another input's
// spans cannot yield the right ids by luck. This also only compiles if
// `PipelineTokenizer: Sync`.
#[cfg(feature = "parallelism")]
#[test]
fn concurrent_encodes_of_different_inputs_stay_independent() {
    use rayon::prelude::*;

    let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();

    let inputs = [
        "hello".to_string(),
        "hello world".to_string(),
        "world hello world".to_string(),
        "hello world ".repeat(25),
    ];
    let want: Vec<Vec<u32>> = inputs
        .iter()
        .map(|input| pipeline_ids(&pipeline, input))
        .collect();
    assert_eq!(
        want,
        vec![vec![1], vec![1, 2], vec![2, 1, 2], [1, 2].repeat(25)]
    );

    let all_match = (0..10_000usize).into_par_iter().all(|i| {
        let case = i % inputs.len();
        pipeline_ids(&pipeline, &inputs[case]) == want[case]
    });
    assert!(all_match);
}

// Every caller compares a WordLevel-backed tokenizer against the reference path.
fn assert_pipeline_matches_reference(tok: &Tokenizer, input: &str) {
    let pipeline = PipelineTokenizer::try_from(tok).unwrap();
    for add_special_tokens in [false, true] {
        let expected = tok
            .encode(input, add_special_tokens)
            .unwrap()
            .get_ids()
            .to_vec();
        let got: Vec<u32> = pipeline
            .encode(input, add_special_tokens)
            .wait()
            .unwrap()
            .first()
            .unwrap()
            .ids()
            .iter()
            .map(|t| t.id())
            .collect();
        assert_eq!(expected, got, "add_special_tokens={add_special_tokens}");
    }
}

#[test]
fn pipeline_runs_bert_post_processor_matching_reference() {
    use tk_encode::processors::bert::BertProcessing;

    let tok = wordlevel_tokenizer(
        vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Bert(BertProcessing::new(
            ("[SEP]".to_string(), 1),
            ("[CLS]".to_string(), 0),
        ))),
    );
    assert_pipeline_matches_reference(&tok, "hello world");

    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let ids = |enc: Vec<Encoding>| {
        enc.first()
            .unwrap()
            .ids()
            .iter()
            .map(|t| t.id())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ids(pipeline.encode("hello world", true).wait().unwrap()),
        vec![0, 2, 3, 1]
    );
    assert_eq!(
        ids(pipeline.encode("hello world", false).wait().unwrap()),
        vec![2, 3]
    );
}

#[test]
fn pipeline_runs_roberta_post_processor_matching_reference() {
    use tk_encode::processors::roberta::RobertaProcessing;

    let tok = wordlevel_tokenizer(
        vec![("<s>", 0), ("</s>", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
            ("</s>".to_string(), 1),
            ("<s>".to_string(), 0),
        ))),
    );
    assert_pipeline_matches_reference(&tok, "hello world");
}

#[test]
fn pipeline_runs_template_post_processor_matching_reference() {
    use tk_encode::processors::template::TemplateProcessing;

    let tok = wordlevel_tokenizer(
        vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Template(
            TemplateProcessing::builder()
                .try_single("[CLS] $0 [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 0u32), ("[SEP]", 1u32)])
                .build()
                .unwrap(),
        )),
    );
    assert_pipeline_matches_reference(&tok, "hello world");
}

#[test]
fn pipeline_bytelevel_post_processor_is_noop() {
    use tk_encode::pre_tokenizers::byte_level::ByteLevel;
    let tok = wordlevel_tokenizer(
        vec![("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::ByteLevel(ByteLevel::default())),
    );
    assert_pipeline_matches_reference(&tok, "hello world");
}

#[test]
fn conversion_rejects_sequence_of_two_special_adding_members() {
    use tk_encode::processors::template::TemplateProcessing;

    // Both members have identity cores (`$A $B`) but add their own special tokens. The
    // reference retags the inner member's output when the outer wraps it, which a static
    // composed template cannot represent, so this must be rejected (not silently miscompiled).
    // Also guards that pass-through detection looks at the whole template, not just its core.
    let member = |prefix: &str, suffix: &str, p_id: u32, s_id: u32| {
        TemplateProcessing::builder()
            .try_single(format!("{prefix} $A {suffix}"))
            .unwrap()
            .try_pair(format!("{prefix} $A $B:1 {suffix}:1"))
            .unwrap()
            .special_tokens(vec![(prefix, p_id), (suffix, s_id)])
            .build()
            .unwrap()
    };
    let tok = wordlevel_tokenizer(
        vec![
            ("[X]", 100),
            ("[Y]", 101),
            ("[P]", 102),
            ("[Q]", 103),
            ("hello", 2),
            ("world", 3),
        ],
        Some(PostProcessorWrapper::Sequence(ProcessorSequence::new(
            vec![
                PostProcessorWrapper::Template(member("[X]", "[Y]", 100, 101)),
                PostProcessorWrapper::Template(member("[P]", "[Q]", 102, 103)),
            ],
        ))),
    );
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[test]
fn conversion_rejects_sequence_with_two_arranging_members() {
    use tk_encode::processors::bert::BertProcessing;

    let tok = wordlevel_tokenizer(
        vec![("A", 100), ("B", 101), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Sequence(ProcessorSequence::new(
            vec![
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("B".to_string(), 101),
                    ("A".to_string(), 100),
                )),
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("B".to_string(), 101),
                    ("A".to_string(), 100),
                )),
            ],
        ))),
    );
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[test]
fn roberta_pair_without_specials_keeps_type_ids_zero() {
    use tk_encode::processors::roberta::RobertaProcessing;

    // RoBERTa tags both pair sides type 0. `add_special_tokens = false` must suppress only the
    // special tokens, not fall back to the default A=0/B=1 tagging.
    let tok = wordlevel_tokenizer(
        vec![
            ("<s>", 0),
            ("</s>", 1),
            ("hello", 2),
            ("world", 3),
            ("foo", 4),
        ],
        Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
            ("</s>".to_string(), 1),
            ("<s>".to_string(), 0),
        ))),
    );
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let batch = pipeline
        .encode(("hello world", "foo"), false)
        .wait()
        .unwrap();
    let enc = batch.first().unwrap();

    assert!(
        enc.type_ids().is_none_or(|t| t.iter().all(|&x| x == 0)),
        "expected all-zero type ids, got {:?}",
        enc.type_ids()
    );
    let expected = tok.encode(("hello world", "foo"), false).unwrap();
    let ids: Vec<u32> = enc.ids().iter().map(|t| t.id()).collect();
    assert_eq!(expected.get_ids(), ids.as_slice());
}

#[test]
fn sequence_keeps_reordering_member_core() {
    use tk_encode::pre_tokenizers::byte_level::ByteLevel;
    use tk_encode::processors::template::TemplateProcessing;

    // ByteLevel has an identity core (safe to drop); the template reorders the pair to `$B $A`.
    // Compose must keep the reordering core, not discard it as trivial.
    let reorder = TemplateProcessing::builder()
        .try_single("$A")
        .unwrap()
        .try_pair("$B $A")
        .unwrap()
        .build()
        .unwrap();
    let tok = wordlevel_tokenizer(
        vec![("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Sequence(ProcessorSequence::new(
            vec![
                PostProcessorWrapper::ByteLevel(ByteLevel::default()),
                PostProcessorWrapper::Template(reorder),
            ],
        ))),
    );
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let batch = pipeline.encode(("hello", "world"), false).wait().unwrap();
    let ids: Vec<u32> = batch
        .first()
        .unwrap()
        .ids()
        .iter()
        .map(|t| t.id())
        .collect();
    // `$B $A` => world (3) before hello (2)
    assert_eq!(ids, vec![3, 2]);
}

#[test]
fn pipeline_sequence_bytelevel_then_template_matches_reference() {
    use tk_encode::pre_tokenizers::byte_level::ByteLevel;
    use tk_encode::processors::template::TemplateProcessing;

    let tok = wordlevel_tokenizer(
        vec![("<|begin_of_text|>", 0), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Sequence(ProcessorSequence::new(
            vec![
                PostProcessorWrapper::ByteLevel(ByteLevel::default()),
                PostProcessorWrapper::Template(
                    TemplateProcessing::builder()
                        .try_single("<|begin_of_text|> $0")
                        .unwrap()
                        .special_tokens(vec![("<|begin_of_text|>", 0u32)])
                        .build()
                        .unwrap(),
                ),
            ],
        ))),
    );
    assert_pipeline_matches_reference(&tok, "hello world");
}

#[test]
fn conversion_rejects_template_referencing_sequence_twice() {
    use tk_encode::processors::template::TemplateProcessing;

    let tok = wordlevel_tokenizer(
        vec![("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Template(
            TemplateProcessing::builder()
                .try_single("$0 $0")
                .unwrap()
                .build()
                .unwrap(),
        )),
    );
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[test]
fn conversion_rejects_template_without_sequence_piece() {
    use tk_encode::processors::template::TemplateProcessing;

    let tok = wordlevel_tokenizer(
        vec![("[CLS]", 0), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Template(
            TemplateProcessing::builder()
                .try_single("[CLS]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 0u32)])
                .build()
                .unwrap(),
        )),
    );
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[test]
fn conversion_rejects_template_with_unknown_special_token() {
    // Deserializing straight from JSON skips `TemplateProcessingBuilder::validate`,
    // so this (unlike the builder) can reach the pipeline with a dangling reference.
    let json = r#"{
        "type":"TemplateProcessing",
        "single":[
            {"SpecialToken":{"id":"[CLS]","type_id":0}},
            {"Sequence":{"id":"A","type_id":0}}
        ],
        "pair":[{"Sequence":{"id":"A","type_id":0}}],
        "special_tokens":{}
    }"#;
    // `TemplateProcessing` carries no serde of its own any more -- `processors::mirror` owns the
    // shape -- so this goes in through the wrapper. Being untagged, it lands on the `Template`
    // variant for exactly the same input, and by the same tag-less-tolerant route.
    let processor: PostProcessorWrapper = serde_json::from_str(json).unwrap();
    assert!(matches!(processor, PostProcessorWrapper::Template(_)));

    let tok = wordlevel_tokenizer(vec![("hello", 2), ("world", 3)], Some(processor));
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[test]
fn conversion_rejects_sequence_containing_unsupported_member() {
    use tk_encode::processors::template::TemplateProcessing;

    let tok = wordlevel_tokenizer(
        vec![("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Sequence(ProcessorSequence::new(
            vec![PostProcessorWrapper::Template(
                TemplateProcessing::builder()
                    .try_single("$0 $0")
                    .unwrap()
                    .build()
                    .unwrap(),
            )],
        ))),
    );
    let err = conversion_error(&tok);
    assert!(err.contains("not supported"), "{}", err);
}

#[cfg(feature = "parallelism")]
use std::sync::{Mutex, PoisonError};
#[cfg(feature = "parallelism")]
static LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "parallelism")]
fn assert_parallel_matches(
    tokenizer: &PipelineTokenizer,
    inputs: Inputs,
    add_special_tokens: bool,
) {
    let _g = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_num_threads(1);
    let serial = tokenizer
        .encode(inputs.clone(), add_special_tokens)
        .wait()
        .unwrap();
    for n in [2, 4, 8] {
        set_num_threads(n);
        for _ in 0..3 {
            let par = tokenizer
                .encode(inputs.clone(), add_special_tokens)
                .wait()
                .unwrap();
            assert_eq!(par, serial);
        }
    }
    set_num_threads(0);
}

#[cfg(feature = "parallelism")]
fn repeat_to(phrase: &str, min_bytes: usize) -> String {
    phrase.repeat(min_bytes / phrase.len() + 1)
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_batch_identity() {
    let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
    for add in [false, true] {
        assert_parallel_matches(&pipeline, inputs.clone(), add);
    }
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_batch_bert() {
    use tk_encode::processors::bert::BertProcessing;
    let tok = wordlevel_tokenizer(
        vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Bert(BertProcessing::new(
            ("[SEP]".to_string(), 1),
            ("[CLS]".to_string(), 0),
        ))),
    );
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
    for add in [false, true] {
        assert_parallel_matches(&pipeline, inputs.clone(), add);
    }
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_pairs_bert() {
    use tk_encode::processors::bert::BertProcessing;
    let tok = wordlevel_tokenizer(
        vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Bert(BertProcessing::new(
            ("[SEP]".to_string(), 1),
            ("[CLS]".to_string(), 0),
        ))),
    );
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec![
        ("hello world".to_string(), "world hello".to_string());
        700
    ]);
    for add in [false, true] {
        assert_parallel_matches(&pipeline, inputs.clone(), add);
    }
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_batch_roberta() {
    use tk_encode::processors::roberta::RobertaProcessing;
    let tok = wordlevel_tokenizer(
        vec![("<s>", 0), ("</s>", 1), ("hello", 2), ("world", 3)],
        Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
            ("</s>".to_string(), 1),
            ("<s>".to_string(), 0),
        ))),
    );
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
    for add in [false, true] {
        assert_parallel_matches(&pipeline, inputs.clone(), add);
    }
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_long_single_with_specials() {
    let mut tok = wordlevel_tokenizer(
        vec![("<unk>", 0), ("hello", 1), ("world", 2), ("<sep>", 3)],
        None,
    );
    tok.add_special_tokens([tk_convert::AddedToken::from("<sep>", true)])
        .unwrap();
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(repeat_to(
        "hello world <sep> ",
        2 * PARALLEL_MIN_BYTES + 4096,
    ));
    for add in [false, true] {
        assert_parallel_matches(&pipeline, inputs.clone(), add);
    }
}

#[cfg(feature = "parallelism")]
#[test]
fn parallel_matches_serial_mixed_batch_with_edges() {
    let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let mut batch = vec![
        String::new(),
        "hello".to_string(),
        "hello world".to_string(),
    ];
    batch.extend(vec!["hello world".to_string(); 1000]);
    assert_parallel_matches(&pipeline, Inputs::from(batch), false);
}

#[cfg(feature = "parallelism")]
#[test]
fn streaming_iterator_yields_each_seq_once() {
    let _g = LOCK.lock().unwrap();
    let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);

    set_num_threads(1);
    let serial = pipeline.encode(inputs.clone(), false).wait().unwrap();

    set_num_threads(4);
    let mut streamed: Vec<Option<Encoding>> = vec![None; serial.len()];
    for (seq, res) in pipeline.encode(inputs, false) {
        assert!(
            streamed[seq].is_none(),
            "seq {seq} was yielded more than once"
        );
        streamed[seq] = Some(res.unwrap());
    }
    set_num_threads(0);

    let streamed: Vec<Encoding> = streamed
        .into_iter()
        .map(|e| e.expect("a seq was never yielded"))
        .collect();
    assert_eq!(streamed, serial);
}

#[cfg(feature = "parallelism")]
#[test]
fn streaming_handle_drop_after_partial_consume_is_clean() {
    let _g = LOCK.lock().unwrap();
    let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);

    set_num_threads(4);
    let mut it = pipeline.encode(inputs, false).into_iter();
    assert!(it.next().is_some());
    assert!(it.next().is_some());
    drop(it);
    set_num_threads(0);
}
