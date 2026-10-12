use std::ops::Range;

use super::*;

/// A rendered sequence and the trusted byte ranges in which special tokens may match.
///
/// Ranges refer to the original UTF-8 text, are end-exclusive, and must be sorted,
/// non-empty and non-overlapping. Adjacent ranges are coalesced. All other text
/// bypasses special-token matching in both the raw and normalized passes, while
/// retaining ordinary added tokens. The caller, not the tokenizer, establishes trust.
///
/// Policy boundaries are encoding boundaries: normalization, added-token matching
/// and model encoding run independently on each maximal trusted/untrusted run.
/// Consequently, merges and context-sensitive normalization do not cross them.
/// This controls special-token matching, not which ids the model vocabulary can emit.
///
/// Pass this type, a pair of these inputs, or a vector of either to
/// [`PipelineTokenizer::encode`]. No tokenizer-wide setting is changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeRangeInput {
    text: String,
    safe_ranges: Vec<Range<usize>>,
}

impl SafeRangeInput {
    /// Validate ranges before encoding. An empty list treats the entire text as ordinary.
    /// A single `0..text.len()` range permits special tokens throughout non-empty text.
    pub fn new(
        text: impl Into<String>,
        safe_ranges: impl IntoIterator<Item = Range<usize>>,
    ) -> Result<Self> {
        let text = text.into();
        let mut ranges: Vec<Range<usize>> = Vec::new();
        for (index, range) in safe_ranges.into_iter().enumerate() {
            if range.start >= range.end
                || range.end > text.len()
                || !text.is_char_boundary(range.start)
                || !text.is_char_boundary(range.end)
            {
                return Err(format!(
                    "invalid safe range {index}: {range:?}; expected a non-empty UTF-8 byte range within 0..{}",
                    text.len()
                )
                .into());
            }
            if let Some(previous) = ranges.last_mut() {
                if range.start < previous.end {
                    return Err(format!(
                        "invalid safe range {index}: {range:?}; ranges must be sorted and non-overlapping"
                    )
                    .into());
                }
                if range.start == previous.end {
                    previous.end = range.end;
                    continue;
                }
            }
            ranges.push(range);
        }
        Ok(Self {
            text,
            safe_ranges: ranges,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Validated, coalesced byte ranges. Input text and ranges cannot be mutated independently.
    pub fn safe_ranges(&self) -> &[Range<usize>] {
        &self.safe_ranges
    }
}

impl From<SafeRangeInput> for Inputs {
    fn from(input: SafeRangeInput) -> Self {
        Self::Single(Input::SingleWithRanges(input))
    }
}

impl From<(SafeRangeInput, SafeRangeInput)> for Inputs {
    fn from((a, b): (SafeRangeInput, SafeRangeInput)) -> Self {
        Self::Single(Input::PairWithRanges(a, b))
    }
}

impl From<Vec<SafeRangeInput>> for Inputs {
    fn from(inputs: Vec<SafeRangeInput>) -> Self {
        Self::Batch(inputs.into_iter().map(Input::SingleWithRanges).collect())
    }
}

impl From<Vec<(SafeRangeInput, SafeRangeInput)>> for Inputs {
    fn from(inputs: Vec<(SafeRangeInput, SafeRangeInput)>) -> Self {
        Self::Batch(
            inputs
                .into_iter()
                .map(|(a, b)| Input::PairWithRanges(a, b))
                .collect(),
        )
    }
}

/// A per-call policy override; never mutates or clones the shared vocabulary.
struct TrustedMatcher<'a>(&'a BucketAddedVocabulary);

impl PipelinePatternMatcher for TrustedMatcher<'_> {
    fn extract_next(
        &self,
        input: &[u8],
        search_offset: usize,
        normalized: bool,
    ) -> Option<((usize, usize), u32)> {
        self.0
            .extract_next_with_policy(input, search_offset, normalized, false)
    }
}

impl PipelineTokenizer {
    pub(super) fn encode_safe_sequence_with(
        &self,
        input: &SafeRangeInput,
        scratch: &mut EncodeScratch,
    ) -> Result<Vec<PipelineToken>> {
        let mut output = Vec::with_capacity(input.text.len() / 4);
        let trusted = TrustedMatcher(&self.inner.added_vocabulary);
        let mut cursor = 0;
        for range in &input.safe_ranges {
            if cursor < range.start {
                self.encode_ordinary_run(&input.text[cursor..range.start], scratch, &mut output)?;
            }
            self.encode_sequence_into_with_matcher(
                &input.text[range.clone()],
                scratch,
                &mut output,
                &trusted,
            )?;
            cursor = range.end;
        }
        if cursor < input.text.len() {
            self.encode_ordinary_run(&input.text[cursor..], scratch, &mut output)?;
        }
        Ok(output)
    }

    fn encode_ordinary_run(
        &self,
        text: &str,
        scratch: &mut EncodeScratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let matcher = self
            .inner
            .ordinary_added_vocabulary
            .get_or_init(|| self.inner.added_vocabulary.without_special_tokens());
        self.encode_sequence_into_with_matcher(text, scratch, output, matcher)
    }
}

#[cfg(test)]
// These are lists of policy ranges, never attempts to collect a range's elements.
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use crate::vocab::bucket_added_vocabulary::AddedToken;

    fn pipeline(skip_special: bool) -> PipelineTokenizer {
        let mut vocab = BucketAddedVocabulary::new();
        vocab
            .add_tokens(
                [
                    AddedToken::from("<s>", true),
                    AddedToken::from("<n>", false).normalized(true),
                    AddedToken::from("<added>", false).normalized(false),
                    AddedToken::from("<ns>", true).normalized(true),
                ],
                8,
                |_| None,
                None::<&NormalizerChain>,
            )
            .unwrap();
        vocab.set_encode_special_tokens(skip_special);
        PipelineTokenizer::from_parts(
            vocab,
            vec![PipelineNormalizer::Lowercase(
                crate::normalizers::utils::Lowercase,
            )],
            PipelinePreTokenizer::None,
            PipelineModel::BPE(super::super::tests::hello_bpe()),
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(99.into(), 0)].into_boxed_slice(),
                    ..Template::default()
                },
                pair: Template {
                    prefix: vec![(99.into(), 0)].into_boxed_slice(),
                    infix: vec![(98.into(), 0)].into_boxed_slice(),
                    b_type_id: Some(1),
                    ..Template::default()
                },
            },
            None,
            Default::default(),
            Some(PaddingParams {
                pad_id: 42,
                ..PaddingParams::default()
            }),
        )
    }

    fn input(text: &str, ranges: &[Range<usize>]) -> SafeRangeInput {
        SafeRangeInput::new(text, ranges.iter().cloned()).unwrap()
    }

    fn ids(tk: &PipelineTokenizer, text: &str, ranges: &[Range<usize>]) -> Vec<u32> {
        tk.encode(input(text, ranges), false).wait().unwrap()[0]
            .ids()
            .iter()
            .map(|id| id.id())
            .collect()
    }

    #[test]
    fn safe_ranges_override_policy_only_for_this_input() {
        for policy in [false, true] {
            let tk = pipeline(policy);
            let text = "hello<s>hello<s>";
            let before = tk.encode(text, false).wait().unwrap();
            assert_eq!(ids(&tk, text, &[8..16]), [7, 7, 8]);
            assert_eq!(tk.encode(text, false).wait().unwrap(), before);
            assert_eq!(
                tk.get_added_vocabulary().get_encode_special_tokens(),
                policy
            );
            assert_eq!(ids(&tk, "hello<s><ns>", &[]), [7]);
        }
    }

    #[test]
    fn safe_ranges_can_contain_text_and_multiple_specials() {
        let tk = pipeline(false);
        let text = "hello<s>HELLO<NS><missing>";
        let full = input(text, &[0..text.len()]);
        assert_eq!(
            tk.encode(full, false).wait().unwrap(),
            tk.encode(text, false).wait().unwrap()
        );
        assert_eq!(ids(&tk, "hello<s>HELLO<NS>", &[0..17]), [7, 8, 7, 11]);
    }

    #[test]
    fn safe_ranges_coalesce_adjacent_ranges_before_encoding() {
        let tk = pipeline(false);
        let s = input("hello<s>", &[0..2, 2..6, 6..8]);
        assert_eq!(s.safe_ranges(), &[0..8]);
        assert_eq!(tk.encode(s, false).wait().unwrap()[0].ids(), &[7, 8]);
        assert_eq!(ids(&tk, "hello", &[]), [7]);
    }

    #[test]
    fn safe_ranges_never_match_a_special_across_a_policy_boundary() {
        let tk = pipeline(false);
        for spelling in ["<s>", "<NS>"] {
            for split in 1..spelling.len() {
                for ranges in [vec![0..split], vec![split..spelling.len()]] {
                    let encoded = ids(&tk, spelling, &ranges);
                    assert!(
                        !encoded.contains(&8) && !encoded.contains(&11),
                        "{spelling} {ranges:?}"
                    );
                }
            }
        }
        // A hole in an otherwise trusted token must not become trusted by concatenation.
        assert!(!ids(&tk, "<s>", &[0..1, 2..3]).contains(&8));
    }

    #[test]
    fn safe_ranges_reject_invalid_bounds_without_panicking() {
        for ranges in [
            vec![1..3], // inside a Chinese character
            vec![0..1],
            vec![0..usize::MAX],
            vec![usize::MAX..usize::MAX],
            vec![3..3],
            vec![Range { start: 3, end: 0 }],
            vec![0..6, 3..6],
            vec![3..6, 0..3],
            vec![0..3, 0..3],
        ] {
            assert!(SafeRangeInput::new("你好", ranges).is_err());
        }
        assert!(SafeRangeInput::new("", [0..0]).is_err());
        assert!(SafeRangeInput::new("", []).is_ok());
        assert_eq!(input("你好<s>", &[6..9]).text(), "你好<s>");
        assert_eq!(ids(&pipeline(false), "你好<s>", &[6..9]), [8]);
    }

    #[test]
    fn safe_ranges_keep_original_offsets_when_normalization_changes_length() {
        let mut tk = pipeline(false);
        Arc::get_mut(&mut tk.inner).unwrap().normalizers.insert(
            0,
            PipelineNormalizer::Replace(
                crate::normalizers::replace::Replace::new("x", "hello").unwrap(),
            ),
        );
        assert_eq!(ids(&tk, "x<NS><NS>", &[5..9]), [7, 11]);
        // A normalizer-generated special in untrusted text remains ordinary.
        Arc::get_mut(&mut tk.inner).unwrap().normalizers.insert(
            0,
            PipelineNormalizer::Replace(
                crate::normalizers::replace::Replace::new("y", "<NS>").unwrap(),
            ),
        );
        assert_eq!(ids(&tk, "y<NS>", &[1..5]), [11]);
    }

    #[test]
    fn safe_ranges_preserve_ordinary_added_tokens_in_both_passes() {
        let tk = pipeline(false);
        assert_eq!(ids(&tk, "hello<added><N><NS><s>", &[]), [7, 10, 9]);
    }

    #[test]
    fn safe_ranges_excluded_specials_do_not_hide_overlapping_ordinary_tokens() {
        let mut tk = pipeline(false);
        Arc::get_mut(&mut tk.inner)
            .unwrap()
            .added_vocabulary
            .add_tokens(
                [
                    AddedToken::from("<outer<added>>", true),
                    AddedToken::from("<added>suffix", true),
                    AddedToken::from("<outer<n>>", true).normalized(true),
                ],
                8,
                |_| None,
                None::<&NormalizerChain>,
            )
            .unwrap();
        for (text, id) in [
            ("<outer<added>>", 10),
            ("<added>suffix", 10),
            ("<outer<n>>", 9),
        ] {
            let encoded = ids(&tk, text, &[]);
            assert!(encoded.contains(&id));
            assert!(encoded.iter().all(|id| *id < 12));
        }
    }

    #[test]
    fn safe_ranges_apply_added_token_flags_within_each_run() {
        let mut tk = pipeline(false);
        Arc::get_mut(&mut tk.inner)
            .unwrap()
            .added_vocabulary
            .add_tokens(
                [
                    AddedToken::from("é", false)
                        .normalized(false)
                        .single_word(true),
                    AddedToken::from("<strip>", true).lstrip(true).rstrip(true),
                    AddedToken::from("word", true).single_word(true),
                ],
                8,
                |_| None,
                None::<&NormalizerChain>,
            )
            .unwrap();
        assert_eq!(
            ids(&tk, "éx é", &[]).iter().filter(|&&id| id == 12).count(),
            1
        );
        assert!(!ids(&tk, "éé", &[]).contains(&12));
        // Stripping cannot consume ordinary text on either side of a trusted run.
        assert_eq!(ids(&tk, "hello <strip> hello", &[6..13]), [7, 13, 7]);
        assert!(!ids(&tk, "sword", &[0..5]).contains(&14));
        assert_eq!(ids(&tk, "word", &[0..4]), [14]);
    }

    #[test]
    fn safe_ranges_post_process_once_and_pad_in_input_order() {
        let tk = pipeline(false);
        let batch = vec![input("hello<s>", &[5..8]), input("", &[])];
        let encoded = tk.encode(batch.clone(), true).wait().unwrap();
        assert_eq!(encoded[0].ids(), &[99, 7, 8]);
        assert_eq!(encoded[1].ids(), &[99, 42, 42]);
        assert_eq!(encoded[1].attention_mask(), Some([1, 0, 0].as_slice()));
        let encoded = tk.encode(batch, false).wait_with_padding(None).unwrap();
        assert_eq!(encoded[0].ids(), &[7, 8]);
        assert!(encoded[1].ids().is_empty());
        assert!(
            tk.encode(Vec::<SafeRangeInput>::new(), false)
                .wait()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn safe_ranges_pair_uses_independent_coordinates_and_type_ids() {
        let tk = pipeline(false);
        let pair = (input("hello<s>", &[5..8]), input("<s>hello", &[]));
        let encoded = tk.encode(pair.clone(), true).wait().unwrap();
        assert_eq!(encoded[0].ids(), &[99, 7, 8, 98, 7]);
        assert_eq!(encoded[0].type_ids(), Some([0, 0, 0, 0, 1].as_slice()));
        assert_eq!(tk.encode(vec![pair], true).wait().unwrap(), encoded);
    }

    #[test]
    fn safe_ranges_concurrent_clones_do_not_change_plain_encoding() {
        let tk = pipeline(false);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let tk = tk.clone();
                scope.spawn(move || {
                    for _ in 0..20 {
                        assert_eq!(ids(&tk, "hello<s><s>", &[8..11]), [7, 8]);
                        assert_eq!(
                            tk.encode("hello<s><s>", false).wait().unwrap()[0].ids(),
                            &[7, 8, 8]
                        );
                    }
                });
            }
        });
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn safe_ranges_streaming_batch_matches_serial_for_large_mixed_inputs() {
        let tk = pipeline(false);
        let text = "hello<s>".repeat(4096);
        let restricted = input(&text, &[text.len() - 3..text.len()]);
        let inputs = Inputs::Batch(vec![
            Input::SingleWithRanges(restricted.clone()),
            Input::PairWithRanges(restricted, input("", &[])),
            Input::SingleWithRanges(input("", &[])),
            Input::Single("hello<s>".into()),
        ]);
        let serial = tk
            .encode_serial(inputs.clone(), true)
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let handle = tk.encode(inputs, true);
        if crate::utils::parallelism::pool().is_some() {
            assert!(matches!(&handle.state, HandleState::Streaming(_)));
        }
        let encoded = handle.wait_with_padding(None).unwrap();
        assert_eq!(encoded, serial);
        assert_eq!(encoded[0].ids().iter().filter(|id| **id == 8).count(), 1);
        assert_eq!(encoded[1].ids().iter().filter(|id| **id == 8).count(), 1);
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn safe_ranges_batch_model_errors_are_isolated() {
        let mut tk = pipeline(false);
        Arc::get_mut(&mut tk.inner).unwrap().model = PipelineModel::WordLevel(
            crate::models::wordlevel::WordLevel::builder()
                .vocab([("hello".to_owned(), 7)].into_iter().collect())
                .build()
                .unwrap(),
        );
        let bad = "missing".repeat(4096);
        let good = "<s>".repeat(4096);
        let mut results: Vec<_> = tk
            .encode(
                vec![input(&bad, &[]), input(&good, &[0..good.len()])],
                false,
            )
            .into_iter()
            .collect();
        results.sort_by_key(|(index, _)| *index);
        assert!(results[0].1.is_err());
        let good = results[1].1.as_ref().unwrap().ids();
        assert_eq!(good.len(), 4096);
        assert!(good.iter().all(|id| *id == 8));
    }
}
