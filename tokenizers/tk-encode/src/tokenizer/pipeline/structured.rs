use super::*;

/// A producer-declared segment. `special` requires an exact registered special-token spelling;
/// ordinary content bypasses special-token matching, but still recognizes non-special added tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeSegment {
    pub content: String,
    pub special: bool,
}

impl EncodeSegment {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            special: false,
        }
    }

    pub fn special(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            special: true,
        }
    }
}

struct OrdinaryMatcher<'a>(&'a BucketAddedVocabulary);

impl PipelinePatternMatcher for OrdinaryMatcher<'_> {
    fn extract_next(
        &self,
        input: &[u8],
        offset: usize,
        normalized: bool,
    ) -> Option<((usize, usize), u32)> {
        self.0
            .extract_next_with_policy(input, offset, normalized, true)
    }
}

impl PipelineTokenizer {
    /// Encode explicit text/special segments, then apply the normal post-processor and padding.
    /// Adjacent ordinary segments are coalesced before normalization and model encoding.
    /// This does not change tokenizer-wide special-token policy. Like rc0's ordinary encoding,
    /// this API returns ids and masks, without offsets or truncation.
    ///
    /// Work completes before this method returns; consume the handle with `wait` to apply padding.
    pub fn encode_segments(
        &self,
        segments: &[EncodeSegment],
        add_special_tokens: bool,
    ) -> EncodeHandle {
        EncodeHandle::blocking(
            vec![self.encode_segments_one(segments, add_special_tokens)],
            self.inner.padding.clone(),
        )
    }

    /// Encode a batch of segmented inputs in input order. Uses rc0's configured worker pool when
    /// available; unlike `encode`, the returned handle is already computed rather than streaming.
    pub fn encode_segments_batch(
        &self,
        batch: &[Vec<EncodeSegment>],
        add_special_tokens: bool,
    ) -> EncodeHandle {
        #[cfg(feature = "parallelism")]
        if let Some(pool) = crate::utils::parallelism::pool() {
            use rayon::prelude::*;
            let results = pool.install(|| {
                batch
                    .par_iter()
                    .map(|segments| self.encode_segments_one(segments, add_special_tokens))
                    .collect()
            });
            return EncodeHandle::blocking(results, self.inner.padding.clone());
        }
        EncodeHandle::blocking(
            batch
                .iter()
                .map(|segments| self.encode_segments_one(segments, add_special_tokens))
                .collect(),
            self.inner.padding.clone(),
        )
    }

    fn encode_segments_one(
        &self,
        segments: &[EncodeSegment],
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // Validate every trusted segment before doing any model work. The exact stored spelling is
        // used for both the raw and normalized added vocabularies.
        let ids: Vec<_> = segments
            .iter()
            .map(|segment| {
                if !segment.special {
                    return Ok(None);
                }
                self.inner
                    .added_vocabulary
                    .special_token_id(&segment.content)
                    .map(Some)
                    .ok_or_else(|| -> crate::Error {
                        format!(
                            "Structured special segment is not a registered special token: {:?}",
                            segment.content
                        )
                        .into()
                    })
            })
            .collect::<Result<_>>()?;
        let mut scratch = self.inner.scratch_pool.get(&self.inner.model);
        let mut output = Vec::new();
        let mut ordinary = String::new();
        let matcher = OrdinaryMatcher(&self.inner.added_vocabulary);
        for (segment, id) in segments.iter().zip(ids) {
            if let Some(id) = id {
                if !ordinary.is_empty() {
                    self.encode_sequence_into_with_matcher(
                        &ordinary,
                        &mut scratch,
                        &mut output,
                        &matcher,
                    )?;
                    ordinary.clear();
                }
                output.push(id.into());
            } else {
                ordinary.push_str(&segment.content);
            }
        }
        if !ordinary.is_empty() {
            self.encode_sequence_into_with_matcher(&ordinary, &mut scratch, &mut output, &matcher)?;
        }
        self.post_process(output, None, add_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::bucket_added_vocabulary::AddedToken;

    fn pipeline(skip_special: bool) -> PipelineTokenizer {
        let model = super::super::tests::hello_bpe();
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
            PipelineModel::BPE(model),
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(99.into(), 0)].into_boxed_slice(),
                    ..Template::default()
                },
                ..PipelinePostProcessor::default()
            },
            None,
            Default::default(),
            Some(PaddingParams {
                pad_id: 42,
                ..PaddingParams::default()
            }),
        )
    }

    fn ids(handle: EncodeHandle) -> Vec<Vec<u32>> {
        handle
            .wait()
            .unwrap()
            .iter()
            .map(|e| e.ids().iter().map(|t| t.id()).collect())
            .collect()
    }

    #[test]
    fn structured_protects_specials_without_mutating_policy() {
        for policy in [false, true] {
            let tk = pipeline(policy);
            let before = ids(tk.encode("hello<s>hello", false));
            let segments = [
                EncodeSegment::text("hello<s>"),
                EncodeSegment::special("<s>"),
                EncodeSegment::text("hello"),
            ];
            assert_eq!(
                ids(tk.encode_segments(&segments, false)),
                vec![vec![7, 8, 7]]
            );
            assert_eq!(ids(tk.encode("hello<s>hello", false)), before);
            assert_eq!(
                tk.get_added_vocabulary().get_encode_special_tokens(),
                policy
            );
        }
    }

    #[test]
    fn structured_coalesces_text_and_retains_added_tokens_in_both_passes() {
        let tk = pipeline(false);
        assert_eq!(
            ids(tk.encode_segments(
                &[
                    EncodeSegment::text("he"),
                    EncodeSegment::text(""),
                    EncodeSegment::text("llo<added><n>")
                ],
                false
            )),
            vec![vec![7, 10, 9]]
        );
    }

    #[test]
    fn structured_normalized_specials_are_explicit_and_case_sensitive() {
        let tk = pipeline(false);
        assert_eq!(ids(tk.encode("HELLO<NS>", false)), vec![vec![7, 11]]);
        assert_eq!(
            ids(tk.encode_segments(
                &[
                    EncodeSegment::text("HELLO<NS>"),
                    EncodeSegment::special("<ns>")
                ],
                false
            )),
            vec![vec![7, 11]]
        );
        assert!(
            tk.encode_segments(&[EncodeSegment::special("<NS>")], false)
                .wait()
                .is_err()
        );
    }

    #[test]
    fn structured_requires_exact_registered_specials() {
        let tk = pipeline(false);
        for text in ["", "<missing>", "hello", "<added>", " <s>", "<s><s>"] {
            assert!(
                tk.encode_segments(&[EncodeSegment::special(text)], false)
                    .wait()
                    .is_err(),
                "accepted {text}"
            );
        }
    }

    #[test]
    fn structured_applies_post_processing_once_and_batch_padding() {
        let tk = pipeline(false);
        let batch = vec![
            vec![EncodeSegment::text("hello"), EncodeSegment::special("<s>")],
            vec![],
        ];
        let encodings = tk.encode_segments_batch(&batch, true).wait().unwrap();
        assert_eq!(encodings[0].ids(), &[99, 7, 8]);
        assert_eq!(encodings[1].ids(), &[99, 42, 42]);
        assert_eq!(encodings[1].attention_mask(), Some([1, 0, 0].as_slice()));
        let unpadded = tk
            .encode_segments_batch(&batch, false)
            .wait_with_padding(None)
            .unwrap();
        assert_eq!(unpadded[0].ids(), &[7, 8]);
        assert!(unpadded[1].ids().is_empty());
        assert!(
            tk.encode_segments_batch(&[], false)
                .wait()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn structured_batch_errors_and_concurrent_calls_are_isolated() {
        let tk = pipeline(false);
        let results: Vec<_> = tk
            .encode_segments_batch(
                &[
                    vec![EncodeSegment::special("missing")],
                    vec![EncodeSegment::special("<s>")],
                ],
                false,
            )
            .into_iter()
            .collect();
        assert!(results[0].1.is_err());
        assert_eq!(results[1].1.as_ref().unwrap().ids(), &[8]);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..20 {
                        assert_eq!(
                            ids(tk.encode_segments(
                                &[
                                    EncodeSegment::text("hello<s>"),
                                    EncodeSegment::special("<s>")
                                ],
                                false
                            )),
                            vec![vec![7, 8]]
                        );
                        assert_eq!(ids(tk.encode("hello<s>", false)), vec![vec![7, 8]]);
                    }
                });
            }
        });
    }
}
