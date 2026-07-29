//! Cutting a whole SentencePiece text into words, where the vocabulary proves it is harmless.
//!
//! SentencePiece vocabularies (gemma, llama-2, …) write a space as `▁` (U+2581), so `"tell me"`
//! reaches the model as `"▁tell▁me"`. That character both opens a word and separates two, which is
//! why the code calls it the delimiter.
//!
//! Some of these tokenizers ship a [`Metaspace`] pre-tokenizer, which writes the delimiters and cuts
//! before every one of them. Those cuts are the tokenizer's own output, so
//! [`super::metaspace::to_normalizer_and_split`] reproduces them exactly and there is nothing to
//! decide.
//!
//! The tokenizers here are the other kind: their normalizer writes the delimiters and they ship no
//! pre-tokenizer that cuts, so the model receives the whole text in one piece. Cutting it into words
//! is only a speed-up — merging a long text costs more than merging its words one after another, and
//! short words are friendlier to the cache. But a speed-up is worthless if it changes the tokens, so
//! here every cut has to be proven harmless first. Two rules do that:
//!
//! 1. A group of delimiters stays with the word that follows it (`a▁▁▁b` → `a`, `▁▁▁b`).
//!    Vocabularies hold pieces made of several delimiters (`▁▁`, `▁▁▁`), and cutting inside a group
//!    would stop those from forming.
//! 2. A cut is dropped when a vocabulary piece could merge across it. A merge only ever produces a
//!    piece that is in the vocabulary, so this can only happen if some piece holds a delimiter that
//!    is not at its start. gemma has exactly one such piece: `>▁</`. See [`Veto`].
//!
//! [`Metaspace`]: super::metaspace::Metaspace

use atomsplit::literal::Literal;

use crate::normalizers::NormalizerWrapper;
use crate::normalizers::replace::{Replace, ReplacePattern};
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::pre_tokenizers::split::SplitPattern;
use crate::tokenizer::Result;
use crate::tokenizer::pipeline::{self, PipelineModel, Span};

/// Splits `▁`-spelled text into words, at the delimiters the vocabulary allows. See the module docs.
#[derive(Debug, Clone)]
pub struct ProvenCuts {
    /// The character words start with, `▁` unless the tokenizer picked another one.
    delimiter: Literal,
    veto: Veto,
}

/// Hand-written because the prepared search inside `Literal` has no equality of its own — and needs
/// none, since it is built from the delimiter.
impl PartialEq for ProvenCuts {
    fn eq(&self, other: &Self) -> bool {
        self.delimiter.pattern() == other.delimiter.pattern() && self.veto == other.veto
    }
}

impl ProvenCuts {
    fn new(delimiter: Literal, veto: Veto) -> Self {
        Self { delimiter, veto }
    }
}

impl pipeline::PreTokenizer for ProvenCuts {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Span>) -> Result<()> {
        let delimiter = self.delimiter.pattern();
        let bytes = text.as_bytes();
        // The word being built runs from `start` to the next delimiter we cut at.
        let mut start = 0usize;
        for at in self.delimiter.matches(bytes) {
            // A delimiter at `start` is the cut we just made, or the text opens with one: either way
            // the word would be empty.
            if at == start {
                continue;
            }
            // A delimiter right in front of this one means we are inside a group of them, and a group
            // stays whole with the word after it.
            if bytes[..at].ends_with(delimiter) || self.veto.forbids(bytes, at) {
                continue;
            }
            out.push(Span::new(start as u32, at as u32));
            start = at;
        }
        if start < bytes.len() {
            out.push(Span::new(start as u32, bytes.len() as u32));
        }
        Ok(())
    }
}

/// The splitter for a tokenizer whose text reaches the model in one piece, or `None` when the text
/// has to stay that way — because something already cuts it, because no normalizer provably writes
/// the delimiters, or because the vocabulary cannot prove the cuts.
pub(crate) fn for_tokenizer(
    normalizer: Option<&NormalizerWrapper>,
    pre_tokenizer: Option<&PreTokenizerWrapper>,
    model: &PipelineModel,
) -> Option<ProvenCuts> {
    if !leaves_the_text_whole(pre_tokenizer) {
        return None;
    }
    let delimiter = delimiter_from_normalizer(normalizer?)?;
    let delimiter = Literal::new(delimiter.to_string().as_bytes()).expect("a char is never empty");
    let veto = veto_from_model(model, &delimiter)?;
    Some(ProvenCuts::new(delimiter, veto))
}

/// Does this pre-tokenizer leave the text in one piece? Only the two shapes below do, and anything
/// else is refused rather than guessed at: cutting text a tokenizer meant to keep whole changes the
/// tokens it produces.
fn leaves_the_text_whole(pre_tokenizer: Option<&PreTokenizerWrapper>) -> bool {
    match pre_tokenizer {
        // Nothing cuts the text at all. llama-2 ships this shape.
        None => true,
        // A `Split` that can never match, because the normalizer already replaced every space it
        // looks for. With nothing to match, every behaviour it could carry leaves the text in one
        // piece. gemma ships this shape.
        Some(PreTokenizerWrapper::Split(split)) => {
            !split.invert
                && matches!(&split.pattern, SplitPattern::String(pattern)
                    if !pattern.is_empty() && pattern.chars().all(|c| c == ' '))
        }
        _ => false,
    }
}

/// The cuts this model's vocabulary forbids, or `None` when it cannot be cut at all.
fn veto_from_model(model: &PipelineModel, delimiter: &Literal) -> Option<Veto> {
    let PipelineModel::BPE(bpe) = model else {
        return None;
    };
    // With `ignore_merges` the model first looks the whole text up in the vocabulary and emits one
    // token when it finds it. Handing it words instead would skip that lookup.
    if bpe.ignore_merges() {
        return None;
    }
    Veto::build(&bpe.vocab_bytes(), delimiter)
}

/// The character every space becomes, if the normalizer provably rewrites all of them.
///
/// Accepts a `Replace` on its own, or a sequence whose last step is one and whose earlier steps only
/// prepend text — a step running after the `Replace` could bring spaces back.
fn delimiter_from_normalizer(normalizer: &NormalizerWrapper) -> Option<char> {
    match normalizer {
        NormalizerWrapper::Replace(replace) => space_replacement(replace),
        NormalizerWrapper::Sequence(sequence) => {
            let (last, rest) = sequence.as_ref().split_last()?;
            rest.iter()
                .all(|step| matches!(step, NormalizerWrapper::Prepend(_)))
                .then(|| match last {
                    NormalizerWrapper::Replace(replace) => space_replacement(replace),
                    _ => None,
                })
                .flatten()
        }
        _ => None,
    }
}

/// The single character this normalizer turns every space into.
fn space_replacement(replace: &Replace) -> Option<char> {
    if replace.pattern() != &ReplacePattern::String(" ".to_string()) {
        return None;
    }
    let mut content = replace.content.chars();
    let delimiter = content.next()?;
    content.next().is_none().then_some(delimiter)
}

/// How many veto pieces we put up with before giving up on cutting at all.
const MAX_VETO_PIECES: usize = 32;

/// A vocabulary piece holding a delimiter that is not at its start, split at that delimiter: `>▁</`
/// becomes `VetoPiece { before: ">", after: "</" }`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VetoPiece {
    before: Vec<u8>,
    after: Vec<u8>,
}

/// The cuts a vocabulary does not allow.
///
/// A merge only ever produces a piece that is in the vocabulary, so a merge can only reach across a
/// cut if some piece holds a delimiter that is not at its start. Those pieces are collected here, and
/// a cut where one of them fits is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Veto {
    /// Empty means no merge can reach across a cut, so every cut is allowed.
    pieces: Vec<VetoPiece>,
    /// Length of the delimiter the pieces were split at. Every `after` half starts at the byte right
    /// behind the delimiter, so the text is compared to it from there.
    delimiter_len: usize,
}

impl Veto {
    /// Reads the vocabulary and collects the pieces a merge could use to reach across a cut.
    ///
    /// `None` turns cutting off: either this is not a SentencePiece vocabulary (no delimiter piece in
    /// it), or its veto pieces are too many or too tangled to rule out cheaply.
    fn build(vocab: &[(Vec<u8>, u32)], delimiter: &Literal) -> Option<Self> {
        let pattern = delimiter.pattern();
        if !vocab.iter().any(|(piece, _)| piece.as_slice() == pattern) {
            return None;
        }
        let mut pieces = Vec::new();
        for (piece, _) in vocab {
            for at in delimiter.matches(piece) {
                // A piece starting with a delimiter sits right after a cut, not across it, and a
                // delimiter following another one is inside a group, where we never cut.
                if at == 0 || piece[..at].ends_with(pattern) {
                    continue;
                }
                let (before, after) = (&piece[..at], &piece[at + pattern.len()..]);
                // A piece with a second delimiter reaches across two cuts at once, and checking one
                // cut at a time no longer proves anything. None of the vocabularies we tested has
                // one, so drop cutting instead.
                if pieces.len() == MAX_VETO_PIECES
                    || delimiter.matches(before).next().is_some()
                    || delimiter.matches(after).next().is_some()
                {
                    return None;
                }
                pieces.push(VetoPiece {
                    before: before.to_vec(),
                    after: after.to_vec(),
                });
            }
        }
        Some(Self {
            pieces,
            delimiter_len: pattern.len(),
        })
    }

    /// Could a piece cover the delimiter at `at`, so that a merge reaches over a cut placed there? A
    /// match only means such a merge is possible, not that the model performs it — either way we
    /// leave the text in one piece.
    fn forbids(&self, text: &[u8], at: usize) -> bool {
        let after = &text[at + self.delimiter_len..];
        self.pieces
            .iter()
            .any(|piece| text[..at].ends_with(&piece.before) && after.starts_with(&piece.after))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::{BPE, BpeBuilder, Merges, PipelineBPE, Vocab};
    use crate::tokenizer::Model;

    const DELIMITER: char = '▁';

    fn delimiter() -> Literal {
        Literal::new(DELIMITER.to_string().as_bytes()).unwrap()
    }

    fn split_on(veto: Veto, text: &str) -> Vec<String> {
        let split = ProvenCuts::new(delimiter(), veto);
        let mut spans = Vec::new();
        pipeline::PreTokenizer::pre_tokenize(&split, text, &mut spans).unwrap();
        spans.iter().map(|s| text[s.range()].to_string()).collect()
    }

    fn veto_of(vocab: &[&str]) -> Option<Veto> {
        let pieces: Vec<(Vec<u8>, u32)> = vocab
            .iter()
            .enumerate()
            .map(|(id, piece)| (piece.as_bytes().to_vec(), id as u32))
            .collect();
        Veto::build(&pieces, &delimiter())
    }

    /// A vocabulary shaped like gemma's: `>▁</` and `p▁` hold a `▁` after another character, so
    /// cutting the text at those `▁` would drop merges the whole text would have made.
    fn metaspace_bpe() -> BPE {
        let vocab: Vocab = [
            ("▁", 0u32),
            ("<", 1),
            (">", 2),
            ("/", 3),
            ("s", 4),
            ("p", 5),
            ("a", 6),
            ("b", 7),
            ("</", 8),
            (">▁", 9),
            (">▁</", 10),
            ("p▁", 11),
            ("sp", 12),
            ("▁sp", 13),
            ("▁a", 14),
            ("▁b", 15),
        ]
        .iter()
        .map(|(piece, id)| ((*piece).into(), *id))
        .collect();
        let merges: Merges = [
            ("<", "/"),
            (">", "▁"),
            (">▁", "</"),
            ("p", "▁"),
            ("s", "p"),
            ("▁", "sp"),
            ("▁", "a"),
            ("▁", "b"),
        ]
        .iter()
        .map(|(a, b)| ((*a).into(), (*b).into()))
        .collect();
        BpeBuilder::default()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    /// The ids the model produces for each word, one word after another.
    fn ids_word_by_word(model: &PipelineBPE, split: &ProvenCuts, text: &str) -> Vec<u32> {
        let mut spans = Vec::new();
        pipeline::PreTokenizer::pre_tokenize(split, text, &mut spans).unwrap();
        let mut out = Vec::new();
        let mut scratch = pipeline::Model::init_scratch(model);
        for span in &spans {
            pipeline::Model::tokenize_pipeline(model, &text[span.range()], &mut scratch, &mut out)
                .unwrap();
        }
        out.iter().map(|token| token.id).collect()
    }

    /// The ids the model produces for the whole text at once — what the cuts must reproduce.
    fn ids_whole_text(model: &BPE, text: &str) -> Vec<u32> {
        model
            .tokenize(text)
            .unwrap()
            .iter()
            .map(|token| token.id)
            .collect()
    }

    #[test]
    fn proven_cuts_keep_the_whole_text_ids() {
        let reference = metaspace_bpe();
        let model = PipelineBPE::from_bpe(reference.clone(), false).unwrap();
        let veto = Veto::build(&model.vocab_bytes(), &delimiter()).unwrap();
        let split = ProvenCuts::new(delimiter(), veto);
        for text in [
            "▁a▁b",
            "a▁b",
            "▁▁▁a",
            "a▁▁▁b",
            "▁sp▁a",
            "sp",
            "a▁",
            "▁",
            "▁a▁sp▁b▁a",
            // The veto pieces, alone and surrounded by other cuts.
            "<b>▁</b>",
            "▁a<b>▁</b>▁b",
            "p▁a",
            "▁ap▁a▁b",
            // Starts like a veto piece but ends differently, so the cut stands.
            "<b>▁a",
            "q▁a",
        ] {
            assert_eq!(
                ids_word_by_word(&model, &split, text),
                ids_whole_text(&reference, text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn proven_cuts_keep_delimiter_groups_whole() {
        let veto = || veto_of(&["▁"]).unwrap();
        assert_eq!(split_on(veto(), "a▁▁▁b"), ["a", "▁▁▁b"]);
        assert_eq!(split_on(veto(), "▁▁▁a"), ["▁▁▁a"]);
        assert_eq!(split_on(veto(), "hello"), ["hello"]);
        assert_eq!(split_on(veto(), ""), Vec::<String>::new());
    }

    #[test]
    fn proven_cuts_skip_what_the_veto_forbids() {
        let veto = || veto_of(&["▁", ">▁</", "▁a"]).unwrap();
        // ">▁</" can merge across the space between an HTML tag and its closing tag.
        assert_eq!(split_on(veto(), "<b>▁</b>"), ["<b>▁</b>"]);
        // ">" in front of the `▁` again, but "</" does not follow, so the piece cannot form.
        assert_eq!(split_on(veto(), "<b>▁a"), ["<b>", "▁a"]);
        // Only the cut the piece covers is dropped, not the other ones.
        assert_eq!(split_on(veto(), "x▁<b>▁</b>▁y"), ["x", "▁<b>▁</b>", "▁y"]);
    }

    #[test]
    fn veto_needs_the_delimiter_in_the_vocabulary() {
        assert!(veto_of(&["a", "b"]).is_none());
        assert!(veto_of(&["a", "▁"]).is_some());
    }

    #[test]
    fn veto_collects_only_the_pieces_that_reach_over_a_cut() {
        assert!(veto_of(&["▁", "▁hello", "▁▁"]).unwrap().pieces.is_empty());
        assert_eq!(
            veto_of(&["▁", "p▁"]).unwrap().pieces,
            [VetoPiece {
                before: b"p".to_vec(),
                after: Vec::new(),
            }]
        );
    }

    #[test]
    fn veto_gives_up_when_it_cannot_prove_anything() {
        // Two delimiters in one piece would reach over two cuts at once.
        assert!(veto_of(&["▁", "a▁b▁c"]).is_none());
        // More pieces than the loop is willing to walk per cut.
        let mut vocab = vec!["▁".to_string()];
        vocab.extend((0..=MAX_VETO_PIECES).map(|i| format!("{i}▁x")));
        let vocab: Vec<&str> = vocab.iter().map(String::as_str).collect();
        assert!(veto_of(&vocab).is_none());
    }

    mod for_tokenizer {
        use super::*;

        /// The shapes a `▁`-spelling tokenizer ships, copied from the files in `data/`.
        const GEMMA_NORMALIZER: &str =
            r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#;
        const GEMMA_PRE_TOKENIZER: &str = r#"{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}"#;
        const LLAMA_NORMALIZER: &str = r#"{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]}"#;

        fn normalizer(json: &str) -> NormalizerWrapper {
            serde_json::from_str(json).unwrap()
        }

        fn pre_tokenizer(json: &str) -> PreTokenizerWrapper {
            serde_json::from_str(json).unwrap()
        }

        fn bpe_model(bpe: BPE) -> PipelineModel {
            PipelineModel::BPE(PipelineBPE::from_bpe(bpe, false).unwrap())
        }

        #[test]
        fn the_shapes_that_leave_the_text_whole_are_cut() {
            let model = bpe_model(metaspace_bpe());
            for (name, normalizer_json, pre_tokenizer_json) in [
                ("gemma", GEMMA_NORMALIZER, Some(GEMMA_PRE_TOKENIZER)),
                ("llama-2", LLAMA_NORMALIZER, None),
            ] {
                let cuts = for_tokenizer(
                    Some(&normalizer(normalizer_json)),
                    pre_tokenizer_json.map(pre_tokenizer).as_ref(),
                    &model,
                )
                .unwrap_or_else(|| panic!("{name} should be cut into words"));
                assert_eq!(cuts.delimiter.pattern(), "▁".as_bytes(), "{name}");
            }
        }

        #[test]
        fn refuses_what_it_cannot_prove() {
            let model = bpe_model(metaspace_bpe());
            let refused: [(&str, Option<&NormalizerWrapper>, Option<&str>); 5] = [
                ("no normalizer and no pre-tokenizer", None, None),
                // Nothing here turns spaces into the delimiter.
                (
                    "a normalizer that keeps spaces",
                    Some(&normalizer(r#"{"type":"NFC"}"#)),
                    None,
                ),
                // A step running after the replace could bring spaces back.
                (
                    "a replace that is not the last step",
                    Some(&normalizer(
                        r#"{"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"String":" "},"content":"▁"},{"type":"NFC"}]}"#,
                    )),
                    None,
                ),
                // This one cuts the text itself, so the model never sees it whole.
                (
                    "a split that does match",
                    Some(&normalizer(GEMMA_NORMALIZER)),
                    Some(
                        r#"{"type":"Split","pattern":{"String":"▁"},"behavior":"MergedWithNext","invert":false}"#,
                    ),
                ),
                (
                    "a byte-level pre-tokenizer",
                    Some(&normalizer(GEMMA_NORMALIZER)),
                    Some(
                        r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#,
                    ),
                ),
            ];
            for (name, normalizer, json) in refused {
                let declared = json.map(pre_tokenizer);
                assert!(
                    for_tokenizer(normalizer, declared.as_ref(), &model).is_none(),
                    "{name}"
                );
            }
            // A vocabulary without the delimiter proves nothing about cutting on it.
            let plain: Vocab = [("a", 0u32), ("b", 1)]
                .iter()
                .map(|(piece, id)| ((*piece).into(), *id))
                .collect();
            let plain = BpeBuilder::default()
                .vocab_and_merges(plain, Vec::new())
                .build()
                .unwrap();
            assert!(
                for_tokenizer(Some(&normalizer(GEMMA_NORMALIZER)), None, &bpe_model(plain))
                    .is_none(),
                "vocabulary without the delimiter"
            );
        }

        /// The real files, so a change to either config shape shows up here. The veto counts are the
        /// premise the whole proof rests on — gemma-4 has one piece that can merge across a word
        /// boundary (`>▁</`) and llama-2 has none — so they are pinned rather than described. Skipped
        /// when the fixtures have not been fetched.
        #[test]
        fn the_real_fixtures_are_cut() {
            for (file, veto_pieces) in [("gemma-4.json", 1), ("llama-2.json", 0)] {
                let path = format!("../data/{file}");
                if !std::path::Path::new(&path).exists() {
                    continue; // fixture not downloaded in this environment
                }
                let tree = crate::Tokenizer::from_file(&path).unwrap();
                let pipeline = crate::pipeline::PipelineTokenizer::try_from(&tree)
                    .unwrap_or_else(|e| panic!("{file} should build a pipeline: {e}"));
                let crate::pipeline::PipelinePreTokenizer::ProvenCuts(cuts) =
                    pipeline.get_pre_tokenizer()
                else {
                    panic!("{file} should be cut into words");
                };
                assert_eq!(cuts.veto.pieces.len(), veto_pieces, "{file} veto pieces");
            }
        }
    }
}
