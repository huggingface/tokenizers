//! Cutting text into words on the SentencePiece delimiter.
//!
//! SentencePiece vocabularies (gemma, llama-2, …) write a space as `▁` (U+2581), so `"tell me"`
//! becomes `"tell▁me"` before the model sees it. The rewriting is the normalizer's job — a
//! pre-tokenizer only returns byte ranges, it cannot change the text — and this module does the rest:
//! it cuts the text before each `▁`, so every word arrives at the model on its own.
//!
//! Two kinds of tokenizers end up here, and they do not want the same cuts.
//!
//! Some ship a `Metaspace` pre-tokenizer, which already cuts before every `▁`. Their tokens are
//! whatever that produces, so we have to make exactly the same cuts.
//!
//! The others ship no pre-tokenizer at all, so the model receives the whole text in one piece.
//! Cutting it into words is only a speed-up: merging a long text costs more than merging its words
//! one after another, and short words are friendlier to the cache. But a speed-up is worthless if it
//! changes the tokens, so here every cut has to be proven harmless first. Two rules do that:
//!
//! 1. A group of delimiters stays with the word that follows it (`a▁▁▁b` → `a`, `▁▁▁b`).
//!    Vocabularies hold pieces made of several delimiters (`▁▁`, `▁▁▁`), and cutting inside a group
//!    would stop those from forming.
//! 2. A cut is dropped when a vocabulary piece could merge across it. A merge only ever produces a
//!    piece that is in the vocabulary, so this can only happen if some piece holds a delimiter that
//!    is not at its start. gemma has exactly one such piece: `>▁</`. See [`Veto`].

use atomsplit::literal::Literal;

use crate::normalizers::NormalizerWrapper;
use crate::normalizers::replace::{Replace, ReplacePattern};
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::pre_tokenizers::metaspace::PrependScheme;
use crate::pre_tokenizers::split::SplitPattern;
use crate::tokenizer::Result;
use crate::tokenizer::pipeline::{self, PipelineModel, Span};

/// Splits `▁`-spelled text into words. See the module docs.
#[derive(Debug, Clone)]
pub struct MetaspaceSplit {
    /// The character words start with, `▁` unless the tokenizer picked another one.
    delimiter: Literal,
    cuts: Cuts,
}

/// Which delimiters may be cut at.
#[derive(Debug, Clone, PartialEq)]
enum Cuts {
    /// All of them, which is what a `Metaspace` pre-tokenizer does.
    Every,
    /// The first of each group, and only where [`Veto`] allows it.
    Proven(Veto),
}

/// Hand-written because the prepared search inside `Literal` has no equality of its own — and needs
/// none, since it is built from the delimiter.
impl PartialEq for MetaspaceSplit {
    fn eq(&self, other: &Self) -> bool {
        self.delimiter.pattern() == other.delimiter.pattern() && self.cuts == other.cuts
    }
}

impl MetaspaceSplit {
    fn new(delimiter: Literal, cuts: Cuts) -> Self {
        Self { delimiter, cuts }
    }
}

impl pipeline::PreTokenizer for MetaspaceSplit {
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
            let cut = match &self.cuts {
                Cuts::Every => true,
                // A delimiter right in front of this one means we are inside a group of them, and
                // a group stays whole with the word after it.
                Cuts::Proven(veto) => !bytes[..at].ends_with(delimiter) && !veto.forbids(bytes, at),
            };
            if cut {
                out.push(Span::new(start as u32, at as u32));
                start = at;
            }
        }
        if start < bytes.len() {
            out.push(Span::new(start as u32, bytes.len() as u32));
        }
        Ok(())
    }
}

/// Rewrites a tokenizer that spells spaces as a delimiter into one [`MetaspaceSplit`], or returns
/// `None` to leave it alone.
///
/// The normalizer has to turn every space into a single character, and the pre-tokenizer has to be
/// one of the three shapes below. Anything else is refused: guessing wrong here changes the tokens a
/// tokenizer produces.
pub(crate) fn lower(
    normalizer: Option<&NormalizerWrapper>,
    pre_tokenizer: Option<&PreTokenizerWrapper>,
    model: &PipelineModel,
) -> Option<MetaspaceSplit> {
    let replacement = delimiter_from_normalizer(normalizer?)?;
    let delimiter = Literal::new(replacement.to_string().as_bytes());
    match pre_tokenizer {
        // No pre-tokenizer: the model gets the whole text, so cuts must be proven harmless.
        None => {
            let veto = proven_cuts(model, &delimiter)?;
            Some(MetaspaceSplit::new(delimiter, Cuts::Proven(veto)))
        }
        // A `Split` that can never match, because the normalizer already replaced every space it
        // looks for. With nothing to match, every behaviour it could carry leaves the text in one
        // piece, so the model gets the whole text here too. gemma ships this shape.
        Some(PreTokenizerWrapper::Split(split))
            if !split.invert
                && matches!(&split.pattern, SplitPattern::String(pattern)
                    if !pattern.is_empty() && pattern.chars().all(|c| c == ' ')) =>
        {
            let veto = proven_cuts(model, &delimiter)?;
            Some(MetaspaceSplit::new(delimiter, Cuts::Proven(veto)))
        }
        // A `Metaspace` that only splits: the normalizer left it no space to replace, and it is set
        // not to prepend anything. Its own cuts are the reference, so we make all of them.
        Some(PreTokenizerWrapper::Metaspace(metaspace))
            if metaspace.get_split()
                && metaspace.get_prepend_scheme() == PrependScheme::Never
                && metaspace.get_replacement() == replacement =>
        {
            Some(MetaspaceSplit::new(delimiter, Cuts::Every))
        }
        _ => None,
    }
}

/// The vocabulary rules for cutting a text the model expects whole, or `None` when this model cannot
/// be cut at all.
fn proven_cuts(model: &PipelineModel, delimiter: &Literal) -> Option<Veto> {
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

/// Width of one padded half of a veto piece. Whatever length the halves really are, they are compared
/// a full `u128` at a time.
const HALF_WIDTH: usize = size_of::<u128>();

/// One half of a [`VetoPiece`], padded out to [`HALF_WIDTH`] bytes.
///
/// The halves are a byte or two long, but their length is only known once the vocabulary is read, and
/// comparing a run-time number of bytes means calling `memcmp`. Padding to a fixed width instead
/// turns the compare into a couple of register operations, which is worth it in a loop that runs once
/// per word.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Half {
    bytes: u128,
    /// `0xff` over the bytes of the half, `0` over the padding.
    mask: u128,
}

impl Half {
    /// The half at the end of the window, where the bytes running up to a cut land.
    fn ending(half: &[u8]) -> Option<Self> {
        Self::padded(half, HALF_WIDTH.checked_sub(half.len())?)
    }

    /// The half at the start of the window, where the bytes following a cut land.
    fn starting(half: &[u8]) -> Option<Self> {
        Self::padded(half, 0)
    }

    /// `None` when the half is wider than the window.
    fn padded(half: &[u8], at: usize) -> Option<Self> {
        let mut bytes = [0u8; HALF_WIDTH];
        let mut mask = [0u8; HALF_WIDTH];
        bytes.get_mut(at..at + half.len())?.copy_from_slice(half);
        mask[at..at + half.len()].fill(0xff);
        Some(Self {
            bytes: u128::from_le_bytes(bytes),
            mask: u128::from_le_bytes(mask),
        })
    }

    fn matches(&self, window: u128) -> bool {
        (window ^ self.bytes) & self.mask == 0
    }
}

/// A vocabulary piece holding a delimiter that is not at its start, split at that delimiter: `>▁</`
/// becomes `VetoPiece { before: ">", after: "</" }`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VetoPiece {
    before: Half,
    after: Half,
}

impl VetoPiece {
    fn new(before: &[u8], after: &[u8]) -> Option<Self> {
        Some(Self {
            before: Half::ending(before)?,
            after: Half::starting(after)?,
        })
    }
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
    /// Last bytes of the `before` halves, as a 256-bit set. A cut with any other byte in front of it
    /// fits no piece, which is how most cuts get away without a single comparison.
    bytes_before: [u64; 4],
    /// Length of the delimiter the pieces were split at. Every `after` half starts at the byte right
    /// behind the delimiter, so the window compared to it has to start there too.
    delimiter_len: usize,
}

impl Veto {
    /// Reads the vocabulary and collects the pieces a merge could use to reach across a cut.
    ///
    /// `None` turns cutting off: either this is not a SentencePiece vocabulary (no delimiter piece in
    /// it), or its veto pieces are too many, too long or too tangled to rule out cheaply.
    fn build(vocab: &[(Vec<u8>, u32)], delimiter: &Literal) -> Option<Self> {
        let pattern = delimiter.pattern();
        if !vocab.iter().any(|(piece, _)| piece.as_slice() == pattern) {
            return None;
        }
        let mut pieces = Vec::new();
        let mut bytes_before = [0u64; 4];
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
                let previous = *before.last().expect("`at` is past the start of the piece");
                bytes_before[(previous >> 6) as usize] |= 1 << (previous & 63);
                pieces.push(VetoPiece::new(before, after)?);
            }
        }
        Some(Self {
            pieces,
            bytes_before,
            delimiter_len: pattern.len(),
        })
    }

    /// Could a piece cover the delimiter at `at`, so that a merge reaches over a cut placed there? A
    /// match only means such a merge is possible, not that the model performs it — either way we
    /// leave the text in one piece. `at` is past the start of the text, so there is a byte in front
    /// of it.
    fn forbids(&self, text: &[u8], at: usize) -> bool {
        let previous = text[at - 1];
        if self.bytes_before[(previous >> 6) as usize] & (1 << (previous & 63)) == 0 {
            return false;
        }
        // Both windows are padded with zeros, so a half can only match beyond the ends of `text` if
        // the half itself holds a zero byte — and one match too many only leaves the text uncut.
        let before = window_ending(&text[..at]);
        let after = window_starting(&text[at + self.delimiter_len..]);
        self.pieces
            .iter()
            .any(|piece| piece.before.matches(before) && piece.after.matches(after))
    }
}

/// The last [`HALF_WIDTH`] bytes of `bytes`, padded on the left when there are fewer, packed the way
/// [`Half::ending`] packs a half.
fn window_ending(bytes: &[u8]) -> u128 {
    match bytes.last_chunk() {
        Some(window) => u128::from_le_bytes(*window),
        None => {
            let mut window = [0u8; HALF_WIDTH];
            window[HALF_WIDTH - bytes.len()..].copy_from_slice(bytes);
            u128::from_le_bytes(window)
        }
    }
}

/// The first [`HALF_WIDTH`] bytes of `bytes`, padded on the right when there are fewer, packed the way
/// [`Half::starting`] packs a half.
fn window_starting(bytes: &[u8]) -> u128 {
    match bytes.first_chunk() {
        Some(window) => u128::from_le_bytes(*window),
        None => {
            let mut window = [0u8; HALF_WIDTH];
            window[..bytes.len()].copy_from_slice(bytes);
            u128::from_le_bytes(window)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::{BPE, BpeBuilder, Merges, PipelineBPE, Vocab};
    use crate::pre_tokenizers::metaspace::Metaspace;
    use crate::tokenizer::{
        Model, OffsetReferential, OffsetType, PreTokenizedString,
        PreTokenizer as LegacyPreTokenizer,
    };

    const DELIMITER: char = '▁';

    fn delimiter() -> Literal {
        Literal::new(DELIMITER.to_string().as_bytes())
    }

    fn split_on(cuts: Cuts, text: &str) -> Vec<String> {
        let split = MetaspaceSplit::new(delimiter(), cuts);
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
    fn ids_word_by_word(model: &PipelineBPE, split: &MetaspaceSplit, text: &str) -> Vec<u32> {
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
        let split = MetaspaceSplit::new(delimiter(), Cuts::Proven(veto));
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
        let cuts = || Cuts::Proven(veto_of(&["▁"]).unwrap());
        assert_eq!(split_on(cuts(), "a▁▁▁b"), ["a", "▁▁▁b"]);
        assert_eq!(split_on(cuts(), "▁▁▁a"), ["▁▁▁a"]);
        assert_eq!(split_on(cuts(), "hello"), ["hello"]);
        assert_eq!(split_on(cuts(), ""), Vec::<String>::new());
    }

    #[test]
    fn proven_cuts_skip_what_the_veto_forbids() {
        let cuts = || Cuts::Proven(veto_of(&["▁", ">▁</", "▁a"]).unwrap());
        // ">▁</" can merge across the space between an HTML tag and its closing tag.
        assert_eq!(split_on(cuts(), "<b>▁</b>"), ["<b>▁</b>"]);
        // ">" in front of the `▁` again, but "</" does not follow, so the piece cannot form.
        assert_eq!(split_on(cuts(), "<b>▁a"), ["<b>", "▁a"]);
        // Only the cut the piece covers is dropped, not the other ones.
        assert_eq!(split_on(cuts(), "x▁<b>▁</b>▁y"), ["x", "▁<b>▁</b>", "▁y"]);
    }

    #[test]
    fn every_cut_matches_the_metaspace_pre_tokenizer() {
        let metaspace = Metaspace::new(DELIMITER, PrependScheme::Never, true);
        for text in ["▁a▁b", "a▁b", "a▁▁▁b", "▁▁▁", "hello", "▁", ""] {
            let mut legacy = PreTokenizedString::from(text);
            metaspace.pre_tokenize(&mut legacy).unwrap();
            let expected: Vec<String> = legacy
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .iter()
                .map(|(word, _, _)| (*word).to_string())
                .collect();
            assert_eq!(split_on(Cuts::Every, text), expected, "{text:?}");
        }
    }

    /// The build-time rewrite of a `▁`-spelling tokenizer into one splitter.
    ///
    /// A `Replace` normalizer cannot even be built without a system-regex backend, so these need the
    /// `fancy-regex` feature.
    #[cfg(feature = "fancy-regex")]
    mod lowering {
        use super::*;

        /// The two shapes a `▁`-spelling tokenizer ships, copied from `data/gemma-4.json` and
        /// `data/llama-2.json`.
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
        fn lower_rewrites_the_shapes_that_hand_the_model_whole_text() {
            let model = bpe_model(metaspace_bpe());
            for (name, normalizer_json, pre_tokenizer_json) in [
                ("gemma", GEMMA_NORMALIZER, Some(GEMMA_PRE_TOKENIZER)),
                ("llama-2", LLAMA_NORMALIZER, None),
            ] {
                let lowered = lower(
                    Some(&normalizer(normalizer_json)),
                    pre_tokenizer_json.map(pre_tokenizer).as_ref(),
                    &model,
                )
                .unwrap_or_else(|| panic!("{name} should be rewritten"));
                assert_eq!(lowered.delimiter.pattern(), "▁".as_bytes(), "{name}");
                assert!(matches!(lowered.cuts, Cuts::Proven(_)), "{name}");
            }
        }

        #[test]
        fn lower_rewrites_a_metaspace_that_only_splits() {
            let metaspace =
                r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}"#;
            let lowered = lower(
                Some(&normalizer(GEMMA_NORMALIZER)),
                Some(&pre_tokenizer(metaspace)),
                &bpe_model(metaspace_bpe()),
            )
            .unwrap();
            assert_eq!(lowered.cuts, Cuts::Every);
        }

        #[test]
        fn lower_refuses_what_it_cannot_prove() {
            let model = bpe_model(metaspace_bpe());
            let refused: [(
                &str,
                Option<&NormalizerWrapper>,
                Option<PreTokenizerWrapper>,
            ); 6] = [
                ("no normalizer at all", None, None),
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
                // The pre-tokenizer cuts the text itself, so the model never sees it whole.
                (
                    "a split that does match",
                    Some(&normalizer(GEMMA_NORMALIZER)),
                    Some(pre_tokenizer(
                        r#"{"type":"Split","pattern":{"String":"▁"},"behavior":"MergedWithNext","invert":false}"#,
                    )),
                ),
                (
                    "a metaspace that prepends",
                    Some(&normalizer(GEMMA_NORMALIZER)),
                    Some(pre_tokenizer(
                        r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#,
                    )),
                ),
                (
                    "a byte-level pre-tokenizer",
                    Some(&normalizer(GEMMA_NORMALIZER)),
                    Some(pre_tokenizer(
                        r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#,
                    )),
                ),
            ];
            for (name, normalizer, pre_tokenizer) in refused {
                assert!(
                    lower(normalizer, pre_tokenizer.as_ref(), &model).is_none(),
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
                lower(Some(&normalizer(GEMMA_NORMALIZER)), None, &bpe_model(plain)).is_none(),
                "vocabulary without the delimiter"
            );
        }

        /// The real files, so a change to either config shape shows up here. Skipped when the fixtures
        /// have not been fetched.
        #[test]
        fn lower_rewrites_the_real_fixtures() {
            for file in ["gemma-4.json", "llama-2.json"] {
                let Ok(tree) = crate::Tokenizer::from_file(format!("../data/{file}")) else {
                    eprintln!("skip {file}: not present (fetch with `make bench-models`)");
                    continue;
                };
                let pipeline =
                    crate::pipeline::PipelineTokenizer::try_from(&tree).expect("supported model");
                assert!(
                    matches!(
                        pipeline.get_pre_tokenizer(),
                        crate::pipeline::PipelinePreTokenizer::MetaspaceSplit(_)
                    ),
                    "{file} should be rewritten into one splitter"
                );
            }
        }
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
            [VetoPiece::new(b"p", b"").unwrap()]
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
        // A half that does not fit the fixed-width compare.
        let wide = "a".repeat(HALF_WIDTH + 1);
        assert!(veto_of(&["▁", &format!("{wide}▁x")]).is_none());
        assert!(veto_of(&["▁", &format!("a▁{wide}")]).is_none());
    }

    #[test]
    fn veto_matches_a_half_that_fills_the_window() {
        let before = "a".repeat(HALF_WIDTH);
        let cuts = || Cuts::Proven(veto_of(&["▁", &format!("{before}▁x"), "▁x"]).unwrap());
        let text = format!("{before}▁x");
        assert_eq!(split_on(cuts(), &text), [text.as_str()]);
        // One byte short of the half: the window pads with a zero the half does not hold, so the
        // piece cannot form and the cut stands.
        let short = &before[1..];
        assert_eq!(split_on(cuts(), &format!("{short}▁x")), [short, "▁x"]);
    }
}
