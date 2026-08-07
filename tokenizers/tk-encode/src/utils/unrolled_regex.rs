//! Recognize a known GPT pre-tokenization regex and route it to the byte-exact native `bitsplit`
//! grammar, so those pre-tokenizers need no system-regex backend. An unrecognized pattern returns
//! `None` and falls back to `SysRegex` (the optional fancy-regex backend).

// The canonical regexes are the recognition keys; the single source of truth is `bitsplit::regexes`.
use bitsplit::Span;
use bitsplit::regexes::{GPT2, KIMI_K2, O200K, TEKKEN};
// cl100k is recognized structurally (see `cl100k_digit_cap`), so the exact pattern is only a test key.
#[cfg(test)]
use bitsplit::regexes::CL100K;

/// A recognized GPT pre-tokenization regex and the `bitsplit` grammar that reproduces its
/// `Isolated` split byte-for-byte. One variant per distinct regex; models sharing a regex share a
/// variant (o200k covers Llama-4, gpt-oss and MiniMax-M2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grammar {
    /// GPT-2 / ByteLevel.
    Gpt2,
    /// cl100k family. `digit_cap` is rule 3's `\p{N}{1,cap}` bound: 3 = cl100k / Llama-3 / GLM-4.6,
    /// 1 = Qwen (`\p{N}`), `usize::MAX` = an unbounded `\p{N}+`.
    Cl100k { digit_cap: usize },
    /// o200k / GPT-4o — and byte-for-byte the regex Llama-4, gpt-oss and MiniMax-M2 ship.
    O200k,
    /// Mistral tekken: o200k with no contraction suffix and one token per digit.
    Tekken,
    /// kimi-k2 / k3: o200k plus a leading `[\p{Han}]+` arm and a plain `[\r\n]*` rule-4 tail.
    Kimi,
}

impl Grammar {
    /// Write the token spans into `out`, returning the count. `starts`/`flag` are `u64` scratch
    /// bitmaps of length >= `text.len().div_ceil(64)`.
    pub fn split(
        self,
        text: &[u8],
        tags: &[u8],
        starts: &mut [u64],
        flag: &mut [u64],
        out: &mut [Span],
    ) -> usize {
        match self {
            Grammar::Gpt2 => bitsplit::bitsplit_byte_level(text, tags, starts, flag, out),
            Grammar::Cl100k { digit_cap: 1 } => {
                bitsplit::bitsplit_qwen(text, tags, starts, flag, out)
            }
            Grammar::Cl100k { .. } => bitsplit::bitsplit_cl100k(text, tags, starts, flag, out),
            Grammar::O200k => bitsplit::bitsplit_o200k(text, tags, starts, flag, out),
            Grammar::Tekken => bitsplit::bitsplit_tekken(text, tags, starts, flag, out),
            Grammar::Kimi => bitsplit::bitsplit_kimi(text, tags, starts, flag, out),
        }
    }

    /// Whether [`Self::split_tiled`] can drive this grammar. Ask *before* classifying: the tiled
    /// driver classifies first and would otherwise leave the fallback to classify the same bytes a
    /// second time, which is a straight regression for every grammar without a tiled emit.
    pub fn supports_tiled(self) -> bool {
        matches!(self, Grammar::Gpt2)
    }

    /// Emit into small tiles handed to `consume`, instead of one span array per document. Returns
    /// `false` for a grammar that has no tiled emit yet, so the caller keeps the whole-chunk path
    /// rather than silently taking a slower or different route.
    pub fn split_tiled<F: FnMut(&[Span])>(
        self,
        text: &[u8],
        tags: &[u8],
        starts: &mut [u64],
        flag: &mut [u64],
        tile: &mut [Span],
        consume: F,
    ) -> bool {
        match self {
            Grammar::Gpt2 => {
                bitsplit::bitsplit_byte_level_tiled(text, tags, starts, flag, tile, consume);
                true
            }
            _ => false,
        }
    }
}

/// The cl100k-family template is fixed except rule 3's digit rule. If `pattern` is that template,
/// return the `\p{N}{1,cap}` bound; else `None`. This is what makes Qwen (cl100k with `\p{N}`) and
/// GLM-4.6 (cl100k verbatim) route without a per-model entry.
fn cl100k_digit_cap(pattern: &str) -> Option<usize> {
    const PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|";
    const SUF: &str = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    match pattern.strip_prefix(PRE)?.strip_suffix(SUF)? {
        r"\p{N}{1,3}" => Some(3),
        r"\p{N}{1,2}" => Some(2),
        r"\p{N}" => Some(1),
        r"\p{N}+" => Some(usize::MAX),
        _ => None,
    }
}

/// If `pattern` is a recognized GPT pre-tokenization regex, name the grammar that reproduces its
/// `Isolated` split byte-for-byte. An unrecognized pattern -> `None` (the SysRegex path handles it).
pub fn recognize(pattern: &str) -> Option<Grammar> {
    match pattern {
        GPT2 => Some(Grammar::Gpt2),
        O200K => Some(Grammar::O200k),
        TEKKEN => Some(Grammar::Tekken),
        KIMI_K2 => Some(Grammar::Kimi),
        _ => cl100k_digit_cap(pattern).map(|digit_cap| Grammar::Cl100k { digit_cap }),
    }
}

/// `Pattern` that runs the native grammar on the legacy (`NormalizedString`) split path, so GPT
/// pre-tokenizers need no system-regex backend. `Isolated` keeps every span and these grammars
/// cover the input contiguously, so this is byte-for-byte the original regex `Isolated` split.
pub struct GrammarPattern(pub Grammar);

impl crate::tokenizer::pattern::Pattern for GrammarPattern {
    fn find_matches(
        &self,
        inside: &str,
    ) -> crate::tokenizer::Result<Vec<(crate::tokenizer::Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        let bytes = inside.as_bytes();
        let n = bytes.len();
        let mut tags = vec![0u8; n];
        bitsplit::classify::classify(bytes, &mut tags);
        let words = n.div_ceil(64) + 1;
        let (mut starts, mut flag) = (vec![0u64; words], vec![0u64; words]);
        let mut spans = vec![Span::default(); n + 1];
        let k = self.0.split(bytes, &tags, &mut starts, &mut flag, &mut spans);
        Ok(spans[..k]
            .iter()
            .map(|sp| ((sp.start as usize, sp.end as usize), true))
            .collect())
    }
}

// deepseek-v3/v4's pre-tokenizer is a `Sequence` of these three Isolated `Split`s (+ a byte-map
// `ByteLevel`), which `bitsplit::bitsplit_deepseek` collapses into one pass. Byte-exact with the
// shipped tokenizer.json — the big pattern carries LITERAL CR/LF, spliced in via `concat!`.
const DS_NUM: &str = r"\p{N}{1,3}";
const DS_CJK: &str = "[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+";
const DS_BIG: &str = concat!(
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^"##,
    "\r\n",
    r##"\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+["##,
    "\r\n",
    r##"]*|\s*["##,
    "\r\n",
    r##"]+|\s+(?!\S)|\s+"##,
);

/// True iff three `Split` patterns are exactly deepseek's `[\p{N}{1,3}, CJK-range, big-regex]` prefix →
/// `bitsplit::bitsplit_deepseek` reproduces the whole composed Isolated split in one pass.
pub fn is_deepseek(p0: &str, p1: &str, p2: &str) -> bool {
    p0 == DS_NUM && p1 == DS_CJK && p2 == DS_BIG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_fsm_recognizes_family_and_extracts_digit_cap() {
        // Exact matches for gpt2 / o200k, structural (any digit cap) for the cl100k family.
        assert_eq!(recognize(GPT2), Some(Grammar::Gpt2));
        assert_eq!(recognize(O200K), Some(Grammar::O200k));
        assert_eq!(recognize(TEKKEN), Some(Grammar::Tekken));
        assert_eq!(recognize(CL100K), Some(Grammar::Cl100k { digit_cap: 3 }));
        // Qwen2: cl100k with rule 3 = `\p{N}` → cap 1.
        let qwen2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(recognize(qwen2), Some(Grammar::Cl100k { digit_cap: 1 }));
        // Other in-family digit caps.
        let cap2 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}{1,2}");
        assert_eq!(recognize(&cap2), Some(Grammar::Cl100k { digit_cap: 2 }));
        let unbounded = CL100K.replace(r"\p{N}{1,3}", r"\p{N}+");
        assert_eq!(
            recognize(&unbounded),
            Some(Grammar::Cl100k {
                digit_cap: usize::MAX
            })
        );
        // Out of family → None (fancy-regex fallback): a foreign digit rule, and a totally unrelated regex.
        assert_eq!(recognize(&CL100K.replace(r"\p{N}{1,3}", r"\p{N}{2,4}")), None);
        assert_eq!(recognize(r"\w+|\s+"), None);
    }
}
