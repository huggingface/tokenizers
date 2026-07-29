//! Recognize a known GPT pre-tokenization regex and route it to the byte-exact native
//! `atomsplit` FSM, so those pre-tokenizers need no system-regex backend. An unrecognized
//! pattern returns `None` and falls back to `SysRegex` (the optional fancy-regex backend).

// Canonical GPT pre-tokenization regexes (the look-ahead originals), used as recognition keys — the
// single source of truth lives in `atomsplit::regexes`. `gpt_fsm` maps each to the `atomsplit` FSM that
// reproduces its `Isolated` split byte-for-byte.
use atomsplit::fsm::Span;
use atomsplit::regexes::{DEEPSEEK_BIG, DEEPSEEK_CJK, DEEPSEEK_NUM, GPT2, O200K, TEKKEN};
// cl100k is recognized structurally (see `cl100k_digit_cap`), so the exact pattern is only a test key.
#[cfg(test)]
use atomsplit::regexes::CL100K;

/// A recognized GPT pre-tokenization regex that maps to a native `atomsplit` FSM (byte-exact).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptFsm {
    /// GPT-2 / ByteLevel regex → `atomsplit::fsm::fsm_byte_level`.
    Gpt2,
    /// cl100k-family regex → `atomsplit::fsm::fsm_cl100k_cap`. `digit_cap` is rule 3's `\p{N}{1,cap}`
    /// bound: 3 = cl100k / Llama-3, 1 = Qwen2 (`\p{N}`), `usize::MAX` = an unbounded `\p{N}+`.
    Cl100k { digit_cap: usize },
    /// o200k / GPT-4o regex → `atomsplit::fsm::fsm_o200k`.
    O200k,
    /// Mistral tekken regex → `atomsplit::fsm::fsm_tekken` (o200k's grammar with no contraction
    /// suffix and one token per digit).
    Tekken,
    /// deepseek Split-1 `\p{N}{1,3}` → `atomsplit::fsm::fsm_deepseek_num`.
    DeepSeekNum,
    /// deepseek Split-2 `[一-龥぀-ゟ゠-ヿ]+` → `atomsplit::fsm::fsm_deepseek_cjk`.
    DeepSeekCjk,
    /// deepseek Split-3 (the big regex) → `atomsplit::fsm::fsm_deepseek_big`.
    DeepSeekBig,
}

impl GptFsm {
    /// Does every byte of the input land in some match? True for the GPT regexes, whose final `\s+`
    /// alternative makes them total. False for deepseek's three, which each match only part of the text
    /// (they are composed, and only together do they cover it) — so those leave *gap* pieces, and any
    /// rewrite that treats "matched" and "kept" as the same set is invalid for them.
    pub fn covers_input(self) -> bool {
        !matches!(
            self,
            Self::DeepSeekNum | Self::DeepSeekCjk | Self::DeepSeekBig
        )
    }

    /// Run this pattern's `atomsplit` FSM: `tags` from `classify`, spans into `out` (len ≥ `text.len()`),
    /// token count returned. The single dispatch point — both the legacy [`GptFsmPattern`] and the
    /// pipeline `Split` route through it.
    #[inline]
    pub fn split_into(self, text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
        use atomsplit::fsm::*;
        match self {
            Self::Gpt2 => fsm_byte_level(text, tags, out),
            Self::Cl100k { digit_cap } => fsm_cl100k_cap(text, tags, out, digit_cap),
            Self::O200k => fsm_o200k(text, tags, out),
            Self::Tekken => fsm_tekken(text, tags, out),
            Self::DeepSeekNum => fsm_deepseek_num(text, tags, out),
            Self::DeepSeekCjk => fsm_deepseek_cjk(text, tags, out),
            Self::DeepSeekBig => fsm_deepseek_big(text, tags, out),
        }
    }
}

/// The cl100k-family template is fixed except rule 3's digit rule. If `pattern` is that template, return
/// the `\p{N}{1,cap}` bound (`\p{N}{1,3}`→3, `\p{N}{1,2}`→2, `\p{N}`→1, `\p{N}+`→`MAX`); else `None`.
/// This is what makes Qwen2 (cl100k with `\p{N}`) unroll without a per-tokenizer exact-string entry.
fn cl100k_digit_cap(pattern: &str) -> Option<usize> {
    // cl100k rules 1-2 (contraction + word) … <DIGIT RULE> … rules 4-7 (other + whitespace).
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

/// If `pattern` is a recognized GPT pre-tokenization regex, name the native FSM that reproduces its
/// `Isolated` split byte-for-byte. GPT-2, o200k, tekken and deepseek's three are matched exactly; the
/// cl100k family is matched structurally ([`cl100k_digit_cap`]) so digit-cap variants (Qwen2 …) unroll
/// too. An unrecognized pattern → `None` (the SysRegex / fancy-regex path handles it).
pub fn gpt_fsm(pattern: &str) -> Option<GptFsm> {
    match pattern {
        GPT2 => Some(GptFsm::Gpt2),
        O200K => Some(GptFsm::O200k),
        TEKKEN => Some(GptFsm::Tekken),
        DEEPSEEK_NUM => Some(GptFsm::DeepSeekNum),
        DEEPSEEK_CJK => Some(GptFsm::DeepSeekCjk),
        DEEPSEEK_BIG => Some(GptFsm::DeepSeekBig),
        _ => cl100k_digit_cap(pattern).map(|digit_cap| GptFsm::Cl100k { digit_cap }),
    }
}

/// `Pattern` that runs the native atomsplit FSM for a [`GptFsm`] — the legacy (`NormalizedString`)
/// split path's equivalent of the pipeline's native routing, so GPT pre-tokenizers need no
/// system-regex backend. `Isolated` behaviour keeps every span, and the FSM spans cover the input
/// contiguously, so this is byte-for-byte identical to the original GPT regex `Isolated` split.
pub struct GptFsmPattern(pub GptFsm);

impl crate::tokenizer::pattern::Pattern for GptFsmPattern {
    fn find_matches(
        &self,
        inside: &str,
    ) -> crate::tokenizer::Result<Vec<(crate::tokenizer::Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        use atomsplit::classify::classify;
        let bytes = inside.as_bytes();
        let mut tags = vec![0u8; bytes.len()];
        classify(bytes, &mut tags);
        let mut spans = vec![Span::default(); bytes.len() + 1];
        let n = self.0.split_into(bytes, &tags, &mut spans);
        Ok(spans[..n]
            .iter()
            .map(|sp| ((sp.start as usize, sp.end as usize), true))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_fsm_recognizes_family_and_extracts_digit_cap() {
        // Exact matches for gpt2 / o200k, structural (any digit cap) for the cl100k family.
        assert_eq!(gpt_fsm(GPT2), Some(GptFsm::Gpt2));
        assert_eq!(gpt_fsm(O200K), Some(GptFsm::O200k));
        assert_eq!(gpt_fsm(TEKKEN), Some(GptFsm::Tekken));
        assert_eq!(gpt_fsm(CL100K), Some(GptFsm::Cl100k { digit_cap: 3 }));
        // Qwen2: cl100k with rule 3 = `\p{N}` → cap 1.
        let qwen2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(gpt_fsm(qwen2), Some(GptFsm::Cl100k { digit_cap: 1 }));
        // Other in-family digit caps.
        let cap2 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}{1,2}");
        assert_eq!(gpt_fsm(&cap2), Some(GptFsm::Cl100k { digit_cap: 2 }));
        let unbounded = CL100K.replace(r"\p{N}{1,3}", r"\p{N}+");
        assert_eq!(
            gpt_fsm(&unbounded),
            Some(GptFsm::Cl100k {
                digit_cap: usize::MAX
            })
        );
        // Out of family → None (fancy-regex fallback): a foreign digit rule, and a totally unrelated regex.
        assert_eq!(gpt_fsm(&CL100K.replace(r"\p{N}{1,3}", r"\p{N}{2,4}")), None);
        assert_eq!(gpt_fsm(r"\w+|\s+"), None);
    }

    #[test]
    fn gpt_fsm_recognizes_deepseeks_three_splits() {
        // The `Sequence` fuses these into one `fsm_deepseek` pass (see `PipelineSequence`), but each
        // must be recognized on its own — that is what lets a deepseek `tokenizer.json` LOAD with no
        // system-regex backend, since `Split::new` only tolerates a missing backend for a known pattern.
        assert_eq!(gpt_fsm(DEEPSEEK_NUM), Some(GptFsm::DeepSeekNum));
        assert_eq!(gpt_fsm(DEEPSEEK_CJK), Some(GptFsm::DeepSeekCjk));
        assert_eq!(gpt_fsm(DEEPSEEK_BIG), Some(GptFsm::DeepSeekBig));
        // The big pattern ships with LITERAL CR/LF; the `\r`/`\n`-escaped form is an equivalent regex
        // but a different string, so it is (correctly) not recognized by string equality.
        assert_eq!(gpt_fsm(&DEEPSEEK_BIG.replace('\r', r"\r")), None);
    }
}
