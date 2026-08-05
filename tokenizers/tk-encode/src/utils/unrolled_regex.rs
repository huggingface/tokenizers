//! Recognize a known GPT pre-tokenization regex and route it to the byte-exact native
//! `atomsplit` FSM, so those pre-tokenizers need no system-regex backend. An unrecognized
//! pattern returns `None` and falls back to `SysRegex` (the optional fancy-regex backend).

// Canonical GPT pre-tokenization regexes (the look-ahead originals), used as recognition keys — the
// single source of truth lives in `atomsplit::regexes`. `gpt_fsm` maps each to the `atomsplit` FSM that
// reproduces its `Isolated` split byte-for-byte.
use atomsplit::regexes::{GPT2, O200K, TEKKEN};
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
}

/// The cl100k-family template is fixed except rule 3's digit rule. If `pattern` is that template, return
/// the `\p{N}{1,cap}` bound (`\p{N}{1,3}`→3, `\p{N}{1,2}`→2, `\p{N}`→1, `\p{N}+`→`MAX`); else `None`.
/// This is what makes Qwen2 (cl100k with `\p{N}`) unroll without a per-tokenizer exact-string entry.
/// The inverse of [`cl100k_digit_cap`]: rebuild the cl100k-family pattern for a digit cap. A
/// `.tok` names the FSM family and its cap rather than carrying the regex source, so the loader
/// needs to spell the pattern back out for `Split::new` to recognise.
pub fn cl100k_pattern(digit_cap: usize) -> String {
    let digits = match digit_cap {
        1 => r"\p{N}".to_string(),
        usize::MAX => r"\p{N}+".to_string(),
        cap => format!(r"\p{{N}}{{1,{cap}}}"),
    };
    format!("{CL100K_PRE}{digits}{CL100K_SUF}")
}

/// cl100k rules 1-2 (contraction + word) and 4-7 (other + whitespace); rule 3 is the digit rule.
const CL100K_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|";
const CL100K_SUF: &str = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn cl100k_digit_cap(pattern: &str) -> Option<usize> {
    match pattern.strip_prefix(CL100K_PRE)?.strip_suffix(CL100K_SUF)? {
        r"\p{N}{1,3}" => Some(3),
        r"\p{N}{1,2}" => Some(2),
        r"\p{N}" => Some(1),
        r"\p{N}+" => Some(usize::MAX),
        _ => None,
    }
}

/// If `pattern` is a recognized GPT pre-tokenization regex, name the native FSM that reproduces its
/// `Isolated` split byte-for-byte. GPT-2, o200k and tekken are matched exactly; the cl100k family is
/// matched structurally ([`cl100k_digit_cap`]) so digit-cap variants (Qwen2 …) unroll too. An
/// unrecognized pattern → `None` (the SysRegex / fancy-regex path handles it).
pub fn gpt_fsm(pattern: &str) -> Option<GptFsm> {
    if pattern == GPT2 {
        Some(GptFsm::Gpt2)
    } else if pattern == O200K {
        Some(GptFsm::O200k)
    } else if pattern == TEKKEN {
        Some(GptFsm::Tekken)
    } else {
        cl100k_digit_cap(pattern).map(|digit_cap| GptFsm::Cl100k { digit_cap })
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
        let mut spans = vec![atomsplit::fsm::Span::default(); bytes.len() + 1];
        let n = match self.0 {
            GptFsm::Gpt2 => atomsplit::fsm::fsm_byte_level(bytes, &tags, &mut spans),
            GptFsm::Cl100k { digit_cap } => {
                atomsplit::fsm::fsm_cl100k_cap(bytes, &tags, &mut spans, digit_cap)
            }
            GptFsm::O200k => atomsplit::fsm::fsm_o200k(bytes, &tags, &mut spans),
            GptFsm::Tekken => atomsplit::fsm::fsm_tekken(bytes, &tags, &mut spans),
        };
        Ok(spans[..n]
            .iter()
            .map(|sp| ((sp.start as usize, sp.end as usize), true))
            .collect())
    }
}

// deepseek-v4's pre-tokenizer is a `Sequence` of these three Isolated `Split`s (+ a byte-map
// `ByteLevel`), which `atomsplit::fsm::fsm_deepseek` collapses into one pass. Byte-exact with the
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
/// `atomsplit::fsm::fsm_deepseek` reproduces the whole composed Isolated split in one pass.
pub fn is_deepseek(p0: &str, p1: &str, p2: &str) -> bool {
    p0 == DS_NUM && p1 == DS_CJK && p2 == DS_BIG
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
}
