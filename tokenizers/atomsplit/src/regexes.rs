//! Canonical GPT pre-tokenization regexes — the **specification** each `fsm` reproduces byte-for-byte
//! under an `Isolated` split. This is the single source of truth: the benches, the parity oracle
//! (`tests/parity.rs`), and tk-encode's runtime recognizer all reference these consts, so the pattern a
//! tokenizer ships, the pattern the FSM is tested against, and the pattern the pipeline recognizes can
//! never drift apart. `atomsplit` never *runs* these at runtime — it works off the tag stream; the
//! consts only document (and gate the tests of) the contract the FSMs implement.

/// GPT-2 / ByteLevel. Reproduced by [`crate::fsm::fsm_byte_level`].
pub const GPT2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// cl100k_base (tiktoken) / Llama-3. Reproduced by [`crate::fsm::fsm_cl100k`]. Rule 3's `\p{N}{1,3}`
/// digit cap is the only free knob — the cl100k *family* (Qwen2's `\p{N}`, …) is recognized structurally
/// around it (see tk-encode's `unrolled_regex`).
pub const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// o200k_base / GPT-4o (case-aware letter runs + contraction suffix + `[\r\n/]` tail). Reproduced by
/// [`crate::fsm::fsm_o200k`].
pub const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Mistral tekken (mistral-small-4 / mistral-4). o200k's grammar with two changes: letter tokens take
/// no contraction suffix, and the digit rule is a bare `\p{N}` — one token per digit. Reproduced by
/// [`crate::fsm::fsm_tekken`].
pub const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// deepseek `Sequence`: `NUM` → `CJK` → `BIG`, each `Isolated`. Fused into one pass by
/// [`crate::fsm::fsm_deepseek`]; each also stands alone ([`crate::fsm::fsm_deepseek_num`] /
/// [`crate::fsm::fsm_deepseek_cjk`] / [`crate::fsm::fsm_deepseek_big`]).
///
/// These are byte-for-byte the strings deepseek's `tokenizer.json` ships, so a loader can recognize a
/// `Split` by string equality — hence `BIG` carries LITERAL CR/LF (spliced in with `concat!`) rather
/// than the `\r` / `\n` escapes an equivalent regex could use.
pub const DEEPSEEK_NUM: &str = r"\p{N}{1,3}";
pub const DEEPSEEK_CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
pub const DEEPSEEK_BIG: &str = concat!(
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^"##,
    "\r\n",
    r##"\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+["##,
    "\r\n",
    r##"]*|\s*["##,
    "\r\n",
    r##"]+|\s+(?!\S)|\s+"##,
);

/// The deepseek chain in application order — convenience for the multi-regex reference.
pub const DEEPSEEK: &[&str] = &[DEEPSEEK_NUM, DEEPSEEK_CJK, DEEPSEEK_BIG];
