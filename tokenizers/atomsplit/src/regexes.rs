//! Canonical GPT pre-tokenization regexes — the **specification** each `fsm` reproduces byte-for-byte
//! under an `Isolated` split. This is the single source of truth: the benches, the parity oracle
//! (`tests/parity.rs`), and tk-encode's runtime recognizer all reference these consts, so the pattern a
//! tokenizer ships, the pattern the FSM is tested against, and the pattern the pipeline recognizes can
//! never drift apart. `atomsplit` never *runs* these at runtime — it works off the tag stream; the
//! consts only document (and gate the tests of) the contract the FSMs implement.

/// GPT-2 / ByteLevel. Reproduced by [`crate::fsm::fsm_byte_level`].
pub const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// cl100k_base (tiktoken) / Llama-3. Reproduced by [`crate::fsm::fsm_cl100k`]. Rule 3's `\p{N}{1,3}`
/// digit cap is the only free knob — the cl100k *family* (Qwen2's `\p{N}`, …) is recognized structurally
/// around it (see tk-encode's `unrolled_regex`).
pub const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// o200k_base / GPT-4o (case-aware letter runs + contraction suffix + `[\r\n/]` tail). Reproduced by
/// [`crate::fsm::fsm_o200k`].
pub const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// deepseek-v3 `Sequence`: `NUM` → `CJK` → `BIG`, each `Isolated`. Reproduced by
/// [`crate::fsm::fsm_deepseek`] as one pass.
pub const DEEPSEEK_NUM: &str = r"\p{N}{1,3}";
pub const DEEPSEEK_CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
pub const DEEPSEEK_BIG: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// The deepseek chain in application order — convenience for the multi-regex reference.
pub const DEEPSEEK: &[&str] = &[DEEPSEEK_NUM, DEEPSEEK_CJK, DEEPSEEK_BIG];
