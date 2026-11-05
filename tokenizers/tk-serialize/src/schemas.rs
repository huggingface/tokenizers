//! The fields of each component's JSON in `tokenizer.json`, for the bindings.
//!
//! A binding needs one class per component, with a constructor and an accessor per field. Each
//! kind's list is a macro that hands its entries to a macro of the binding's, which writes the
//! classes: [`normalizer_schemas!`](crate::normalizer_schemas),
//! [`pre_tokenizer_schemas!`](crate::pre_tokenizer_schemas) and
//! [`decoder_schemas!`](crate::decoder_schemas).
//!
//! ```ignore
//! tk_serialize::normalizer_schemas!(my_classes! { arguments of my_classes });
//! // expands to
//! my_classes! { { arguments of my_classes } /// doc
//!     Strip { strip_left: Bool = true, strip_right: Bool = true }, ... }
//! ```
//!
//! An entry is its doc, its JSON `"type"`, and its fields with their constructor defaults. A field
//! names its JSON shape: `Bool`, `String`, `Char` (a one-character string), `Usize`,
//! `OptionalBool` (a `bool` or `null`), `Choice(choices)` (one of the [`Choice`]s), `Pattern`
//! (`{"String": ...}` for a literal or `{"Regex": ...}`) or `Base64Bytes` (bytes, as a base64
//! string). A default is a single token, and a `Choice`'s default is its name. A field with none
//! is a required argument. A component
//! with no fields is its `"type"` alone.
//!
//! Registering a component is one entry here. Post-processors and models are not listed: their
//! fields are templates and vocabularies, which no field shape describes.

/// One value of a `Choice` field: the `name` the bindings take, and the `json` string
/// `tokenizer.json` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Choice {
    pub name: &'static str,
    pub json: &'static str,
}

const fn choice(name: &'static str, json: &'static str) -> Choice {
    Choice { name, json }
}

/// How `Split` and `Punctuation` treat the delimiters they find.
pub const SPLIT_BEHAVIORS: &[Choice] = &[
    choice("removed", "Removed"),
    choice("isolated", "Isolated"),
    choice("merged_with_previous", "MergedWithPrevious"),
    choice("merged_with_next", "MergedWithNext"),
    choice("contiguous", "Contiguous"),
];

/// When a Metaspace puts its replacement character in front of the text.
pub const PREPEND_SCHEMES: &[Choice] = &[
    choice("always", "always"),
    choice("first", "first"),
    choice("never", "never"),
];

/// Calls `$callback! { { $args } entries }` with one entry per normalizer.
#[macro_export]
macro_rules! normalizer_schemas {
    ($callback:ident! { $($args:tt)* }) => {
        $callback! {
            { $($args)* }
            /// Lowercases the text.
            Lowercase,
            /// Replaces each byte of the text with its character in the GPT-2 byte-to-unicode
            /// mapping.
            ByteLevel,
            /// Removes combining marks. Use it after `NFD` or `NFKD` to strip accents.
            StripAccents,
            /// Unicode NFC normalization.
            NFC,
            /// Unicode NFD normalization.
            NFD,
            /// Unicode NFKC normalization.
            NFKC,
            /// Unicode NFKD normalization.
            NFKD,
            /// Removes control characters and replaces line breaks, tabs and some invisible
            /// characters with a space, like the Google NMT pipeline.
            Nmt,
            /// Replaces every match of `pattern` with `content`. A string pattern is a literal.
            Replace { pattern: Pattern, content: String },
            /// SentencePiece's normalization rules, compiled into a charsmap (T5, ALBERT, XLM-R).
            Precompiled { precompiled_charsmap: Base64Bytes },
            /// BERT's normalization: control characters removed, spaces around CJK characters,
            /// accents stripped and the text lowercased, each optional. `strip_accents` unset strips
            /// accents when `lowercase` is set.
            BertNormalizer {
                clean_text: Bool = true,
                handle_chinese_chars: Bool = true,
                strip_accents: OptionalBool = None,
                lowercase: Bool = true,
            },
            /// Removes whitespace from the start and the end of the text.
            Strip { strip_left: Bool = true, strip_right: Bool = true },
            /// Adds `prepend` at the start of the text, unless the text is empty.
            Prepend { prepend: String },
            /// SentencePiece's whitespace handling: replaces spaces with `replacement`. `prepend` says
            /// when `replacement` also goes in front of the text: `"always"`, `"first"` (only at the
            /// start of the input) or `"never"`. `drop_whitespace` drops runs of whitespace instead
            /// of replacing them.
            MetaspaceNormalizer {
                replacement: Char = '▁',
                prepend: Choice($crate::schemas::PREPEND_SCHEMES) = "always",
                drop_whitespace: Bool = false,
            },
        }
    };
}

/// Calls `$callback! { { $args } entries }` with one entry per pre-tokenizer.
#[macro_export]
macro_rules! pre_tokenizer_schemas {
    ($callback:ident! { $($args:tt)* }) => {
        $callback! {
            { $($args)* }
            /// Splits the text into runs of word characters and runs of punctuation, like the regex
            /// `\w+|[^\w\s]+`. Whitespace is dropped.
            Whitespace,
            /// Splits the text on whitespace, like `str.split` with no arguments.
            WhitespaceSplit,
            /// Splits the text on whitespace, and isolates each punctuation character, like BERT.
            BertPreTokenizer,
            /// Splits the text where the Unicode script changes, like SentencePiece. Hiragana and
            /// Katakana count as Han.
            UnicodeScripts,
            /// Splits the text on every match of `pattern`, the matches being the delimiters. A
            /// string pattern is a literal. `invert` splits on what does not match instead.
            Split {
                pattern: Pattern,
                behavior: Choice($crate::schemas::SPLIT_BEHAVIORS) = "isolated",
                invert: Bool = false,
            },
            /// Splits the text around runs of digits, or around each digit with `individual_digits`.
            Digits { individual_digits: Bool = false },
            /// Splits the text on each punctuation character. `behavior` is that of `Split`.
            Punctuation { behavior: Choice($crate::schemas::SPLIT_BEHAVIORS) = "isolated" },
            /// Splits the text on each `delimiter` character, which is dropped, like
            /// `str.split(delimiter)` without the empty pieces.
            CharDelimiterSplit { delimiter: Char },
            /// Splits the text into pieces of `length` characters. The last piece may be shorter.
            FixedLength { length: Usize = 5 },
        }
    };
}

/// Calls `$callback! { { $args } entries }` with one entry per decoder.
#[macro_export]
macro_rules! decoder_schemas {
    ($callback:ident! { $($args:tt)* }) => {
        $callback! {
            { $($args)* }
            /// Maps each character of the GPT-2 byte-to-unicode alphabet back to its byte, and
            /// decodes the bytes as UTF-8.
            ByteLevel,
            /// Turns each run of `<0xNN>` byte tokens into the text those bytes encode in UTF-8. A
            /// run that is not valid UTF-8 becomes one `U+FFFD` per byte.
            ByteFallback,
            /// Joins every token into one string, with no separator.
            Fuse,
            /// Removes up to `start` leading and `stop` trailing `content` characters from each
            /// token.
            Strip { content: Char = ' ', start: Usize = 0, stop: Usize = 0 },
            /// Joins BPE tokens, turning each `suffix` that ends a word into a space.
            BPEDecoder { suffix: String = "</w>" },
            /// Joins WordPiece tokens, gluing each token that starts with `prefix` to the one before
            /// it. `cleanup` also removes the spaces English tokenization puts before punctuation and
            /// contractions.
            WordPiece { prefix: String = "##", cleanup: Bool = true },
            /// Turns each `replacement` character back into a space. Unless `prepend_scheme` is
            /// `"never"`, the first token's `replacement` characters are dropped instead.
            Metaspace {
                replacement: Char = '▁',
                prepend_scheme: Choice($crate::schemas::PREPEND_SCHEMES) = "always",
            },
            /// Decodes CTC output: merges repeated tokens, drops `pad_token`, and turns
            /// `word_delimiter_token` into a space. `cleanup` is that of `WordPiece`.
            CTC { pad_token: String = "<pad>", word_delimiter_token: String = "|", cleanup: Bool = true },
            /// Replaces every match of `pattern` with `content` in each token. A string pattern is a
            /// literal.
            Replace { pattern: Pattern, content: String },
        }
    };
}
