//! Lists of components and their JSON fields.
//!
//! The Python bindings have one class per component, such as `normalizers.Strip`.
//! Each Python class has a constructor, and a getter and a setter per field.
//! To avoid writing and maintaining that code by hand, we use Rust macros in the
//! bindings to generate classes.
//!
//! This works because most components are plain data.
//! Their Python class is only a view over a small JSON object, with no additional logic.
//! Models and post-processors are different: their fields are vocabularies and templates,
//! with logic around them. Model and post-processor classes are written by hand.
//!
//! This module defines lists of components with their JSON fields. Those lists are
//! used by the Python bindings macros to generate code systematically.
//!
//! # Component definitions
//!
//! Here is an example of a component definition:
//!
//! ```text
//! /// Removes whitespace from the start and the end of the text.
//! Strip { strip_left: Bool = true, strip_right: Bool = true },
//! ```
//!
//! From this definition, the Python bindings generate the class `normalizers.Strip`:
//!
//! ```python
//! @final
//! class Strip(Normalizer):
//!     """
//!     Removes whitespace from the start and the end of the text.
//!     """
//!     def __new__(cls, /, strip_left: bool = True, strip_right: bool = True) -> Strip:
//!         ...
//!     @property
//!     def strip_left(self, /) -> bool:
//!         ...
//!     @strip_left.setter
//!     def strip_left(self, /, strip_left: bool) -> None:
//!         ...
//!     @property
//!     def strip_right(self, /) -> bool:
//!         ...
//!     @strip_right.setter
//!     def strip_right(self, /, strip_right: bool) -> None:
//!         ...
//! ```
//!
//! The corresponding JSON object looks like:
//! ```json
//! {"type": "Strip", "strip_left": false, "strip_right": true}
//! ```
//!
//! A component definition has three parts:
//!
//! - **The doc** becomes the class docstring.
//! - **The `"type"`**, here `Strip`, is also the class name.
//! - **The fields**, in braces, if the component has any.
//!
//! Each field has a key, a field type and an optional default.
//! The key names the field in the JSON, in the constructor and on the class.
//! The field type says which JSON value the field holds:
//!
//! - `Bool`: a bool.
//! - `OptionalBool`: a bool or `null`.
//! - `Usize`: a non-negative integer.
//! - `String`: a string.
//! - `Char`: a string of one character.
//! - `Choice(choices)`: the `json` of one of `choices`, a list of [`Choice`].
//!   The default is a `name`.
//! - `Pattern`: `{"String": "..."}` or `{"Regex": "..."}`.
//! - `Base64Bytes`: bytes, written as a base64 string.
//!
//! A field with no default is a required argument.
//! The getter also returns the default when the JSON leaves the field out.
//!
//! # Component lists
//!
//! There is one list per component kind:
//!
//! - [`normalizer_schemas!`](crate::normalizer_schemas)
//! - [`pre_tokenizer_schemas!`](crate::pre_tokenizer_schemas)
//! - [`decoder_schemas!`](crate::decoder_schemas)
//!
//! `Sequence` components have special handling in the Python bindings.
//!
//! The Python bindings call the list macros with their own `component_module!` macro:
//!
//! ```ignore
//! tk::normalizer_schemas!(component_module! {
//!     kind: Normalizer,
//!     module: normalizers,
//!     ...
//! });
//! ```
//!
//! # Adding a component to the bindings
//!
//! 1. Make the reader in `from_json` accept the new `"type"`.
//! 2. Add a definition to the list of its kind. The Python class gets automatically generated.

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

/// Calls a macro with the definition of every normalizer.
///
/// `normalizer_schemas!(my_macro! { args })` expands to:
///
/// ```rust,ignore
/// my_macro! {
///     { args }
///     /// Lowercases the text.
///     Lowercase,
///     /// Removes whitespace from the start and the end of the text.
///     Strip { strip_left: Bool = true, strip_right: Bool = true },
///     ...
/// }
/// ```
///
/// The [module doc](crate::schemas) explains definitions and their fields.
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

/// Calls a macro with the definition of every pre-tokenizer.
///
/// `pre_tokenizer_schemas!(my_macro! { args })` expands to:
///
/// ```rust,ignore
/// my_macro! {
///     { args }
///     /// Splits the text into runs of word characters and runs of punctuation, ...
///     Whitespace,
///     /// Splits the text around runs of digits, or around each digit with `individual_digits`.
///     Digits { individual_digits: Bool = false },
///     ...
/// }
/// ```
///
/// The [module doc](crate::schemas) explains definitions and their fields.
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
            /// `behavior` is `"removed"`, `"isolated"`, `"merged_with_previous"`,
            /// `"merged_with_next"` or `"contiguous"`.
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

/// Calls a macro with the definition of every decoder.
///
/// `decoder_schemas!(my_macro! { args })` expands to:
///
/// ```rust,ignore
/// my_macro! {
///     { args }
///     /// Maps each character of the GPT-2 byte-to-unicode alphabet back to its byte, ...
///     ByteLevel,
///     /// Joins every token into one string, with no separator.
///     Fuse,
///     ...
/// }
/// ```
///
/// The [module doc](crate::schemas) explains definitions and their fields.
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

#[cfg(all(
    test,
    feature = "deserialize",
    feature = "normalizers",
    feature = "unicode-scripts"
))]
mod tests;
