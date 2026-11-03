//! The fields of each component's JSON in `tokenizer.json`, for the bindings' code generators.
//!
//! A binding needs one class per component, with a constructor and an accessor per field. A
//! generator reads [`NORMALIZERS`], [`PRE_TOKENIZERS`] and [`DECODERS`] and writes those classes.
//!
//! Registering a component is one entry in its kind's `components!` list: its doc, its JSON
//! `"type"`, and its fields with their constructor defaults. A component with no fields is its
//! `"type"` alone. Post-processors and models are not listed: their fields are templates and
//! vocabularies, which no [`FieldType`] describes.

/// The schemas of one component kind, such as the normalizers.
pub struct KindSchema {
    /// The kind's class name, such as `"Normalizer"`.
    pub kind: &'static str,
    /// The binding's submodule holding the kind's classes, such as `"normalizers"`.
    pub module: &'static str,
    /// The kind's `Sequence`, or `None` for a kind with none.
    pub sequence: Option<SequenceSchema>,
    /// The concrete components, other than `Sequence`.
    pub components: &'static [ComponentSchema],
}

/// A kind's `Sequence`, which runs several components of the kind in turn.
pub struct SequenceSchema {
    /// The JSON key listing the items, such as `"normalizers"`.
    pub items_key: &'static str,
    pub doc: &'static str,
}

/// The JSON fields of one component.
pub struct ComponentSchema {
    /// The JSON `"type"`, such as `"Strip"`.
    pub json_type: &'static str,
    /// The class's doc, shared by every binding.
    pub doc: &'static str,
    /// The JSON keys, in constructor order.
    pub fields: &'static [FieldSchema],
}

/// One JSON key of a component.
pub struct FieldSchema {
    pub key: &'static str,
    pub field_type: FieldType,
    /// The constructor's default. `None` makes the argument required.
    pub default: Option<FieldValue>,
}

/// The JSON shape of a field.
pub enum FieldType {
    Bool,
    String,
    /// A string of exactly one character.
    Char,
    Usize,
    /// A `bool` or `null`.
    OptionalBool,
    /// One of these strings.
    Choice(&'static [&'static str]),
    /// `{"String": ...}` for a literal or `{"Regex": ...}`.
    Pattern,
    /// Bytes, written as a base64 string.
    Base64Bytes,
}

/// A constructor default, as JSON holds it.
pub enum FieldValue {
    Bool(bool),
    Str(&'static str),
    Char(char),
    Usize(usize),
    Null,
}

/// Builds a `&[ComponentSchema]` from one entry per component:
///
/// ```ignore
/// /// Removes whitespace from the start and the end of the text.
/// Strip { strip_left: Bool = true, strip_right: Bool = true },
/// ```
///
/// The name is the JSON `"type"`. Each field names a [`FieldType`] variant, and a `Choice` gives
/// its strings in parentheses. A default is a single token, of the field's type.
macro_rules! components {
    ($(
        $(#[doc = $doc:literal])+
        $json_type:ident $({
            $($key:ident: $field_type:ident $(($choices:expr))? $(= $default:tt)?),* $(,)?
        })?
    ),* $(,)?) => {
        &[$(ComponentSchema {
            json_type: stringify!($json_type),
            doc: concat!($($doc),+),
            fields: &[$($(FieldSchema {
                key: stringify!($key),
                field_type: components!(@type $field_type $(($choices))?),
                default: components!(@default $field_type $($default)?),
            }),*)?],
        }),*]
    };
    (@type Choice($choices:expr)) => { FieldType::Choice($choices) };
    (@type $field_type:ident) => { FieldType::$field_type };
    (@default $field_type:ident) => { None };
    (@default OptionalBool None) => { Some(FieldValue::Null) };
    (@default OptionalBool $value:tt) => { Some(FieldValue::Bool($value)) };
    (@default Bool $value:tt) => { Some(FieldValue::Bool($value)) };
    (@default String $value:tt) => { Some(FieldValue::Str($value)) };
    (@default Choice $value:tt) => { Some(FieldValue::Str($value)) };
    (@default Char $value:tt) => { Some(FieldValue::Char($value)) };
    (@default Usize $value:tt) => { Some(FieldValue::Usize($value)) };
}

/// How `Split` and `Punctuation` treat the delimiters they find.
const SPLIT_BEHAVIORS: &[&str] = &[
    "Removed",
    "Isolated",
    "MergedWithPrevious",
    "MergedWithNext",
    "Contiguous",
];

/// When a Metaspace puts its replacement character in front of the text.
const PREPEND_SCHEMES: &[&str] = &["always", "first", "never"];

pub const NORMALIZERS: KindSchema = KindSchema {
    kind: "Normalizer",
    module: "normalizers",
    sequence: Some(SequenceSchema {
        items_key: "normalizers",
        doc: "Applies each normalizer in turn.",
    }),
    components: components![
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
            prepend: Choice(PREPEND_SCHEMES) = "always",
            drop_whitespace: Bool = false,
        },
    ],
};

pub const PRE_TOKENIZERS: KindSchema = KindSchema {
    kind: "PreTokenizer",
    module: "pre_tokenizers",
    sequence: Some(SequenceSchema {
        items_key: "pretokenizers",
        doc: "Applies each pre-tokenizer in turn, each one splitting the pieces the previous one \
              made.",
    }),
    components: components![
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
            behavior: Choice(SPLIT_BEHAVIORS) = "Isolated",
            invert: Bool = false,
        },
        /// Splits the text around runs of digits, or around each digit with `individual_digits`.
        Digits { individual_digits: Bool = false },
        /// Splits the text on each punctuation character. `behavior` is that of `Split`.
        Punctuation { behavior: Choice(SPLIT_BEHAVIORS) = "Isolated" },
        /// Splits the text on each `delimiter` character, which is dropped, like
        /// `str.split(delimiter)` without the empty pieces.
        CharDelimiterSplit { delimiter: Char },
        /// Splits the text into pieces of `length` characters. The last piece may be shorter.
        FixedLength { length: Usize = 5 },
    ],
};

pub const DECODERS: KindSchema = KindSchema {
    kind: "Decoder",
    module: "decoders",
    sequence: Some(SequenceSchema {
        items_key: "decoders",
        doc: "Applies each decoder in turn.",
    }),
    components: components![
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
        Metaspace { replacement: Char = '▁', prepend_scheme: Choice(PREPEND_SCHEMES) = "always" },
        /// Decodes CTC output: merges repeated tokens, drops `pad_token`, and turns
        /// `word_delimiter_token` into a space. `cleanup` is that of `WordPiece`.
        CTC { pad_token: String = "<pad>", word_delimiter_token: String = "|", cleanup: Bool = true },
        /// Replaces every match of `pattern` with `content` in each token. A string pattern is a
        /// literal.
        Replace { pattern: Pattern, content: String },
    ],
};
