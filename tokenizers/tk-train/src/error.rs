use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("vocab_size is required")]
    MissingVocabSize,
    #[error("unk_token {0:?} is not one of the special tokens")]
    UnkTokenNotSpecial(String),
    #[error("the post-processor uses {0:?}, which is not one of the special tokens")]
    TemplateTokenNotSpecial(String),
    #[error(
        "vocab_size {vocab_size} leaves no room for a learned token after {reserved} reserved ones"
    )]
    VocabTooSmall { vocab_size: usize, reserved: usize },
    #[error(
        "byte-level text needs a ByteLevel decoder, or decoded text stays in the byte alphabet \
         (a space comes back as 'Ġ')"
    )]
    ByteLevelDecoderMissing,
    #[error(
        "the ByteLevel decoder needs byte-level text, from `byte_level()` or the ByteLevel \
         normalizer, or it corrupts characters like 'é'"
    )]
    ByteLevelDecoderWithoutByteLevel,
    #[error(
        "`byte_level()` needs a BPE trainer, the one model that reads text as bytes; other models \
         can read byte-level text through the ByteLevel normalizer"
    )]
    ByteLevelUnsupported,
    #[error(
        "limit_alphabet {limit_alphabet} drops some of the 256 byte tokens a byte-level vocabulary \
         needs"
    )]
    AlphabetTooSmall { limit_alphabet: usize },
    #[error("`byte_level()` and the ByteLevel normalizer both turn text into bytes; use one")]
    ByteLevelTwice,
    #[error("the pad token {0:?} is not one of the special tokens")]
    PadTokenNotSpecial(String),
    #[error("the {role} role names {token:?}, which is not one of the special tokens")]
    RoleTokenNotSpecial { role: String, token: String },
    #[error("the post-processor uses id {0}, which is not in the vocabulary")]
    UnknownTemplateId(u32),
    #[error("sequence too long to pre-tokenize: {0} bytes after normalization, the limit is {max}", max = u32::MAX)]
    SequenceTooLong(usize),
    #[error("this build of tk-train has no {0} trainer")]
    TrainerNotCompiled(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A normalizer, pre-tokenizer or model failed. They all report through `tk_encode::Error`.
    #[error(transparent)]
    Pipeline(#[from] tk_encode::Error),
}

pub type Result<T> = std::result::Result<T, TrainingError>;
