use tk::normalizer::NormalizedString;
use tk::pre_tokenizers::{bert::BertPreTokenizer, PreTokenizerWrapper};
use tk::{PreTokenizedString, PreTokenizer};

#[cxx::bridge(namespace = "huggingface::tokenizers::ffi")]
mod ffi {
    extern "C++" {
        include!("pre_tokenizers.h");
    }

    extern "Rust" {
        type NormalizedString;
        type PreTokenizedString;
        type PreTokenizerWrapper;
        type BertPreTokenizer;

        fn pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString>;

        fn bert_pre_tokenizer() -> Box<BertPreTokenizer>;

        fn pre_tokenize_any(
            pre_tokenizer: &PreTokenizerWrapper,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;

        fn pre_tokenize_bert(
            pre_tokenizer: &BertPreTokenizer,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;
    }
}

fn pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString> {
    Box::new(normalized.clone().into())
}

fn bert_pre_tokenizer() -> Box<BertPreTokenizer> {
    Box::new(BertPreTokenizer)
}

fn pre_tokenize_any(
    pre_tokenizer: &PreTokenizerWrapper,
    pre_tokenized: &mut PreTokenizedString,
) -> tk::Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn pre_tokenize_bert(
    pre_tokenizer: &BertPreTokenizer,
    pre_tokenized: &mut PreTokenizedString,
) -> tk::Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}
