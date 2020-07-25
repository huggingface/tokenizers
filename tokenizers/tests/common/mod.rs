use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{tokenizer::Tokenizer, Model};

#[allow(dead_code)]
pub fn get_empty() -> Tokenizer {
    Tokenizer::new(Box::new(BPE::default()))
}

#[allow(dead_code)]
pub fn get_byte_level_bpe() -> BPE {
    BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_byte_level(add_prefix_space: bool, trim_offsets: bool) -> Tokenizer {
    Tokenizer::new(Box::new(get_byte_level_bpe()))
        .with_pre_tokenizer(Box::new(
            ByteLevel::default().add_prefix_space(add_prefix_space),
        ))
        .with_decoder(Box::new(ByteLevel::default()))
        .with_post_processor(Box::new(ByteLevel::default().trim_offsets(trim_offsets)))
}

#[allow(dead_code)]
pub fn get_bert_wordpiece() -> WordPiece {
    WordPiece::from_files("data/bert-base-uncased-vocab.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_bert() -> Tokenizer {
    let model = Box::new(get_bert_wordpiece());
    const SEP_TOKEN: &'static str = "[SEP]";
    const CLS_TOKEN: &'static str = "[CLS]";
    let sep = model.token_to_id(SEP_TOKEN).unwrap();
    let cls = model.token_to_id(CLS_TOKEN).unwrap();
    let tokenizer = Tokenizer::new(model)
        .with_normalizer(Box::new(BertNormalizer::default()))
        .with_pre_tokenizer(Box::new(BertPreTokenizer))
        .with_decoder(Box::new(WordPieceDecoder::default()))
        .with_post_processor(Box::new(BertProcessing::new(
            (String::from(SEP_TOKEN), sep),
            (String::from(CLS_TOKEN), cls),
        )));

    tokenizer
}
