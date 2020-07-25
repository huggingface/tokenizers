use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::{Model, Tokenizer};

#[allow(dead_code)]
pub fn get_empty() -> Tokenizer<BPE, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper> {
    Tokenizer::new(BPE::default())
}

#[allow(dead_code)]
pub fn get_byte_level_bpe() -> BPE {
    BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_byte_level(
    add_prefix_space: bool,
    trim_offsets: bool,
) -> Tokenizer<BPE, NormalizerWrapper, ByteLevel, ByteLevel> {
    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer.with_pre_tokenizer(ByteLevel::default().add_prefix_space(add_prefix_space));
    tokenizer.with_decoder(Box::new(ByteLevel::default()));
    tokenizer.with_post_processor(ByteLevel::default().trim_offsets(trim_offsets));

    tokenizer
}

#[allow(dead_code)]
pub fn get_bert_wordpiece() -> WordPiece {
    WordPiece::from_files("data/bert-base-uncased-vocab.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_bert() -> Tokenizer<WordPiece, BertNormalizer, BertPreTokenizer, BertProcessing> {
    let mut tokenizer = Tokenizer::new(get_bert_wordpiece());
    tokenizer.with_normalizer(BertNormalizer::default());
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    tokenizer.with_decoder(Box::new(WordPieceDecoder::default()));
    tokenizer.with_post_processor(BertProcessing::new(
        (
            String::from("[SEP]"),
            tokenizer.get_model().token_to_id("[SEP]").unwrap(),
        ),
        (
            String::from("[CLS]"),
            tokenizer.get_model().token_to_id("[CLS]").unwrap(),
        ),
    ));

    tokenizer
}
