//! Trains a BERT-style WordPiece tokenizer from scratch and saves it as a `tokenizer.json`.
//!
//!   cargo run --release -p tk-train --example train_from_scratch -- tokenizer.json corpus.txt [more.txt ...]

use tk_encode::DecoderRuntime;
use tk_encode::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tk_encode::normalizers::bert::BertNormalizer;
use tk_encode::pipeline::{PipelineNormalizer, PipelinePreTokenizer};
use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_train::{PostProcessorSpec, TemplateSpec, TokenizerTrainer, WordPieceTrainer};

fn main() -> tk_encode::Result<()> {
    let mut args = std::env::args().skip(1);
    let output = args
        .next()
        .expect("usage: train_from_scratch <output.json> <corpus.txt>...");
    let files: Vec<String> = args.collect();

    let tokenizer = TokenizerTrainer::builder(WordPieceTrainer::builder().build())
        .normalizers(vec![PipelineNormalizer::Bert(BertNormalizer::new(
            true, true, None, true,
        ))])
        .pre_tokenizer(PipelinePreTokenizer::Bert(BertPreTokenizer))
        .decoder(DecoderRuntime::WordPiece(WordPieceDecoder::new(
            "##".into(),
            true,
        )))
        .special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        .unk_token("[UNK]")
        .post_processor(bert_template())
        .vocab_size(30_000)
        .build()?
        .train_files(&files)?;

    std::fs::write(&output, tk_serialize::to_json(&tokenizer)?)?;
    Ok(())
}

/// `[CLS] A [SEP]` for one sequence, `[CLS] A [SEP] B [SEP]` for a pair, with B's tokens typed 1.
fn bert_template() -> PostProcessorSpec {
    PostProcessorSpec {
        single: TemplateSpec {
            prefix: vec![("[CLS]".into(), 0)],
            infix: vec![],
            suffix: vec![("[SEP]".into(), 0)],
            a_type_id: 0,
            b_type_id: None,
        },
        pair: TemplateSpec {
            prefix: vec![("[CLS]".into(), 0)],
            infix: vec![("[SEP]".into(), 0)],
            suffix: vec![("[SEP]".into(), 1)],
            a_type_id: 0,
            b_type_id: Some(1),
        },
    }
}
