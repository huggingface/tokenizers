//! The `model` object, with an explicit `"type"` on every one of them.
use super::writer::Out;
use tk_encode::pipeline::PipelineModel;
use tk_encode::tokenizer::Result;

pub(super) fn write_model(out: &mut Out, model: &PipelineModel) -> Result<()> {
    match model {
        PipelineModel::BPE(bpe) => {
            let config = bpe.to_config()?;
            out.obj_open();
            out.type_tag("BPE");
            out.field_opt_str("unk_token", config.unk_token.as_deref());
            out.field_opt_str(
                "continuing_subword_prefix",
                config.continuing_subword_prefix.as_deref(),
            );
            out.field_opt_str("end_of_word_suffix", config.end_of_word_suffix.as_deref());
            out.field_bool("fuse_unk", config.fuse_unk);
            out.field_bool("byte_fallback", config.byte_fallback);
            out.field_bool("ignore_merges", config.ignore_merges);
            // Byte-level is a property of the vocabulary encoding, so it is stated on the model.
            // A `ByteLevel` pre-tokenizer used to be the only place it lived, which is why the
            // reader had to work out where that tag sat in a `Sequence`.
            out.field_bool("byte_level", bpe.is_byte_level());
            write_vocab_object(out, config.vocab.into_iter().collect());
            out.key("merges");
            out.arr_open();
            // Rank order, which is what decides which merge wins: the array's index *is* the rank.
            for (left, right) in &config.merges {
                out.arr_open();
                out.str(left);
                out.str(right);
                out.arr_close();
            }
            out.arr_close();
            out.obj_close();
        }
        #[cfg(feature = "unigram")]
        PipelineModel::Unigram(unigram) => {
            out.obj_open();
            out.type_tag("Unigram");
            match unigram.unk_id() {
                Some(id) => out.field_usize("unk_id", id),
                None => out.field_null("unk_id"),
            }
            out.key("vocab");
            out.arr_open();
            // Index order, because a Unigram id *is* the index into this array.
            for (token, score) in unigram.vocab() {
                out.arr_open();
                out.str(token);
                // The whole reason `float_literal` exists: these decide the Viterbi lattice, and a
                // one-ULP shift in one of them can move an id.
                out.f64(*score)?;
                out.arr_close();
            }
            out.arr_close();
            out.field_bool("byte_fallback", unigram.byte_fallback());
            out.obj_close();
        }
        #[cfg(feature = "wordpiece")]
        PipelineModel::WordPiece(wordpiece) => {
            out.obj_open();
            out.type_tag("WordPiece");
            // The reader requires all four fields, so a missing `unk_token` is an error rather than
            // a `null`: the lowering keeps the id, and a name it could not resolve is unrecoverable.
            let unk_token = wordpiece.unk_token().ok_or_else(|| -> tk_encode::Error {
                "this `WordPiece` model's `unk_token` is not in its vocabulary, so the name it was \
                 configured with is gone and cannot be written back"
                    .into()
            })?;
            out.field_str("unk_token", unk_token);
            out.field_str(
                "continuing_subword_prefix",
                wordpiece.continuing_subword_prefix(),
            );
            out.field_usize(
                "max_input_chars_per_word",
                wordpiece.max_input_chars_per_word(),
            );
            write_vocab_object(out, wordpiece.vocab());
            out.obj_close();
        }
        #[cfg(feature = "wordlevel")]
        PipelineModel::WordLevel(wordlevel) => {
            out.obj_open();
            out.type_tag("WordLevel");
            out.field_str("unk_token", &wordlevel.unk_token);
            write_vocab_object(
                out,
                wordlevel
                    .vocab
                    .iter()
                    .map(|(token, &id)| (token.clone(), id))
                    .collect(),
            );
            out.obj_close();
        }
    }
    Ok(())
}

/// `{"token": id}`, sorted by id.
fn write_vocab_object(out: &mut Out, mut vocab: Vec<(String, u32)>) {
    // By id, then by token, so a vocabulary that repeats an id still has one spelling.
    vocab.sort_unstable_by(|(a_token, a_id), (b_token, b_id)| {
        a_id.cmp(b_id).then_with(|| a_token.cmp(b_token))
    });
    out.key("vocab");
    out.obj_open();
    for (token, id) in &vocab {
        out.field_u32(token, *id);
    }
    out.obj_close();
}
