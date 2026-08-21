//! The `decoder` object, written straight from the `DecoderRuntime` the reader built.
//!
//! The easiest component of the lot: the decoder is off the encode path, so nothing about it was
//! rewritten or fused on the way in. Every variant keeps its own fields, and this is a
//! one-arm-per-variant transcription back out.

use super::normalizers::write_replace_pattern;
use super::writer::Out;
use tk_encode::decoders::DecoderRuntime;
use tk_encode::pre_tokenizers::metaspace::PrependScheme;
use tk_encode::tokenizer::Result;

pub(super) fn write_decoder(out: &mut Out, decoder: Option<&DecoderRuntime>) -> Result<()> {
    match decoder {
        // A missing `decoder` reads back as `None`, so absent is exact.
        None => out.null(),
        Some(decoder) => write_one(out, decoder),
    }
    Ok(())
}

fn write_one(out: &mut Out, decoder: &DecoderRuntime) {
    match decoder {
        DecoderRuntime::ByteLevel(_) => {
            out.obj_open();
            out.type_tag("ByteLevel");
            out.obj_close();
        }
        DecoderRuntime::Replace(replace) => {
            out.obj_open();
            out.type_tag("Replace");
            write_replace_pattern(out, replace.pattern());
            out.field_str("content", replace.content());
            out.obj_close();
        }
        DecoderRuntime::ByteFallback(_) => {
            out.obj_open();
            out.type_tag("ByteFallback");
            out.obj_close();
        }
        DecoderRuntime::Fuse(_) => {
            out.obj_open();
            out.type_tag("Fuse");
            out.obj_close();
        }
        DecoderRuntime::Strip(strip) => {
            out.obj_open();
            out.type_tag("Strip");
            out.field_str("content", strip.content.encode_utf8(&mut [0; 4]));
            out.field_usize("start", strip.start);
            out.field_usize("stop", strip.stop);
            out.obj_close();
        }
        // The one tag that is not the type's name, in the file as much as here.
        DecoderRuntime::BPE(bpe) => {
            out.obj_open();
            out.type_tag("BPEDecoder");
            out.field_str("suffix", &bpe.suffix);
            out.obj_close();
        }
        DecoderRuntime::WordPiece(wordpiece) => {
            out.obj_open();
            out.type_tag("WordPiece");
            out.field_str("prefix", &wordpiece.prefix);
            out.field_bool("cleanup", wordpiece.cleanup);
            out.obj_close();
        }
        DecoderRuntime::Metaspace(metaspace) => {
            out.obj_open();
            out.type_tag("Metaspace");
            out.field_str(
                "replacement",
                metaspace.get_replacement().encode_utf8(&mut [0; 4]),
            );
            // Canonical spelling. Unlike the *pre-tokenizer*, a decoder keeps all three schemes:
            // nothing here has to be expressible as a normalizer, so `first` survives.
            out.field_str(
                "prepend_scheme",
                match metaspace.prepend_scheme {
                    PrependScheme::Always => "always",
                    PrependScheme::First => "first",
                    PrependScheme::Never => "never",
                },
            );
            out.field_bool("split", metaspace.split);
            out.obj_close();
        }
        DecoderRuntime::CTC(ctc) => {
            out.obj_open();
            out.type_tag("CTC");
            out.field_str("pad_token", &ctc.pad_token);
            out.field_str("word_delimiter_token", &ctc.word_delimiter_token);
            out.field_bool("cleanup", ctc.cleanup);
            out.obj_close();
        }
        DecoderRuntime::Sequence(members) => {
            out.obj_open();
            out.type_tag("Sequence");
            out.key("decoders");
            out.arr_open();
            for member in members {
                write_one(out, member);
            }
            out.arr_close();
            out.obj_close();
        }
    }
}
