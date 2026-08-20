//! The `model` object: one reader per model kind, plus the shared `{"token": id}` vocabulary.

use crate::json::Json;
use tk_encode::models::bpe::{Merges, PipelineBpeOptions, Vocab};
use tk_encode::tokenizer::Result;

/// The three parts [`PipelineBPE::from_vocab_and_merges`] takes, read out of the config. They are
/// returned rather than consumed here so that the vocabulary can serve as [`VocabOnly`] first.
pub(super) fn read_bpe(
    cfg: &Json<'_>,
    with_byte_level: bool,
) -> Result<(Vocab, Merges, PipelineBpeOptions)> {
    let vocab = read_vocab_object(cfg)?;

    let merges_arr = cfg
        .get_some("merges")
        .and_then(Json::as_arr)
        .ok_or_else(|| -> tk_encode::Error { "BPE model has no `merges` array".into() })?;
    let mut merges: Merges = Vec::with_capacity(merges_arr.len());
    for entry in merges_arr {
        match entry {
            // Canonical: ["a", "b"].
            Json::Arr(pair) if pair.len() == 2 => {
                let (a, b) = (pair[0].as_str(), pair[1].as_str());
                match (a, b) {
                    (Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                    _ => return Err("a merge pair is not a pair of strings".into()),
                }
            }
            // Legacy: "a b". Split on the first space, the way the config path does. Ambiguous when
            // a token contains a space, which is exactly why pairs became canonical.
            Json::Str(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| -> tk_encode::Error {
                    format!("legacy merge {s:?} has no space to split on").into()
                })?;
                merges.push((a.to_string(), b.to_string()));
            }
            _ => return Err("a merge is neither a pair nor a string".into()),
        }
    }

    // Each option is left at its default unless the config names it, which is what the builder's
    // `if let Some(..)` chain used to say. A key that is present but null reads as absent, exactly
    // as it did.
    let options = PipelineBpeOptions {
        dropout: cfg
            .get_some("dropout")
            .and_then(Json::as_f64)
            .map(|v| v as f32),
        unk_token: cfg
            .get_some("unk_token")
            .and_then(Json::as_str)
            .map(str::to_string),
        continuing_subword_prefix: cfg
            .get_some("continuing_subword_prefix")
            .and_then(Json::as_str)
            .map(str::to_string),
        end_of_word_suffix: cfg
            .get_some("end_of_word_suffix")
            .and_then(Json::as_str)
            .map(str::to_string),
        fuse_unk: cfg
            .get_some("fuse_unk")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        byte_fallback: cfg
            .get_some("byte_fallback")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        ignore_merges: cfg
            .get_some("ignore_merges")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        with_byte_level,
        ..PipelineBpeOptions::default()
    };
    Ok((vocab, merges, options))
}

/// `{"token": id, ...}`, which is how every model but Unigram spells its vocabulary.
fn read_vocab_object(cfg: &Json<'_>) -> Result<Vocab> {
    let vocab_obj = cfg
        .get_some("vocab")
        .and_then(Json::as_obj)
        .ok_or_else(|| -> tk_encode::Error { "model has no `vocab` object".into() })?;
    let mut vocab = Vocab::with_capacity(vocab_obj.len());
    for (token, id) in vocab_obj {
        let id = id.as_u32().ok_or_else(|| -> tk_encode::Error {
            format!("vocab entry {token:?} has a bad id").into()
        })?;
        vocab.insert(token.to_string(), id);
    }
    Ok(vocab)
}

/// All four fields are required, exactly as on the config path — `WordPiece`'s deserializer collects
/// missing ones and errors, so a config that omits `unk_token` has never loaded.
#[cfg(feature = "wordpiece")]
pub(super) fn read_wordpiece(cfg: &Json<'_>) -> Result<tk_encode::models::wordpiece::WordPiece> {
    let vocab = read_vocab_object(cfg)?;
    let field = |name: &str| -> Result<&str> {
        cfg.get_some(name)
            .and_then(Json::as_str)
            .ok_or_else(|| format!("WordPiece model has no `{name}`").into())
    };
    let max_input_chars_per_word = cfg
        .get_some("max_input_chars_per_word")
        .and_then(Json::as_usize)
        .ok_or_else(|| -> tk_encode::Error {
            "WordPiece model has no `max_input_chars_per_word`".into()
        })?;
    tk_encode::models::wordpiece::WordPiece::builder()
        .vocab(vocab)
        .unk_token(field("unk_token")?.to_string())
        .continuing_subword_prefix(field("continuing_subword_prefix")?.to_string())
        .max_input_chars_per_word(max_input_chars_per_word)
        .build()
}

/// Unigram's vocab is an array of `[token, score]` pairs, and the scores decide the lattice, so a
/// score that is not a number is an error rather than a `0.0`.
///
/// ## The scores do not bit-match the config path, and this reader is the correct one
///
/// [`tk_encode::tokenizer::json`] parses numbers with `f64::from_str`, which is correctly rounded.
/// `serde_json` without its `float_roundtrip` feature does not: it accumulates the digits into a
/// `u64` and divides by a power of ten, which is off by one ULP for 8334 of t5's 32100 scores. Every
/// score in that file is exactly an `f32` widened to `f64`, and `from_str` lands on it; `serde_json`
/// lands next to it.
///
/// The lattice is a sum of scores, so one ULP only matters where two segmentations very nearly tie —
/// 2 ids out of 1.25 M on the english fixture, always a run of one repeated character. `json_oracle`
/// reports those cells as `SLIM MISMATCH`, and it is the *config* side that is imprecise. Closing it
/// means turning on `serde_json/float_roundtrip` and re-recording the digests, which is a decision
/// about the config path, not something to paper over here by reproducing the error.
#[cfg(feature = "unigram")]
pub(super) fn read_unigram(cfg: &Json<'_>) -> Result<tk_encode::models::unigram::Unigram> {
    let entries = cfg
        .get_some("vocab")
        .and_then(Json::as_arr)
        .ok_or_else(|| -> tk_encode::Error { "Unigram model has no `vocab` array".into() })?;
    let mut vocab = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = entry
            .as_arr()
            .filter(|p| p.len() == 2)
            .ok_or_else(|| -> tk_encode::Error {
                "a Unigram vocab entry is not a [token, score] pair".into()
            })?;
        let token = pair[0].as_str().ok_or_else(|| -> tk_encode::Error {
            "a Unigram vocab token is not a string".into()
        })?;
        let score = pair[1].as_f64().ok_or_else(|| -> tk_encode::Error {
            "a Unigram vocab score is not a number".into()
        })?;
        vocab.push((token.to_string(), score));
    }
    // `get_some`, so an explicit `"unk_id": null` reads as "no unk", which is what it means.
    let unk_id = match cfg.get_some("unk_id") {
        Some(v) => Some(v.as_usize().ok_or_else(|| -> tk_encode::Error {
            "Unigram `unk_id` is not a usable index".into()
        })?),
        None => None,
    };
    let byte_fallback = cfg
        .get_some("byte_fallback")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    tk_encode::models::unigram::Unigram::from(vocab, unk_id, byte_fallback)
}

#[cfg(feature = "wordlevel")]
pub(super) fn read_wordlevel(cfg: &Json<'_>) -> Result<tk_encode::models::wordlevel::WordLevel> {
    let vocab = read_vocab_object(cfg)?;
    let unk_token = cfg
        .get_some("unk_token")
        .and_then(Json::as_str)
        .ok_or_else(|| -> tk_encode::Error { "WordLevel model has no `unk_token`".into() })?;
    tk_encode::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk_token.to_string())
        .build()
}
