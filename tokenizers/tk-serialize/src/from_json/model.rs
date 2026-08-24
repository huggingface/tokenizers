use crate::json::Json;
use tk_encode::models::bpe::{BpeConfig, Merges, Vocab};
use tk_encode::tokenizer::Result;

/// TODO: for now we are forced to return the raw vocab to make sure the added tokens are properly
/// created. This will be updated with a better flow.
pub(super) fn read_bpe(cfg: &Json<'_>, byte_level: bool) -> Result<(Vocab, Merges, BpeConfig)> {
    let vocab = read_vocab_object(cfg)?;

    let merges_arr = cfg.need("BPE model", "merges", Json::as_array)?;
    let mut merges: Merges = Vec::with_capacity(merges_arr.len());
    for entry in merges_arr {
        // Canonical: ["a", "b"].
        if let Some(pair) = entry.as_array() {
            match (
                pair.len(),
                pair.first().and_then(Json::as_str),
                pair.get(1).and_then(Json::as_str),
            ) {
                (2, Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                _ => return Err("a merge pair is not a pair of strings".into()),
            }
        // Legacy: "a b". Split on the first space, the way the config path does. Ambiguous when
        // a token contains a space, which is exactly why pairs became canonical.
        } else if let Some(s) = entry.as_str() {
            let (a, b) = s.split_once(' ').ok_or_else(|| -> tk_encode::Error {
                format!("legacy merge {s:?} has no space to split on").into()
            })?;
            merges.push((a.to_string(), b.to_string()));
        } else {
            return Err("a merge is neither a pair nor a string".into());
        }
    }

    // Each option is left at its default unless the config names it, which is what the builder's
    let options = BpeConfig {
        dropout: cfg
            .field("dropout")
            .and_then(Json::as_f64)
            .map(|v| v as f32),
        unk_token: cfg
            .field("unk_token")
            .and_then(Json::as_str)
            .map(str::to_string),
        continuing_subword_prefix: cfg
            .field("continuing_subword_prefix")
            .and_then(Json::as_str)
            .map(str::to_string),
        end_of_word_suffix: cfg
            .field("end_of_word_suffix")
            .and_then(Json::as_str)
            .map(str::to_string),
        fuse_unk: cfg
            .field("fuse_unk")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        byte_fallback: cfg
            .field("byte_fallback")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        ignore_merges: cfg
            .field("ignore_merges")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        byte_level,
        ..BpeConfig::default()
    };
    Ok((vocab, merges, options))
}

fn read_vocab_object(cfg: &Json<'_>) -> Result<Vocab> {
    let entries = cfg
        .field("vocab")
        .and_then(Json::entries)
        .ok_or_else(|| -> tk_encode::Error { "model has no `vocab` object".into() })?;
    let mut vocab = Vocab::with_capacity(entries.len());
    for (token, id) in entries {
        let id = id.as_u32().ok_or_else(|| -> tk_encode::Error {
            format!("vocab entry {token:?} has a bad id").into()
        })?;
        vocab.insert(token.to_string(), id);
    }
    Ok(vocab)
}

#[cfg(feature = "wordpiece")]
pub(super) fn read_wordpiece(cfg: &Json<'_>) -> Result<tk_encode::models::wordpiece::WordPiece> {
    let vocab = read_vocab_object(cfg)?;
    let max_input_chars_per_word = cfg.need(
        "WordPiece model",
        "max_input_chars_per_word",
        Json::as_usize,
    )?;
    tk_encode::models::wordpiece::WordPiece::builder()
        .vocab(vocab)
        .unk_token(
            cfg.need("WordPiece model", "unk_token", Json::as_str)?
                .to_string(),
        )
        .continuing_subword_prefix(
            cfg.need("WordPiece model", "continuing_subword_prefix", Json::as_str)?
                .to_string(),
        )
        .max_input_chars_per_word(max_input_chars_per_word)
        .build()
}

/// Unigram's vocab is an array of `[token, score]` pairs, and the scores decide the lattice, so a
/// score that is not a number is an error rather than a `0.0`.
///
/// [`tk_encode::tokenizer::json`] parses numbers with `f64::from_str`, which is correctly rounded.
/// `serde_json` without its `float_roundtrip` feature does not: it accumulates the digits into a
/// `u64` and divides by a power of ten, which is off by one ULP for 8334 of t5's 32100 scores. Every
/// score in that file is exactly an `f32` widened to `f64`, and `from_str` lands on it; `serde_json`
/// lands next to it.
#[cfg(feature = "unigram")]
pub(super) fn read_unigram(cfg: &Json<'_>) -> Result<tk_encode::models::unigram::Unigram> {
    let entries = cfg.need("Unigram model", "vocab", Json::as_array)?;
    let mut vocab = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair =
            entry
                .as_array()
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
    // `field`, so an explicit `"unk_id": null` reads as "no unk", which is what it means.
    let unk_id = match cfg.field("unk_id") {
        Some(v) => Some(v.as_usize().ok_or_else(|| -> tk_encode::Error {
            "Unigram `unk_id` is not a usable index".into()
        })?),
        None => None,
    };
    let byte_fallback = cfg
        .field("byte_fallback")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    tk_encode::models::unigram::Unigram::from(vocab, unk_id, byte_fallback)
}

#[cfg(feature = "wordlevel")]
pub(super) fn read_wordlevel(cfg: &Json<'_>) -> Result<tk_encode::models::wordlevel::WordLevel> {
    let vocab = read_vocab_object(cfg)?;
    let unk_token = cfg.need("WordLevel model", "unk_token", Json::as_str)?;
    tk_encode::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk_token.to_string())
        .build()
}
