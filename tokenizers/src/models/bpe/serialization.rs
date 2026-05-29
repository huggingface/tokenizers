use super::{super::OrderedVocabIter, convert_merges_to_hashmap, BpeBuilder, Pair, BPE};
use ahash::AHashMap;
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

/// Resolve a single legacy `merges: ["tok_a tok_b", ...]` entry against the
/// vocabulary.
///
/// The legacy v1 format uses a single ASCII space as the separator between
/// the two tokens of a merge. This is unambiguous as long as neither side
/// contains a space, but if the vocabulary holds tokens with internal
/// spaces (which the v2 array format `[["tok_a", "tok_b"]]` does support),
/// the line is no longer parsable by a naive `split(' ')`.
///
/// This helper recovers a unique `(a, b)` split by checking every possible
/// space-delimited cut and keeping the one for which `a`, `b`, and
/// `a ++ b` are all present in the vocabulary. The common case where the
/// line contains exactly one space is handled by the fast path.
///
/// Returns `Ok(Some((a, b)))` on a successful disambiguation,
/// `Ok(None)` if no candidate split resolved (caller surfaces the original
/// error), and `Err(line)` if more than one candidate split resolved
/// against the vocabulary (genuine ambiguity).
fn disambiguate_legacy_merge(
    line: &str,
    vocab: &AHashMap<String, u32>,
) -> Result<Option<(String, String)>, String> {
    let space_positions: Vec<usize> = line
        .char_indices()
        .filter_map(|(i, c)| if c == ' ' { Some(i) } else { None })
        .collect();
    if space_positions.len() <= 1 {
        // Single-space (or no-space) lines are handled by the existing
        // fast path; nothing to disambiguate here.
        return Ok(None);
    }

    let mut found: Option<(String, String)> = None;
    for cut in space_positions {
        let a = &line[..cut];
        let b = &line[cut + 1..];
        if !vocab.contains_key(a) {
            continue;
        }
        if !vocab.contains_key(b) {
            continue;
        }
        let mut merged = String::with_capacity(a.len() + b.len());
        merged.push_str(a);
        merged.push_str(b);
        if !vocab.contains_key(&merged) {
            continue;
        }
        if found.is_some() {
            return Err(line.to_string());
        }
        found = Some((a.to_string(), b.to_string()));
    }
    Ok(found)
}

/// Convert a legacy `merges: ["..."]` array, disambiguating any entries
/// whose tokens contain internal spaces by consulting the vocabulary.
///
/// Single-space lines, the overwhelmingly common case, are forwarded to
/// `convert_merges_to_hashmap` unchanged. Multi-space lines are resolved
/// against the vocabulary; if exactly one split position satisfies
/// `vocab.contains_key(a) && vocab.contains_key(b) && vocab.contains_key(a ++ b)`,
/// that split wins. Zero or multiple matches surface explicit errors.
fn legacy_merges_to_tuples(
    merges: Vec<String>,
    vocab: &AHashMap<String, u32>,
) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::with_capacity(merges.len());
    let mut needs_fallback: Vec<String> = Vec::with_capacity(merges.len());
    let mut resolved_flags: Vec<Option<(String, String)>> = Vec::with_capacity(merges.len());

    for (rank, line) in merges.iter().enumerate() {
        match disambiguate_legacy_merge(line, vocab) {
            Ok(Some(pair)) => resolved_flags.push(Some(pair)),
            Ok(None) => {
                // Either the line has 0 or 1 spaces (fast path takes it),
                // or none of the multi-space candidate splits resolved
                // against the vocabulary.
                let space_count = line.chars().filter(|c| *c == ' ').count();
                if space_count > 1 {
                    return Err(format!(
                        "Legacy v1 merge entry at rank {rank} `{line}` could not be \
                         resolved against the vocabulary; the line contains multiple \
                         spaces and no candidate split satisfies the vocab triple \
                         `(a, b, a ++ b)`. Re-serialise the tokenizer as v2 array merges."
                    ));
                }
                resolved_flags.push(None);
                needs_fallback.push(line.clone());
            }
            Err(ambiguous) => {
                return Err(format!(
                    "Legacy v1 merge entry at rank {rank} `{ambiguous}` is ambiguous \
                     against the vocabulary; more than one candidate split satisfies \
                     the vocab triple `(a, b, a ++ b)`. Re-serialise the tokenizer \
                     as v2 array merges to disambiguate."
                ));
            }
        }
    }

    let fallback = convert_merges_to_hashmap(needs_fallback.into_iter(), vocab)
        .map_err(|e| e.to_string())?;
    let mut fallback_iter = fallback.into_iter();
    for r in resolved_flags {
        match r {
            Some(pair) => out.push(pair),
            None => {
                if let Some(pair) = fallback_iter.next() {
                    out.push(pair);
                } else {
                    return Err("internal error: fallback exhausted".to_string());
                }
            }
        }
    }
    Ok(out)
}

impl Serialize for BPE {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("BPE", 8)?;

        // Start by small fields
        model.serialize_field("type", "BPE")?;
        model.serialize_field("dropout", &self.dropout)?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("end_of_word_suffix", &self.end_of_word_suffix)?;
        model.serialize_field("fuse_unk", &self.fuse_unk)?;
        model.serialize_field("byte_fallback", &self.byte_fallback)?;
        model.serialize_field("ignore_merges", &self.ignore_merges)?;

        // Then the large ones
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        let merges = merges
            .into_iter()
            .map(|(pair, _)| (self.vocab_r[&pair.0].clone(), self.vocab_r[&pair.1].clone()))
            .collect::<Vec<_>>();
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);

        model.serialize_field("vocab", &ordered_vocab)?;
        model.serialize_field("merges", &merges)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for BPE {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "BPE",
            &[
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
                "ignore_merges",
                "vocab",
                "merges",
            ],
            BPEVisitor,
        )
    }
}

struct BPEVisitor;
impl<'de> Visitor<'de> for BPEVisitor {
    type Value = BPE;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct BPE")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = BpeBuilder::new();
        let mut vocab: Option<AHashMap<String, u32>> = None;

        #[derive(Debug, Deserialize)]
        #[serde(untagged)]
        enum MergeType {
            Tuple(Vec<(String, String)>),
            Legacy(Vec<String>),
        }
        let mut merges: Option<MergeType> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "dropout" => {
                    if let Some(dropout) = map.next_value()? {
                        builder = builder.dropout(dropout);
                    }
                }
                "unk_token" => {
                    if let Some(unk) = map.next_value()? {
                        builder = builder.unk_token(unk);
                    }
                }
                "continuing_subword_prefix" => {
                    if let Some(prefix) = map.next_value()? {
                        builder = builder.continuing_subword_prefix(prefix);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.end_of_word_suffix(suffix);
                    }
                }
                "fuse_unk" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.fuse_unk(suffix);
                    }
                }
                "byte_fallback" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.byte_fallback(suffix);
                    }
                }
                "ignore_merges" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.ignore_merges(suffix);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                "merges" => merges = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "BPE" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"BPE",
                        ))
                    }
                },
                _ => {}
            }
        }
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            let merges = match merges {
                MergeType::Tuple(merges) => merges,
                MergeType::Legacy(merges) => {
                    legacy_merges_to_tuples(merges, &vocab).map_err(Error::custom)?
                }
            };
            builder = builder.vocab_and_merges(vocab, merges);
            Ok(builder.build().map_err(Error::custom)?)
        } else {
            Err(Error::custom("Missing vocab/merges"))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::bpe::Vocab;

    #[test]
    fn test_serialization() {
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let legacy = serde_json::from_str(legacy).unwrap();
        assert_eq!(bpe, legacy);

        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);

        // With a space in the token
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b c d".into(), 2),
            ("ab c d".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b c d".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();
        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);
    }

    #[test]
    fn test_legacy_v1_merges_with_space_in_token_round_trip() {
        // The v2 array form `[["a","b c d"]]` and the v1 legacy form
        // `["a b c d"]` describe the same merge once disambiguated against
        // the vocabulary. Previously the v1 form failed with
        // `Merges text file invalid at line 1` because the parser used
        // a strict `split(' ')` with `parts.len() != 2`; the v2 form
        // worked. After the fix the two forms round-trip to the same BPE.
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b c d".into(), 2),
            ("ab c d".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let v2 = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]}"#;
        let v1 = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":["a b c d"]}"#;

        let bpe_v2: BPE = serde_json::from_str(v2).unwrap();
        let bpe_v1: BPE = serde_json::from_str(v1).unwrap();
        assert_eq!(bpe_v1, bpe_v2);

        // Sanity: the resolved merge is `(a, "b c d")`, mapping to the
        // vocab id of `"ab c d"`.
        assert_eq!(bpe_v1.merges.get(&(1, 2)).copied(), Some((0u32, 3u32)));
        let _ = vocab;
    }

    #[test]
    fn test_legacy_v1_merges_single_space_fast_path_unchanged() {
        // The single-space fast path must be byte-for-byte identical to
        // the previous behaviour for tokenizers whose vocabulary holds
        // no space-containing tokens.
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let bpe: BPE = serde_json::from_str(legacy).unwrap();
        assert_eq!(bpe.merges.get(&(1, 2)).copied(), Some((0u32, 3u32)));
        let _ = vocab;
    }

    #[test]
    fn test_legacy_v1_merges_unresolvable_multi_space_errors_clearly() {
        // A multi-space legacy line where no candidate split produces a
        // `(a, b, a ++ b)` triple in the vocabulary must surface the new,
        // explicit error rather than the prior cryptic
        // `Merges text file invalid at line 1`.
        let unresolvable = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2},"merges":["a b c d"]}"#;
        let err = serde_json::from_str::<BPE>(unresolvable).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("could not be resolved against the vocabulary"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn test_legacy_v1_merges_ambiguous_multi_space_errors_clearly() {
        // Two different splits both produce a valid `(a, b, a ++ b)`
        // triple. The fix must refuse rather than silently picking one.
        // Vocab carries both `a b` + `c` + `a bc` and `a` + `b c` + `ab c`.
        // For the line `a b c`, the candidate splits are:
        //   (a, "b c")    : a in vocab, "b c" in vocab, "ab c" in vocab -> OK
        //   ("a b", c)    : "a b" in vocab, c in vocab, "a bc" in vocab -> OK
        // The fix must surface the ambiguity rather than picking one.
        let ambiguous = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c":2,"ab c":3,"a b":4,"c":5,"a bc":6},"merges":["a b c"]}"#;
        let err = serde_json::from_str::<BPE>(ambiguous).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ambiguous against the vocabulary"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn test_serialization_ignore_merges() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);

        bpe.ignore_merges = false;
        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);
    }
}
