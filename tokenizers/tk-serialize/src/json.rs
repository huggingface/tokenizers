//! We decided to use `hifijson` as it has 0 dependencies and is very lightweight. Most of the
//! code in this file is just to reproduce the previous behavior when parsing a tokenizer.json and
//! fp64 values. As such we just vendored a small part of the serde json library with
//! attritbution, in [`crate::vendored`].
//!

use std::borrow::Cow;
use std::fmt;

use hifijson::token::Lex as _;

use crate::vendored::{f64_from_parts, number};

/// Deepest nesting accepted. A real `tokenizer.json` nests about six levels max
/// (`post_processor.special_tokens.<tok>.tokens[0]`); this leaves plenty of room while keeping a
/// hostile file from recursing the parser off the stack.
pub const MAX_DEPTH: usize = 64;

/// A parsed `tokenizer.json`.
///
/// `hifijson`'s tree instantiated for slice input, which is why the two type parameters are what
/// they are: a string borrows from the input unless it carries an escape (`Cow`), and a number is
/// the `&str` it was written as, paired with the parts (`zero`/`dot`/`exp`) the lexer saw. Nothing
/// becomes an `f64` until [`JsonExt::as_f64`] asks, which is the whole reason this parser and not
/// another — see [`f64_from_parts`].
pub type Json<'a> = hifijson::value::Value<&'a str, Cow<'a, str>>;

/// A parse failure, and how far into the document it happened.
#[derive(Debug, PartialEq)]
pub struct Error {
    /// Byte offset of the first byte the lexer had not consumed when it gave up. Worth keeping: a
    /// `tokenizer.json` runs to tens of megabytes, so "invalid JSON" on its own says very little.
    pub at: usize,
    /// What `hifijson` refused.
    pub kind: hifijson::Error,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.kind)
    }
}

impl std::error::Error for Error {}

/// The lookups the reader needs, as a trait because [`Json`] is `hifijson`'s type, not ours.
///
/// Every accessor answers `None` for "not there or not that kind", which is what every caller
/// wants: an absent field and a wrongly-typed one both mean "not available", and the reader turns
/// that into its own message naming the field.
pub trait JsonExt<'a>: Sized {
    /// Parse a whole document. Trailing content other than whitespace is an error.
    fn parse(input: &'a str) -> Result<Self, Error>;

    /// Look a key up in an object. `None` for a missing key *or* a non-object.
    fn get(&self, key: &str) -> Option<&Json<'a>>;

    /// [`get`](JsonExt::get), but `null` reads as absent. `tokenizer.json` spells "no normalizer"
    /// as `"normalizer": null`, and every caller wants that to behave like a missing key.
    fn get_some(&self, key: &str) -> Option<&Json<'a>>;

    /// The `"type"` tag, when there is one. Absent for the legacy untagged configs.
    fn type_tag(&self) -> Option<&str>;

    fn as_str(&self) -> Option<&str>;
    fn as_bool(&self) -> Option<bool>;

    /// The number as `serde_json`'s default build would have parsed it. See [`f64_from_parts`].
    fn as_f64(&self) -> Option<f64>;

    /// A token id or a count. Rejects fractions, negatives and anything past `u32::MAX`, so a
    /// malformed file cannot silently truncate an id.
    fn as_u32(&self) -> Option<u32>;

    fn as_usize(&self) -> Option<usize>;
    fn as_arr(&self) -> Option<&[Json<'a>]>;
    fn as_obj(&self) -> Option<&[(Cow<'a, str>, Json<'a>)]>;
}

impl<'a> JsonExt<'a> for Json<'a> {
    fn parse(input: &'a str) -> Result<Self, Error> {
        let mut lexer = hifijson::SliceLexer::new(input.as_bytes());
        // `exactly_one` is what rejects trailing content; `parse_bounded` is what caps recursion.
        let parsed = lexer.exactly_one(hifijson::token::Lex::ws_peek, |next, lexer| {
            hifijson::value::parse_bounded(MAX_DEPTH, next, lexer)
        });
        parsed.map_err(|kind| Error {
            // What the lexer has not eaten yet, measured against the whole input.
            at: input.len() - lexer.as_slice().len(),
            kind,
        })
    }

    fn get(&self, key: &str) -> Option<&Json<'a>> {
        match self {
            // Linear, because an object here has a handful of keys and hashing them costs more.
            Json::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn get_some(&self, key: &str) -> Option<&Json<'a>> {
        self.get(key).filter(|v| !matches!(v, Json::Null))
    }

    fn type_tag(&self) -> Option<&str> {
        self.get("type")?.as_str()
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(&**s),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number((digits, _)) => {
                let (positive, significand, exponent) = number(digits);
                Some(f64_from_parts(positive, significand, exponent))
            }
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&n) {
            return None;
        }
        Some(n as u32)
    }

    fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=9_007_199_254_740_992.0).contains(&n) {
            return None;
        }
        Some(n as usize)
    }

    fn as_arr(&self) -> Option<&[Json<'a>]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_obj(&self) -> Option<&[(Cow<'a, str>, Json<'a>)]> {
        match self {
            Json::Object(entries) => Some(entries),
            _ => None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Json<'_> {
        Json::parse(s).expect("should parse")
    }


    #[test]
    fn scalars_and_containers() {
        assert_eq!(p("null"), Json::Null);
        assert_eq!(p("true"), Json::Bool(true));
        assert_eq!(p(" false "), Json::Bool(false));
        assert_eq!(p("0").as_f64(), Some(0.0));
        assert_eq!(p("-12").as_f64(), Some(-12.0));
        assert_eq!(p("1.5e2").as_f64(), Some(150.0));
        assert_eq!(p("[]").as_arr().unwrap().len(), 0);
        assert_eq!(p("{}").as_obj().unwrap().len(), 0);
        assert_eq!(p(r#"[1,2,3]"#).as_arr().unwrap().len(), 3);
    }

    #[test]
    fn nested_lookup() {
        let v = p(r#"{"a": {"b": [10, 20]}, "c": null}"#);
        assert_eq!(
            v.get("a").unwrap().get("b").unwrap().as_arr().unwrap()[1].as_u32(),
            Some(20)
        );
        // `null` is present via `get` but absent via `get_some`.
        assert!(v.get("c").is_some());
        assert!(v.get_some("c").is_none());
        assert!(v.get("nope").is_none());
        // A non-object parent reads as absent rather than panicking.
        assert!(p("42").get("a").is_none());
    }

    #[test]
    fn strings_borrow_when_they_can() {
        let v = p(r#""plain""#);
        assert!(matches!(v, Json::String(Cow::Borrowed("plain"))));
        let v = p(r#""with\nescape""#);
        assert!(matches!(v, Json::String(Cow::Owned(_))));
        assert_eq!(v.as_str(), Some("with\nescape"));
    }

    #[test]
    fn all_the_escapes() {
        let v = p(r#""q\"b\\s\/n\nr\rt\tb\bf\f""#);
        assert_eq!(v.as_str(), Some("q\"b\\s/n\nr\rt\tb\u{8}f\u{c}"));
    }

    #[test]
    fn unicode_escapes_including_surrogate_pairs() {
        assert_eq!(p(r#""A""#).as_str(), Some("A"));
        // Ġ (U+0120) — the byte-level marker that fills GPT-2 vocabularies.
        assert_eq!(p(r#""Ġ""#).as_str(), Some("\u{120}"));
        // A pair: U+1F600 GRINNING FACE.
        assert_eq!(p(r#""😀""#).as_str(), Some("😀"));
        // Mixed with literal text on both sides.
        assert_eq!(p(r#""aĠb""#).as_str(), Some("a\u{120}b"));
    }

    #[test]
    fn integer_accessors_are_checked() {
        assert_eq!(p("7").as_u32(), Some(7));
        assert_eq!(p("7.5").as_u32(), None, "fractions are not ids");
        assert_eq!(p("-1").as_u32(), None, "negatives are not ids");
        assert_eq!(p("4294967296").as_u32(), None, "past u32::MAX");
        assert_eq!(p("4294967295").as_u32(), Some(u32::MAX));
        assert_eq!(p("-0.5").as_usize(), None);
    }

    #[test]
    fn malformed_input_is_rejected() {
        for bad in [
            "",
            "{",
            "[",
            "{\"a\"}",
            "{\"a\":}",
            "{\"a\":1,}",
            "[1,]",
            "[1 2]",
            "\"unterminated",
            r#""\q""#,
            r#""\u00""#,
            r#""\ud83d""#,  // lone high surrogate
            r#""\ud83dQ""#, // high surrogate, no \u after
            r#""\udc00""#,  // lone low surrogate
            "tru",
            "1 2",
            "{} {}",
            "nul",
            // These three the hand-written parser used to accept, which made it looser than
            // serde_json. `hifijson` rejects all of them.
            "007",
            "01",
            "\"a\u{1}b\"", // unescaped control character
        ] {
            assert!(
                Json::parse(bad).is_err(),
                "should have been rejected: {bad:?}"
            );
        }
    }

    /// The cap is a stack guard, so the boundary itself is the interesting part: `MAX_DEPTH` nested
    /// values parse and one more does not. Pinned exactly because it is `parse_bounded`'s
    /// `depth` argument that has to keep reproducing it.
    #[test]
    fn depth_is_capped() {
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(Json::parse(&ok).is_ok(), "{MAX_DEPTH} levels must parse");
        let too_deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        let err = Json::parse(&too_deep).expect_err("should refuse");
        assert_eq!(err.kind, hifijson::Error::Depth);
    }

    /// Structural equality against `serde_json` over every real `tokenizer.json` in `data/`.
    ///
    /// The unit tests above cover the shapes I thought of; this covers the ones the Hub actually
    /// ships. Skipped when the fixtures are not fetched (`make models`).
    #[test]
    fn agrees_with_serde_json_on_every_real_config() {
        /// `Ok(())` or the JSON path of the first difference, so a failure says *where*.
        fn same(a: &Json<'_>, b: &serde_json::Value, at: &str) -> Result<(), String> {
            let bad = |why: &str| Err(format!("{at}: {why}"));
            match (a, b) {
                (Json::Null, serde_json::Value::Null) => Ok(()),
                (Json::Bool(x), serde_json::Value::Bool(y)) if x == y => Ok(()),
                // Bit-exact, deliberately: ids must not move, so this reader reproduces
                // `serde_json`'s float rounding rather than improving on it -- see
                // `f64_from_parts`. Comparing bits rather than `==` also catches a -0.0 vs 0.0
                // divergence, which `==` would call equal.
                (Json::Number(_), serde_json::Value::Number(y)) => {
                    let x = a.as_f64().expect("a number reads as an f64");
                    match y.as_f64() {
                        Some(y) if x.to_bits() == y.to_bits() => Ok(()),
                        Some(y) => bad(&format!(
                            "number {x} ({:016x}) vs {y} ({:016x})",
                            x.to_bits(),
                            y.to_bits()
                        )),
                        None => bad("serde_json number is not an f64"),
                    }
                }
                (Json::String(x), serde_json::Value::String(y)) if x.as_ref() == y.as_str() => {
                    Ok(())
                }
                (Json::String(x), serde_json::Value::String(y)) => {
                    bad(&format!("string {:?} vs {:?}", x.as_ref(), y.as_str()))
                }
                (Json::Array(x), serde_json::Value::Array(y)) => {
                    if x.len() != y.len() {
                        return bad(&format!("array len {} vs {}", x.len(), y.len()));
                    }
                    for (i, (a, b)) in x.iter().zip(y).enumerate() {
                        same(a, b, &format!("{at}[{i}]"))?;
                    }
                    Ok(())
                }
                (Json::Object(x), serde_json::Value::Object(y)) => {
                    if x.len() != y.len() {
                        return bad(&format!("object len {} vs {}", x.len(), y.len()));
                    }
                    for (k, v) in x {
                        match y.get(k.as_ref()) {
                            Some(w) => same(v, w, &format!("{at}.{k}"))?,
                            None => return bad(&format!("key {k:?} missing on the serde side")),
                        }
                    }
                    Ok(())
                }
                _ => bad("different kinds"),
            }
        }

        let dir = std::path::Path::new("../data");
        if !dir.exists() {
            return;
        }
        let mut checked = 0;
        for entry in std::fs::read_dir(dir).expect("read data/") {
            let path = entry.expect("entry").path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let mine = Json::parse(&text)
                .unwrap_or_else(|e| panic!("{}: our parser rejected it: {e}", path.display()));
            let theirs: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{}: serde_json rejected it: {e}", path.display()));
            if let Err(why) = same(&mine, &theirs, "$") {
                panic!("{}: {why}", path.display());
            }
            checked += 1;
        }
        // Deliberately >= 1, not a higher count: how many fixtures exist depends on the CI leg.
        // The Windows job fetches only gpt2.json and albert-base-v1-tokenizer.json, while a full
        // `make models` checkout has 22. The guard's job is to stop this passing vacuously when the
        // directory exists but is empty, not to police how many files are in it.
        assert!(
            checked >= 1,
            "../data exists but held no readable *.json, so this asserted nothing"
        );
        eprintln!("agreed with serde_json on {checked} real config(s)");
    }

    #[test]
    fn reads_a_tokenizer_json_shaped_document() {
        let doc = r#"{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": null,
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                            "use_regex": true},
          "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false,
                             "use_regex": true},
          "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                      "use_regex": true},
          "model": {"type": "BPE", "dropout": null, "unk_token": null, "vocab": {"a": 0, "b": 1},
                    "merges": [["a", "b"]]}
        }"#;
        let v = p(doc);
        assert_eq!(v.get("version").unwrap().as_str(), Some("1.0"));
        assert!(v.get_some("truncation").is_none());
        assert_eq!(
            v.get("pre_tokenizer").unwrap().type_tag(),
            Some("ByteLevel")
        );
        let model = v.get("model").unwrap();
        assert_eq!(model.type_tag(), Some("BPE"));
        assert_eq!(
            model.get("vocab").unwrap().get("b").unwrap().as_u32(),
            Some(1)
        );
        let merge = &model.get("merges").unwrap().as_arr().unwrap()[0];
        assert_eq!(merge.as_arr().unwrap()[0].as_str(), Some("a"));
        let added = &v.get("added_tokens").unwrap().as_arr().unwrap()[0];
        assert_eq!(added.get("id").unwrap().as_u32(), Some(0));
        assert_eq!(added.get("special").unwrap().as_bool(), Some(true));
        // A `dropout: null` field reads as absent, which is how the BC default works.
        assert!(model.get_some("dropout").is_none());
    }
}
