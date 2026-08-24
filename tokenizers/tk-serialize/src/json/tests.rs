//! The parser's tests, including the one that checks it against `serde_json` on every real
//! config: the numbers have to agree bit for bit.

use super::*;

fn p(s: &str) -> Json<'_> {
    Json::parse(s).expect("should parse")
}

#[test]
fn scalars_and_containers() {
    assert_eq!(p("null").0, Raw::Null);
    assert_eq!(p("true").0, Raw::Bool(true));
    assert_eq!(p(" false ").0, Raw::Bool(false));
    assert_eq!(p("0").as_f64(), Some(0.0));
    assert_eq!(p("-12").as_f64(), Some(-12.0));
    assert_eq!(p("1.5e2").as_f64(), Some(150.0));
    assert_eq!(p("[]").as_array().unwrap().len(), 0);
    assert_eq!(p("{}").entries().unwrap().len(), 0);
    assert_eq!(p(r#"[1,2,3]"#).as_array().unwrap().len(), 3);
}

#[test]
fn nested_lookup() {
    let v = p(r#"{"a": {"b": [10, 20]}, "c": null}"#);
    assert_eq!(
        v.get("a").unwrap().get("b").unwrap().as_array().unwrap()[1].as_u32(),
        Some(20)
    );
    // `null` is present via `get` but absent via `field`.
    assert!(v.get("c").is_some());
    assert!(v.field("c").is_none());
    assert!(v.get("nope").is_none());
    // A non-object parent reads as absent rather than panicking.
    assert!(p("42").get("a").is_none());
}

#[test]
fn strings_borrow_when_they_can() {
    let v = p(r#""plain""#);
    assert!(matches!(v.0, Raw::String(Cow::Borrowed("plain"))));
    let v = p(r#""with\nescape""#);
    assert!(matches!(v.0, Raw::String(Cow::Owned(_))));
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
        match (&a.0, b) {
            (Raw::Null, serde_json::Value::Null) => Ok(()),
            (Raw::Bool(x), serde_json::Value::Bool(y)) if x == y => Ok(()),
            // Bit-exact, deliberately: ids must not move, so this reader reproduces
            // `serde_json`'s float rounding rather than improving on it -- see
            // `f64_from_parts`. Comparing bits rather than `==` also catches a -0.0 vs 0.0
            // divergence, which `==` would call equal.
            (Raw::Number(_), serde_json::Value::Number(y)) => {
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
            (Raw::String(x), serde_json::Value::String(y)) if x.as_ref() == y.as_str() => {
                Ok(())
            }
            (Raw::String(x), serde_json::Value::String(y)) => {
                bad(&format!("string {:?} vs {:?}", x.as_ref(), y.as_str()))
            }
            (Raw::Array(x), serde_json::Value::Array(y)) => {
                if x.len() != y.len() {
                    return bad(&format!("array len {} vs {}", x.len(), y.len()));
                }
                for (i, (a, b)) in x.iter().zip(y).enumerate() {
                    same(Json::wrap(a), b, &format!("{at}[{i}]"))?;
                }
                Ok(())
            }
            (Raw::Object(x), serde_json::Value::Object(y)) => {
                if x.len() != y.len() {
                    return bad(&format!("object len {} vs {}", x.len(), y.len()));
                }
                for (k, v) in x {
                    match y.get(k.as_ref()) {
                        Some(w) => same(Json::wrap(v), w, &format!("{at}.{k}"))?,
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
      "version": "2.0",
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
    assert_eq!(v.get("version").unwrap().as_str(), Some("2.0"));
    assert!(v.field("truncation").is_none());
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
    let merge = &model.get("merges").unwrap().as_array().unwrap()[0];
    assert_eq!(merge.as_array().unwrap()[0].as_str(), Some("a"));
    let added = &v.get("added_tokens").unwrap().as_array().unwrap()[0];
    assert_eq!(added.get("id").unwrap().as_u32(), Some(0));
    assert_eq!(added.get("special").unwrap().as_bool(), Some(true));
    // A `dropout: null` field reads as absent, which is how the BC default works.
    assert!(model.field("dropout").is_none());
}
