//! Reading a `tokenizer.json` without linking `serde_json`.
//!
//! The parsing itself is [`hifijson`]'s: a JSON lexer with no required dependencies, whose tree is
//! already the shape this crate wants — insertion-ordered objects, strings that borrow from the
//! input, and numbers left as the digits they were written with. What lives here is the thin layer
//! on top: the accessors `from_json` walks the tree with, and the float reassembly that turns those
//! digits into the exact `f64` `serde_json` would have produced.
//!
//! Numbers are why this file is not simply a `pub use`. A Unigram score decides a Viterbi near-tie,
//! so the arithmetic has to match `serde_json`'s default build bit for bit — being *more* accurate
//! would move ids that ship today. `hifijson` is the parser that lets us do that, because it never
//! parses a float itself: [`number`] and [`f64_from_parts`] below do, reproducing serde's rounding
//! error on purpose.
//!
//! A `tokenizer.json` is usually fetched from the Hub, so this is a trust boundary: nesting is
//! capped ([`MAX_DEPTH`]) because the parser recurses, and every integer accessor is checked.

use std::borrow::Cow;
use std::fmt;

use hifijson::token::Lex as _;

/// Deepest nesting accepted. A real `tokenizer.json` nests about six levels
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

// ---------------------------------------------------------------------------------------------
// The float reassembly below (`POW10` and `f64_from_parts`) is ported from `serde_json` 1.0,
// `src/de.rs` — its `#[cfg(not(feature = "float_roundtrip"))]` path, which is the default build.
//
//     Copyright (c) Erick Tryzelaar and David Tolnay
//     Licensed under either of Apache License, Version 2.0 or MIT license, at your option.
//     https://github.com/serde-rs/json
//
// Ported rather than depended upon because this crate's whole purpose is to read a
// `tokenizer.json` without linking serde — but the *arithmetic* has to match serde's exactly, bug
// for bug, or token ids move. See `f64_from_parts` for why.
//
// One divergence from serde_json is left, and no real `tokenizer.json` exercises it: an
// overflowing exponent (`1e400`) yields infinity here where serde reports a range error. The two
// others this comment used to list -- a leading zero (`007`) and an unescaped control character in
// a string, both of which the hand-written scanner accepted -- are gone, because `hifijson` rejects
// them. So the float *values* are bit-identical, which is what ids depend on, and the accept/reject
// boundary now differs in exactly one place, pinned by
// `an_overflowing_exponent_is_the_last_divergence_from_serde`.
// ---------------------------------------------------------------------------------------------

/// `10^n` for `n` in `0..=308`, as `serde_json` spells it. Rust's float-literal parsing is
/// correctly rounded, so these are bit-identical to its table.
#[rustfmt::skip]
static POW10: [f64; 309] = [
    1e000, 1e001, 1e002, 1e003, 1e004, 1e005, 1e006, 1e007,
    1e008, 1e009, 1e010, 1e011, 1e012, 1e013, 1e014, 1e015,
    1e016, 1e017, 1e018, 1e019, 1e020, 1e021, 1e022, 1e023,
    1e024, 1e025, 1e026, 1e027, 1e028, 1e029, 1e030, 1e031,
    1e032, 1e033, 1e034, 1e035, 1e036, 1e037, 1e038, 1e039,
    1e040, 1e041, 1e042, 1e043, 1e044, 1e045, 1e046, 1e047,
    1e048, 1e049, 1e050, 1e051, 1e052, 1e053, 1e054, 1e055,
    1e056, 1e057, 1e058, 1e059, 1e060, 1e061, 1e062, 1e063,
    1e064, 1e065, 1e066, 1e067, 1e068, 1e069, 1e070, 1e071,
    1e072, 1e073, 1e074, 1e075, 1e076, 1e077, 1e078, 1e079,
    1e080, 1e081, 1e082, 1e083, 1e084, 1e085, 1e086, 1e087,
    1e088, 1e089, 1e090, 1e091, 1e092, 1e093, 1e094, 1e095,
    1e096, 1e097, 1e098, 1e099, 1e100, 1e101, 1e102, 1e103,
    1e104, 1e105, 1e106, 1e107, 1e108, 1e109, 1e110, 1e111,
    1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
    1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127,
    1e128, 1e129, 1e130, 1e131, 1e132, 1e133, 1e134, 1e135,
    1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142, 1e143,
    1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 1e150, 1e151,
    1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
    1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167,
    1e168, 1e169, 1e170, 1e171, 1e172, 1e173, 1e174, 1e175,
    1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183,
    1e184, 1e185, 1e186, 1e187, 1e188, 1e189, 1e190, 1e191,
    1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
    1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207,
    1e208, 1e209, 1e210, 1e211, 1e212, 1e213, 1e214, 1e215,
    1e216, 1e217, 1e218, 1e219, 1e220, 1e221, 1e222, 1e223,
    1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 1e230, 1e231,
    1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
    1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247,
    1e248, 1e249, 1e250, 1e251, 1e252, 1e253, 1e254, 1e255,
    1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
    1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271,
    1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
    1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287,
    1e288, 1e289, 1e290, 1e291, 1e292, 1e293, 1e294, 1e295,
    1e296, 1e297, 1e298, 1e299, 1e300, 1e301, 1e302, 1e303,
    1e304, 1e305, 1e306, 1e307, 1e308,
];

/// Reassemble a float the way `serde_json` does, **including its rounding error**.
///
/// This is deliberate bug-compatibility, not an oversight. `serde_json` (without its
/// `float_roundtrip` feature, which is off by default and which we do not enable) accumulates the
/// significant digits into a `u64` and then applies a single multiply or divide by a power of ten.
/// When the significand exceeds 2^53 the `as f64` conversion rounds, and the divide rounds again —
/// two roundings, so the result can sit 1 ULP away from the correctly-rounded value that
/// `f64::from_str` produces.
///
/// That matters because Unigram scores feed a Viterbi lattice: on t5-base, 8,334 of 32,100 scores
/// differ by 1 ULP between the two algorithms, which flips a near-tie about twice per 1.25M tokens.
/// Being *more* accurate here would silently change the ids that ship today, so the slim reader
/// reproduces the config path bit-for-bit instead. `numbers_are_bit_identical_to_serde_json` pins it.
/// The value this reader gives a number literal, without building a [`Json`] around it.
///
/// [`JsonExt::as_f64`] is this function plus a match on the variant, and a writer needs exactly
/// this half: to check that a literal it is about to emit reads back as the `f64` it started from,
/// it has to go through the same arithmetic, not through `f64::from_str`. Sharing the function is
/// what makes that a tautology rather than a second implementation to keep in step.
#[cfg(any(feature = "serialize", test))]
pub(crate) fn f64_from_literal(digits: &str) -> f64 {
    let (positive, significand, exponent) = number(digits);
    f64_from_parts(positive, significand, exponent)
}

fn f64_from_parts(positive: bool, significand: u64, mut exponent: i32) -> f64 {
    let mut f = significand as f64;
    loop {
        match POW10.get(exponent.unsigned_abs() as usize) {
            Some(&pow10) => {
                if exponent >= 0 {
                    f *= pow10;
                    if f.is_infinite() {
                        return if positive {
                            f64::INFINITY
                        } else {
                            f64::NEG_INFINITY
                        };
                    }
                } else {
                    f /= pow10;
                }
                break;
            }
            None => {
                // The exponent is past the table; step it down in 308-sized chunks, as serde does.
                if f == 0.0 {
                    break;
                }
                if exponent >= 0 {
                    return if positive {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                }
                f /= 1e308;
                exponent += 308;
            }
        }
    }
    if positive { f } else { -f }
}

/// Split the digits `hifijson` handed back into the three parts [`f64_from_parts`] wants, exactly
/// as `serde_json` accumulates them.
///
/// The input is a number `hifijson` has already validated, so it matches
/// `-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?` and there is nothing left to reject — which is why this
/// returns the parts rather than a `Result`. What it must get right is *where digits are dropped*:
/// once the integer part would overflow a `u64` the remaining integer digits are dropped and the
/// exponent bumped, while the first fraction digit that would overflow stops accumulation for good
/// and every digit after it is ignored, because they sit to the right of the point. Both are
/// serde's behaviour, and both change the resulting float, so
/// `numbers_are_bit_identical_to_serde_json` pins them.
fn number(digits: &str) -> (bool, u64, i32) {
    let s = digits.as_bytes();
    let mut i = 0;
    let positive = if s.first() == Some(&b'-') {
        i = 1;
        false
    } else {
        true
    };

    // Integer part (serde's `parse_long_integer`).
    let mut significand: u64 = 0;
    let mut exponent: i32 = 0;
    let mut saturated = false;
    while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
        let digit = u64::from(c - b'0');
        if saturated {
            exponent += 1;
        } else {
            match significand
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit))
            {
                Some(v) => significand = v,
                None => {
                    saturated = true;
                    exponent += 1;
                }
            }
        }
        i += 1;
    }

    // Fraction (serde's `parse_decimal`). The first digit that would overflow ends accumulation
    // for good -- serde hands off to `parse_decimal_overflow`, which ignores every remaining digit
    // rather than retrying them. Retrying is observably different, not a rounding difference:
    // `significand * 10` can still fit a *smaller* later digit, and `zz`-style sweeps against
    // serde put the disagreement in the 16th significant digit.
    if s.get(i) == Some(&b'.') {
        i += 1;
        while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
            let digit = u64::from(c - b'0');
            if !saturated {
                match significand
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(digit))
                {
                    Some(v) => {
                        significand = v;
                        exponent -= 1;
                    }
                    None => saturated = true,
                }
            }
            i += 1;
        }
    }

    // Explicit exponent.
    if matches!(s.get(i), Some(b'e' | b'E')) {
        i += 1;
        let negate = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let mut exp: i32 = 0;
        while let Some(c) = s.get(i).filter(|c| c.is_ascii_digit()) {
            exp = exp.saturating_mul(10).saturating_add(i32::from(c - b'0'));
            i += 1;
        }
        exponent = exponent.saturating_add(if negate { -exp } else { exp });
    }

    (positive, significand, exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Json<'_> {
        Json::parse(s).expect("should parse")
    }

    /// The score that forced this design. `albert-base-v1` and t5 both ship
    /// `-3.8403830528259277`. `f64::from_str` is correctly rounded and lands on
    /// `c00eb91ac0000000`; `serde_json`'s default path lands on `...0001`.
    ///
    /// We match **serde**, not `from_str`. Being more accurate would move ids that ship today --
    /// on t5-base it flips a Viterbi near-tie roughly twice per 1.25M tokens -- so
    /// bug-compatibility is the requirement, and this test says so out loud.
    #[test]
    // Compares against `serde_json` itself, which is a dev-dependency here.
    fn matches_serde_not_from_str_on_a_real_unigram_score() {
        let lit = "-3.8403830528259277";
        let ours = Json::parse(lit).unwrap().as_f64().unwrap();
        let serde: f64 = serde_json::from_str(lit).unwrap();
        let correctly_rounded: f64 = lit.parse().unwrap();

        assert_eq!(
            ours.to_bits(),
            serde.to_bits(),
            "we must reproduce serde_json bit-for-bit"
        );
        assert_ne!(
            serde.to_bits(),
            correctly_rounded.to_bits(),
            "if serde ever becomes correctly rounded by default, this emulation should be deleted"
        );
    }

    /// Every number shape a `tokenizer.json` contains, plus the awkward ones that exercise the
    /// significand-overflow paths.
    #[test]
    // Compares against `serde_json` itself, which is a dev-dependency here.
    fn numbers_are_bit_identical_to_serde_json() {
        for lit in [
            "0",
            "-0",
            "1",
            "-1",
            "100",
            "50256",
            "4294967295",
            "100277",
            "0.0",
            "-0.0",
            "1.5",
            "0.1",
            "0.3",
            "1e10",
            "1e-10",
            "1E+2",
            "-2.5e-3",
            // Real Unigram scores -- the case that actually matters.
            "-3.8403830528259277",
            "-3.1902313232421875",
            "-10.522398948669434",
            "-0.0001",
            "-1e-300",
            "1e308",
            "1e-320",
            // More significant digits than a u64 holds, so drop-and-bump runs on both sides.
            "123456789012345678901234567890",
            "1.2345678901234567890123456789",
            "0.000000000000000000000000001",
            "18446744073709551615",
            "18446744073709551616",
        ] {
            let ours = Json::parse(lit).unwrap().as_f64().unwrap();
            let serde: f64 = serde_json::from_str(lit).unwrap();
            assert_eq!(
                ours.to_bits(),
                serde.to_bits(),
                "{lit}: ours={ours} ({:016x}) serde={serde} ({:016x})",
                ours.to_bits(),
                serde.to_bits()
            );
        }

        // Where the fraction digits overflow a u64 mid-number. Swept rather than sampled because
        // the bug this catches needed a *specific* pair of digits -- one that overflows followed by
        // a smaller one that would still fit -- and 336 of these 4000 disagreed with serde before
        // `number` learned to stop accumulating for good. 1844674407370955161 is the largest
        // 19-digit significand that fits, so the 20th digit is where it tips.
        for prefix in [
            "1844674407370955161",
            "1844674407370955160",
            "9223372036854775807",
            "1234567890123456789",
        ] {
            for suffix in 0..1000u32 {
                let lit = format!("0.{prefix}{suffix:03}");
                let ours = Json::parse(&lit).unwrap().as_f64().unwrap();
                let serde: f64 = serde_json::from_str(&lit).unwrap();
                assert_eq!(
                    ours.to_bits(),
                    serde.to_bits(),
                    "{lit}: ours={ours:?} serde={serde:?}"
                );
            }
        }
    }

    /// The one place this reader is still looser than `serde_json`, kept here so it is a documented
    /// fact rather than a surprise: an exponent past the `POW10` table saturates to infinity, where
    /// serde reports "number out of range". No `tokenizer.json` contains such a literal -- a
    /// Unigram score is a log-probability -- and closing it would mean reintroducing a range check
    /// that the float emulation does not otherwise need.
    #[test]
    fn an_overflowing_exponent_is_the_last_divergence_from_serde() {
        assert_eq!(p("1e400").as_f64(), Some(f64::INFINITY));
        assert_eq!(p("-1e400").as_f64(), Some(f64::NEG_INFINITY));
        assert!(serde_json::from_str::<f64>("1e400").is_err());
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
