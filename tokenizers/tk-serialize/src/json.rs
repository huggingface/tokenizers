//! A minimal JSON reader, so the default build can load a `tokenizer.json` without linking
//! `serde_json`.
//!
//! This is not a general-purpose JSON library and should not grow into one. It parses the subset a
//! `tokenizer.json` actually uses, borrows every string it can, and gives up early and loudly on
//! anything else. `serde_json`'s parser plus the derive glue is most of what a `.node` binding pays
//! for the config layer; a `tokenizer.json` is a fixed shape we control, so we can read it directly.
//!
//! What it deliberately does *not* do: preserve object key order beyond insertion, deduplicate keys,
//! accept comments or trailing commas, or parse numbers it cannot represent.
//!
//! A `tokenizer.json` is usually fetched from the Hub, so this is a trust boundary: nesting is
//! capped ([`MAX_DEPTH`]) because the parser recurses, and every integer accessor is checked.

use std::borrow::Cow;
use std::fmt;

/// Deepest nesting accepted. A real `tokenizer.json` nests about six levels
/// (`post_processor.special_tokens.<tok>.tokens[0]`); this leaves plenty of room while keeping a
/// hostile file from recursing the parser off the stack.
pub const MAX_DEPTH: usize = 64;

#[derive(Debug, PartialEq)]
pub enum Json<'a> {
    Null,
    Bool(bool),
    /// Every JSON number, kept as `f64`. Unigram scores need the fraction; ids go through
    /// [`Json::as_u32`], which rejects anything non-integral or out of range.
    Num(f64),
    Str(Cow<'a, str>),
    Arr(Vec<Json<'a>>),
    /// Insertion-ordered, because duplicate keys are not our problem to resolve and a linear scan
    /// over the handful of keys an object has beats hashing them.
    Obj(Vec<(Cow<'a, str>, Json<'a>)>),
}

#[derive(Debug, PartialEq)]
pub struct Error {
    pub at: usize,
    pub msg: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.msg)
    }
}

impl std::error::Error for Error {}

type R<T> = Result<T, Error>;

impl<'a> Json<'a> {
    /// Parse a whole document. Trailing content other than whitespace is an error.
    pub fn parse(input: &'a str) -> R<Self> {
        let mut p = Parser {
            s: input.as_bytes(),
            i: 0,
            depth: 0,
        };
        p.ws();
        let value = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing content after the top-level value"));
        }
        Ok(value)
    }

    /// Look a key up in an object. `None` for a missing key *or* a non-object, which is what every
    /// caller wants: an absent field and a wrongly-typed parent both mean "not available".
    pub fn get(&self, key: &str) -> Option<&Json<'a>> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `get`, but `null` reads as absent. `tokenizer.json` spells "no normalizer" as
    /// `"normalizer": null`, and every caller wants that to behave like a missing key.
    pub fn get_some(&self, key: &str) -> Option<&Json<'a>> {
        self.get(key).filter(|v| !matches!(v, Json::Null))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// A token id or a count. Rejects fractions, negatives and anything past `u32::MAX`, so a
    /// malformed file cannot silently truncate an id.
    pub fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&n) {
            return None;
        }
        Some(n as u32)
    }

    pub fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=9_007_199_254_740_992.0).contains(&n) {
            return None;
        }
        Some(n as usize)
    }

    pub fn as_arr(&self) -> Option<&[Json<'a>]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(Cow<'a, str>, Json<'a>)]> {
        match self {
            Json::Obj(entries) => Some(entries),
            _ => None,
        }
    }

    /// The `"type"` tag, when there is one. Absent for the legacy untagged configs.
    pub fn type_tag(&self) -> Option<&str> {
        self.get("type")?.as_str()
    }
}

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

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &'static str) -> Error {
        Error { at: self.i, msg }
    }

    fn ws(&mut self) {
        while let Some(c) = self.s.get(self.i) {
            match c {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> R<Json<'a>> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nested too deeply"));
        }
        let out = match self.s.get(self.i) {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat(b"true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat(b"false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat(b"null") => Ok(Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("expected a value")),
        };
        self.depth -= 1;
        out
    }

    fn object(&mut self) -> R<Json<'a>> {
        self.i += 1; // '{'
        let mut entries = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expected a key"));
            }
            let key = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err(self.err("expected ':'"));
            }
            self.i += 1;
            self.ws();
            let value = self.value()?;
            entries.push((key, value));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(entries));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> R<Json<'a>> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    /// Borrows when there is no escape to resolve, which is the overwhelmingly common case: a
    /// 200k-entry vocab would otherwise mean 200k little allocations.
    fn string(&mut self) -> R<Cow<'a, str>> {
        self.i += 1; // opening quote
        let start = self.i;
        while let Some(c) = self.s.get(self.i) {
            match c {
                b'"' => {
                    // SAFETY-free: `self.s` came from a `&str`, and we only ever stop on ASCII
                    // delimiters, so this range is on char boundaries.
                    let raw = std::str::from_utf8(&self.s[start..self.i])
                        .map_err(|_| self.err("string is not valid UTF-8"))?;
                    self.i += 1;
                    return Ok(Cow::Borrowed(raw));
                }
                b'\\' => return self.string_escaped(start),
                _ => self.i += 1,
            }
        }
        Err(self.err("unterminated string"))
    }

    /// The slow path, taken only once an escape has actually been seen.
    fn string_escaped(&mut self, start: usize) -> R<Cow<'a, str>> {
        let mut out = String::from(
            std::str::from_utf8(&self.s[start..self.i])
                .map_err(|_| self.err("string is not valid UTF-8"))?,
        );
        while let Some(c) = self.s.get(self.i) {
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(Cow::Owned(out));
                }
                b'\\' => {
                    self.i += 1;
                    let esc = *self
                        .s
                        .get(self.i)
                        .ok_or_else(|| self.err("dangling '\\'"))?;
                    self.i += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                _ => {
                    let from = self.i;
                    while matches!(self.s.get(self.i), Some(c) if *c != b'"' && *c != b'\\') {
                        self.i += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.s[from..self.i])
                            .map_err(|_| self.err("string is not valid UTF-8"))?,
                    );
                }
            }
        }
        Err(self.err("unterminated string"))
    }

    /// `\uXXXX`, joining a surrogate pair when one follows. Byte-level vocabularies are full of
    /// these, so getting the pair case wrong would corrupt real tokenizers.
    fn unicode_escape(&mut self) -> R<char> {
        let hi = self.hex4()?;
        // Not a high surrogate: a standalone scalar value.
        if !(0xD800..0xDC00).contains(&hi) {
            return char::from_u32(hi).ok_or_else(|| self.err("escape is not a Unicode scalar"));
        }
        if !(self.s[self.i..].starts_with(b"\\u")) {
            return Err(self.err("high surrogate with no low surrogate"));
        }
        self.i += 2;
        let lo = self.hex4()?;
        if !(0xDC00..0xE000).contains(&lo) {
            return Err(self.err("expected a low surrogate"));
        }
        let combined = 0x1_0000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
        char::from_u32(combined).ok_or_else(|| self.err("surrogate pair is not a scalar"))
    }

    fn hex4(&mut self) -> R<u32> {
        let bytes = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| self.err("truncated \\u escape"))?;
        let mut v = 0u32;
        for b in bytes {
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(self.err("bad hex digit in \\u escape")),
            };
            v = v * 16 + u32::from(d);
        }
        self.i += 4;
        Ok(v)
    }

    /// Parse a number exactly as `serde_json` does — see [`f64_from_parts`] for why bit-compatibility
    /// matters more here than accuracy. Digits accumulate into a `u64` significand with a decimal
    /// exponent; overflow drops digits rather than widening, which is also what serde does.
    fn number(&mut self) -> R<Json<'a>> {
        let positive = if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
            false
        } else {
            true
        };

        // Integer part. Once the significand would overflow, further digits are dropped and the
        // exponent is bumped instead (serde's `parse_long_integer`).
        let int_start = self.i;
        let mut significand: u64 = 0;
        let mut exponent: i32 = 0;
        let mut saturated = false;
        while let Some(c) = self.s.get(self.i) {
            if !c.is_ascii_digit() {
                break;
            }
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
            self.i += 1;
        }
        if self.i == int_start {
            return Err(self.err("expected a digit"));
        }

        // Fraction. A digit that would overflow is dropped with no exponent change, because it sits
        // to the right of the point — serde stops accumulating at that moment too.
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            let frac_start = self.i;
            while let Some(c) = self.s.get(self.i) {
                if !c.is_ascii_digit() {
                    break;
                }
                let digit = u64::from(c - b'0');
                if let Some(v) = significand
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(digit))
                {
                    significand = v;
                    exponent -= 1;
                }
                self.i += 1;
            }
            if self.i == frac_start {
                return Err(self.err("expected a digit after '.'"));
            }
        }

        // Explicit exponent.
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            let negate = match self.s.get(self.i) {
                Some(b'-') => {
                    self.i += 1;
                    true
                }
                Some(b'+') => {
                    self.i += 1;
                    false
                }
                _ => false,
            };
            let exp_start = self.i;
            let mut exp: i32 = 0;
            while let Some(c) = self.s.get(self.i) {
                if !c.is_ascii_digit() {
                    break;
                }
                exp = exp.saturating_mul(10).saturating_add(i32::from(c - b'0'));
                self.i += 1;
            }
            if self.i == exp_start {
                return Err(self.err("expected a digit in the exponent"));
            }
            exponent = exponent.saturating_add(if negate { -exp } else { exp });
        }

        Ok(Json::Num(f64_from_parts(positive, significand, exponent)))
    }
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
    }

    #[test]
    fn scalars_and_containers() {
        assert_eq!(p("null"), Json::Null);
        assert_eq!(p("true"), Json::Bool(true));
        assert_eq!(p(" false "), Json::Bool(false));
        assert_eq!(p("0"), Json::Num(0.0));
        assert_eq!(p("-12"), Json::Num(-12.0));
        assert_eq!(p("1.5e2"), Json::Num(150.0));
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
        assert!(matches!(v, Json::Str(Cow::Borrowed("plain"))));
        let v = p(r#""with\nescape""#);
        assert!(matches!(v, Json::Str(Cow::Owned(_))));
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
        ] {
            assert!(
                Json::parse(bad).is_err(),
                "should have been rejected: {bad:?}"
            );
        }
    }

    #[test]
    fn depth_is_capped() {
        // Well under the cap parses; far past it errors instead of overflowing the stack.
        let ok = "[".repeat(MAX_DEPTH - 1) + &"]".repeat(MAX_DEPTH - 1);
        assert!(Json::parse(&ok).is_ok());
        let too_deep = "[".repeat(MAX_DEPTH + 5) + &"]".repeat(MAX_DEPTH + 5);
        let err = Json::parse(&too_deep).expect_err("should refuse");
        assert_eq!(err.msg, "nested too deeply");
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
                (Json::Num(x), serde_json::Value::Number(y)) => match y.as_f64() {
                    Some(y) if x.to_bits() == y.to_bits() => Ok(()),
                    Some(y) => bad(&format!(
                        "number {x} ({:016x}) vs {y} ({:016x})",
                        x.to_bits(),
                        y.to_bits()
                    )),
                    None => bad("serde_json number is not an f64"),
                },
                (Json::Str(x), serde_json::Value::String(y)) if x.as_ref() == y.as_str() => Ok(()),
                (Json::Str(x), serde_json::Value::String(y)) => {
                    bad(&format!("string {:?} vs {:?}", x.as_ref(), y.as_str()))
                }
                (Json::Arr(x), serde_json::Value::Array(y)) => {
                    if x.len() != y.len() {
                        return bad(&format!("array len {} vs {}", x.len(), y.len()));
                    }
                    for (i, (a, b)) in x.iter().zip(y).enumerate() {
                        same(a, b, &format!("{at}[{i}]"))?;
                    }
                    Ok(())
                }
                (Json::Obj(x), serde_json::Value::Object(y)) => {
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
