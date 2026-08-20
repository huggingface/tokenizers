//! The JSON text itself: a small emitter, string escaping, and the float rule this crate lives or
//! dies by.
//!
//! No serde, so the punctuation is this module's job. [`Out`] tracks one bit of state per open
//! container — whether it is still empty — which is all a comma needs, and it is what stops every
//! writer above from threading "is this the first member" through its own code.

// Both exist for the float path alone, which is `unigram`-only -- see `Out::f64`.
#[cfg(any(feature = "unigram", test))]
use crate::json::f64_from_literal;
#[cfg(any(feature = "unigram", test))]
use tk_encode::tokenizer::Result;

/// A JSON document under construction.
///
/// Containers are opened and closed by hand rather than through a closure. A closure API reads
/// better in isolation, but every component writer here is a `match` whose arms open the same object
/// and fill different fields, and threading a `&mut` builder through those is what a closure form
/// makes awkward.
pub(super) struct Out {
    buf: String,
    /// One flag per open container: whether it is still empty, i.e. whether the next value goes in
    /// without a leading comma. The outermost entry stands for the document itself.
    empty: Vec<bool>,
    /// Set by [`Self::key`], because the value that follows a key is not a new member and must not
    /// be given a comma of its own.
    after_key: bool,
}

impl Out {
    pub(super) fn new() -> Self {
        Self {
            buf: String::new(),
            empty: vec![true],
            after_key: false,
        }
    }

    pub(super) fn finish(self) -> String {
        debug_assert_eq!(self.empty.len(), 1, "a container was left open");
        self.buf
    }

    /// The comma between members, and nothing at all when one is not due.
    fn sep(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        // `expect`: the root entry is pushed by `new` and only popped by a `close` without an
        // `open`, which the `debug_assert` in `finish` catches.
        let empty = self.empty.last_mut().expect("a container is open");
        if *empty {
            *empty = false;
        } else {
            self.buf.push(',');
        }
    }

    fn open(&mut self, brace: char) {
        self.sep();
        self.buf.push(brace);
        self.empty.push(true);
    }

    fn close(&mut self, brace: char) {
        self.empty.pop();
        self.buf.push(brace);
    }

    pub(super) fn obj_open(&mut self) {
        self.open('{');
    }

    pub(super) fn obj_close(&mut self) {
        self.close('}');
    }

    pub(super) fn arr_open(&mut self) {
        self.open('[');
    }

    pub(super) fn arr_close(&mut self) {
        self.close(']');
    }

    /// An object key. The value written next becomes this member's value.
    pub(super) fn key(&mut self, key: &str) {
        self.sep();
        escape_into(&mut self.buf, key);
        self.buf.push(':');
        self.after_key = true;
    }

    pub(super) fn str(&mut self, value: &str) {
        self.sep();
        escape_into(&mut self.buf, value);
    }

    pub(super) fn bool(&mut self, value: bool) {
        self.sep();
        self.buf.push_str(if value { "true" } else { "false" });
    }

    pub(super) fn null(&mut self) {
        self.sep();
        self.buf.push_str("null");
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.sep();
        self.buf.push_str(itoa(u64::from(value)).as_str());
    }

    pub(super) fn usize(&mut self, value: usize) {
        self.sep();
        self.buf.push_str(itoa(value as u64).as_str());
    }

    /// A float, spelled so that this crate's own parser reads it back unchanged. See
    /// [`float_literal`], which is where all the difficulty is.
    ///
    /// Behind `unigram` because a Unigram score is the only float a `tokenizer.json` holds:
    /// a BPE's `dropout` is written as `null` unconditionally, and every other number in the
    /// format is an id or a count.
    #[cfg(feature = "unigram")]
    pub(super) fn f64(&mut self, value: f64) -> Result<()> {
        let literal = float_literal(value)?;
        self.sep();
        self.buf.push_str(&literal);
        Ok(())
    }

    // Convenience: a key and its value, which is what almost every call site wants.
    pub(super) fn field_str(&mut self, key: &str, value: &str) {
        self.key(key);
        self.str(value);
    }

    pub(super) fn field_bool(&mut self, key: &str, value: bool) {
        self.key(key);
        self.bool(value);
    }

    pub(super) fn field_u32(&mut self, key: &str, value: u32) {
        self.key(key);
        self.u32(value);
    }

    pub(super) fn field_usize(&mut self, key: &str, value: usize) {
        self.key(key);
        self.usize(value);
    }

    pub(super) fn field_null(&mut self, key: &str) {
        self.key(key);
        self.null();
    }

    /// `"key": "value"` when there is one, `"key": null` when there is not.
    ///
    /// Spelled rather than omitted because the reader's `get_some` treats an explicit `null` as
    /// absent, so the two are the same document to it, and a present key says out loud that the
    /// writer considered the field.
    pub(super) fn field_opt_str(&mut self, key: &str, value: Option<&str>) {
        match value {
            Some(value) => self.field_str(key, value),
            None => self.field_null(key),
        }
    }

    /// The `"type"` tag every component carries in the canonical format.
    pub(super) fn type_tag(&mut self, tag: &str) {
        self.field_str("type", tag);
    }
}

/// A `u64` as decimal, without `format!`'s machinery.
///
/// Small enough to be worth writing: `format!` pulls in the whole `core::fmt` width/fill/precision
/// path for every id in a 250k-entry vocabulary, and an id is at most 20 digits.
fn itoa(mut value: u64) -> ArrayStr<20> {
    let mut out = ArrayStr::<20>::new();
    if value == 0 {
        out.push(b'0');
        return out;
    }
    let mut digits = [0u8; 20];
    let mut n = 0;
    while value > 0 {
        digits[n] = b'0' + (value % 10) as u8;
        value /= 10;
        n += 1;
    }
    for i in (0..n).rev() {
        out.push(digits[i]);
    }
    out
}

/// A stack string, so [`itoa`] allocates nothing.
struct ArrayStr<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> ArrayStr<N> {
    fn new() -> Self {
        Self {
            bytes: [0; N],
            len: 0,
        }
    }

    fn push(&mut self, byte: u8) {
        self.bytes[self.len] = byte;
        self.len += 1;
    }

    fn as_str(&self) -> &str {
        // SAFETY: only ASCII digits are ever pushed.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }
}

/// A JSON string literal, quotes included.
///
/// `"` and `\` take their short escapes, and so do the five control characters that have one. Every
/// other control character below `0x20` becomes a `\u00XX`, because a raw one is not legal JSON and
/// `hifijson` -- the parser on the way back in -- rejects it rather than tolerating it. Nothing
/// above `0x7F` is escaped: JSON strings are UTF-8, a Rust `&str` is already valid UTF-8, and
/// escaping would only make the file bigger and the tokens harder to read.
fn escape_into(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u00");
                let byte = c as u32;
                out.push(char::from_digit(byte >> 4, 16).expect("a nibble is one hex digit"));
                out.push(char::from_digit(byte & 0xF, 16).expect("a nibble is one hex digit"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// How many `{:.N e}` precisions [`float_literal`] will try before giving up. 17 significant digits
/// is what pins an `f64` under a correctly-rounded parser, so a ladder that reaches it and beyond
/// has run out of honest candidates.
#[cfg(any(feature = "unigram", test))]
const MAX_PRECISION: usize = 17;

/// A float spelled so that **this crate's own parser reads it back to the identical bits**.
///
/// ## Why "shortest round-trip" is the wrong rule here
///
/// The usual rule is to emit the shortest decimal that round-trips, and `{}` does exactly that. But
/// "round-trips" is defined against a *correctly-rounded* parser, and [`crate::json`]'s is
/// deliberately not one: it reproduces `serde_json`'s default arithmetic, a `u64` significand
/// converted with `as f64` and then multiplied or divided by one power of ten. Above 2^53 that is
/// two roundings, so the result can sit a ULP away from the correctly-rounded value -- which is the
/// whole point, because ids that ship today were produced by that arithmetic.
///
/// So the shortest form is a *candidate*, not an answer. This checks it, through
/// [`f64_from_literal`], which is the same code the reader uses, and only accepts a literal whose
/// bits match. Otherwise it climbs a ladder of increasing precisions and checks each one.
///
/// ## What is actually needed
///
/// Measured over every score in every Unigram config in `data/` -- 342,102 of them, including the
/// 16,243 whose value is *not* the correctly-rounded reading of the file -- the shortest form is
/// accepted every time, at worst 17 significant digits. `shortest_form_round_trips_every_real_score`
/// is that sweep. The ladder has therefore never fired on real data; it is here because the above
/// is a measurement of one float formatter's output and not a proof, and a silent one-ULP shift in
/// a Unigram score is exactly the kind of thing that moves ids without moving a test.
///
/// A value the reader produced is reachable by construction -- the file's own literal maps to it --
/// so failure means the ladder did not find *a* spelling, not that none exists. It is reported
/// rather than approximated.
#[cfg(any(feature = "unigram", test))]
pub(super) fn float_literal(value: f64) -> Result<String> {
    if !value.is_finite() {
        // Neither has a JSON spelling, and a score that is one is a broken model rather than a
        // formatting problem.
        return Err(format!("cannot write the non-finite number {value:?} as JSON").into());
    }
    // `{}` never uses exponent notation and never emits a bare `.`, so its output is always a legal
    // JSON number; `-0.0` prints as `-0`, which reads back with its sign.
    let shortest = keep_it_a_float(value.to_string());
    if f64_from_literal(&shortest).to_bits() == value.to_bits() {
        return Ok(shortest);
    }
    for precision in 0..=MAX_PRECISION {
        let candidate = format!("{value:.precision$e}");
        if f64_from_literal(&candidate).to_bits() == value.to_bits() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "no decimal spelling of {value:?} ({:016x}) reads back to the same bits through this \
         crate's parser: tried the shortest form and {} precisions",
        value.to_bits(),
        MAX_PRECISION + 1
    )
    .into())
}

/// `0` becomes `0.0`, and any other whole number likewise.
///
/// JSON draws no line between an integer and a float, and this crate's reader does not either — it
/// reads `0` as `0.0` quite happily. But a *score* written as `0` is one that some other reader will
/// hand back as an integer, and `Tokenizer::save` has always written `0.0`. Keeping the point costs
/// two characters and keeps the column readable as what it is.
///
/// Safe by construction — appending `.0` to a decimal integer names the same value — and the caller
/// re-checks the result through the parser regardless.
#[cfg(any(feature = "unigram", test))]
fn keep_it_a_float(literal: String) -> String {
    if literal.contains(['.', 'e', 'E']) {
        literal
    } else {
        literal + ".0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{Json, JsonExt};

    /// Round-trip through the *public* accessor, not just `f64_from_literal`, so the test covers
    /// the path a reader really takes.
    fn reads_back_as(literal: &str) -> f64 {
        Json::parse(literal)
            .expect("the writer emits parseable JSON")
            .as_f64()
            .expect("a number literal reads as an f64")
    }

    #[test]
    fn floats_round_trip_through_our_own_parser() {
        for value in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            // Two real Unigram scores rather than round numbers, both read through the parser they
            // have to survive: `-3.8403830528259277` is the one
            // `matches_serde_not_from_str_on_a_real_unigram_score` pins, where the value our parser
            // gives is a ULP off the correctly-rounded one.
            f64_from_literal("-13.5321998596191"),
            f64_from_literal("-3.8403830528259277"),
            f64::MIN_POSITIVE,
            f64::MAX,
            1e-300,
            1e300,
            std::f64::consts::PI,
        ] {
            let literal = float_literal(value).expect("every finite float has a spelling");
            assert_eq!(
                reads_back_as(&literal).to_bits(),
                value.to_bits(),
                "{value:?} was written as {literal}, which reads back as {}",
                reads_back_as(&literal)
            );
        }
    }

    #[test]
    fn non_finite_numbers_are_refused_rather_than_mangled() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(float_literal(value).is_err(), "{value:?} has no JSON form");
        }
    }

    /// Every escape `hifijson` insists on, plus the ones it does not but a reader would find
    /// surprising. A raw control character is not legal JSON, so the `\u00XX` arm is not cosmetic.
    #[test]
    fn strings_escape_what_json_requires() {
        let mut out = Out::new();
        out.str("a\"b\\c\nd\te\rf\u{8}g\u{c}h\u{1}i\u{1f}j");
        let written = out.finish();
        // Spelled as an ordinary string literal rather than a raw one, so that every escape the
        // writer is expected to produce is visible here instead of being a raw control byte.
        assert_eq!(
            written, "\"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u0001i\\u001fj\"",
            "escaping changed"
        );
        assert_eq!(
            Json::parse(&written)
                .expect("escaped output parses")
                .as_str(),
            Some("a\"b\\c\nd\te\rf\u{8}g\u{c}h\u{1}i\u{1f}j"),
            "the escaped form does not read back as the original"
        );
    }

    /// Non-ASCII goes out raw, which is what keeps a byte-level vocabulary readable.
    #[test]
    fn non_ascii_is_not_escaped() {
        let mut out = Out::new();
        out.str("Ġthe▁世界");
        let written = out.finish();
        assert_eq!(written, "\"Ġthe▁世界\"");
        assert_eq!(
            Json::parse(&written).expect("parses").as_str(),
            Some("Ġthe▁世界")
        );
    }

    #[test]
    fn containers_get_their_commas() {
        let mut out = Out::new();
        out.obj_open();
        out.type_tag("Demo");
        out.field_bool("flag", true);
        out.field_u32("id", 7);
        out.field_null("nothing");
        out.key("list");
        out.arr_open();
        out.u32(1);
        out.u32(2);
        out.obj_open();
        out.field_str("k", "v");
        out.obj_close();
        out.arr_close();
        out.obj_close();
        let written = out.finish();
        assert_eq!(
            written,
            r#"{"type":"Demo","flag":true,"id":7,"nothing":null,"list":[1,2,{"k":"v"}]}"#
        );
        // And it is a document our own parser accepts, which is the property that matters.
        let parsed = Json::parse(&written).expect("emitted JSON parses");
        assert_eq!(parsed.type_tag(), Some("Demo"));
    }

    #[test]
    fn integers_are_written_without_a_fraction() {
        let mut out = Out::new();
        out.arr_open();
        out.u32(0);
        out.u32(u32::MAX);
        out.usize(50256);
        out.arr_close();
        assert_eq!(out.finish(), "[0,4294967295,50256]");
    }
}
