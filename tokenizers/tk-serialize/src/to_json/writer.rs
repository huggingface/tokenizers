//! The JSON text. `Out` tracks one bit per open container -- whether it is still empty -- which
//! is all a comma needs.

#[cfg(any(feature = "unigram", test))]
use crate::vendored::f64_from_literal;
#[cfg(any(feature = "unigram", test))]
use tk_encode::tokenizer::Result;

pub(super) struct Out {
    buf: String,
    empty: Vec<bool>,
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

    fn sep(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
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

    #[cfg(feature = "unigram")]
    /// Behind `unigram` because a Unigram score is the only float a `tokenizer.json` holds.
    pub(super) fn f64(&mut self, value: f64) -> Result<()> {
        let literal = float_literal(value)?;
        self.sep();
        self.buf.push_str(&literal);
        Ok(())
    }

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

    pub(super) fn field_opt_str(&mut self, key: &str, value: Option<&str>) {
        match value {
            Some(value) => self.field_str(key, value),
            None => self.field_null(key),
        }
    }

    pub(super) fn type_tag(&mut self, tag: &str) {
        self.field_str("type", tag);
    }
}

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

/// Control characters below `0x20` without a short escape become `\u00XX`; `hifijson` rejects raw
/// ones. Nothing above `0x7F` is escaped -- JSON strings are UTF-8 already.
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

#[cfg(any(feature = "unigram", test))]
const MAX_PRECISION: usize = 17;

#[cfg(any(feature = "unigram", test))]
/// A float spelled so that *this crate's own parser* reads it back to the identical bits.
///
/// The usual "shortest form that round-trips" rule is defined against a correctly-rounded parser,
/// and [`crate::vendored`]'s deliberately is not one -- it reproduces `serde_json`'s arithmetic, so
/// it can land a ULP off. The shortest form is therefore a candidate, checked through the same
/// `f64_from_literal` the reader uses; the precision ladder is the fallback and has never fired on
/// real data.
pub(super) fn float_literal(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(format!("cannot write the non-finite number {value:?} as JSON").into());
    }
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

#[cfg(any(feature = "unigram", test))]
/// `0` becomes `0.0`: JSON does not distinguish them, but `Tokenizer::save` has always written the
/// point and other readers hand a bare `0` back as an integer.
fn keep_it_a_float(literal: String) -> String {
    if literal.contains(['.', 'e', 'E']) {
        literal
    } else {
        literal + ".0"
    }
}
