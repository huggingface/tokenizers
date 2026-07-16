use std::borrow::Cow;

mod norm;
mod scan;
#[cfg(target_arch = "aarch64")]
mod simd_norm;
mod tables;

pub fn nfd(input: &str) -> Cow<'_, str> {
    norm::decompose::<false, true>(input)
}
pub fn nfkd(input: &str) -> Cow<'_, str> {
    norm::decompose::<true, true>(input)
}
pub fn nfc(input: &str) -> Cow<'_, str> {
    norm::compose::<false, true>(input)
}
pub fn nfkc(input: &str) -> Cow<'_, str> {
    norm::compose::<true, true>(input)
}
pub fn nfd_char(c: char, f: impl FnMut(char)) {
    norm::nfd_char(c, f)
}

pub fn lowercase(input: &str) -> Cow<'_, str> {
    scan::lowercase::<true>(input)
}
pub fn strip_accents(input: &str) -> Cow<'_, str> {
    scan::strip_accents::<true>(input)
}
pub fn nmt(input: &str) -> Cow<'_, str> {
    scan::nmt::<true>(input)
}
pub fn bert(
    input: &str,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
) -> Cow<'_, str> {
    scan::bert::<true>(
        input,
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
    )
}

pub struct Scanner(Box<scan::Scan>);

impl std::fmt::Debug for Scanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Scanner")
    }
}

impl Scanner {
    pub fn new(bmp: &[u64; 1024], astral_hot: bool) -> Self {
        Scanner(Box::new(scan::Scan::build_runtime(bmp, astral_hot)))
    }
    pub fn next_member(&self, input: &str, i: usize) -> usize {
        self.0.next_member::<true>(input.as_bytes(), i)
    }
    pub fn contains(&self, c: char) -> bool {
        self.0.contains(c as u32)
    }
}

#[doc(hidden)]
pub mod scalar {
    use std::borrow::Cow;
    pub fn nfd(input: &str) -> Cow<'_, str> {
        crate::norm::decompose::<false, false>(input)
    }
    pub fn nfkd(input: &str) -> Cow<'_, str> {
        crate::norm::decompose::<true, false>(input)
    }
    pub fn nfc(input: &str) -> Cow<'_, str> {
        crate::norm::compose::<false, false>(input)
    }
    pub fn nfkc(input: &str) -> Cow<'_, str> {
        crate::norm::compose::<true, false>(input)
    }
    pub fn lowercase(input: &str) -> Cow<'_, str> {
        crate::scan::lowercase::<false>(input)
    }
    pub fn strip_accents(input: &str) -> Cow<'_, str> {
        crate::scan::strip_accents::<false>(input)
    }
    pub fn nmt(input: &str) -> Cow<'_, str> {
        crate::scan::nmt::<false>(input)
    }
    pub fn bert(input: &str, ct: bool, cc: bool, sa: bool, lc: bool) -> Cow<'_, str> {
        crate::scan::bert::<false>(input, ct, cc, sa, lc)
    }
}
