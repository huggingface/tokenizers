extern crate lazy_static;
extern crate libc;

#[cfg(feature = "opencc")]
extern crate opencc_rust;

use crate::tokenizer::{NormalizedString, Normalizer, Result};

use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

use fnv::FnvHashMap;
use fnv::FnvHashSet;
use std::sync::Mutex;

#[cfg(feature = "opencc")]
use opencc_rust::{DefaultConfig, OpenCC};

use std::panic;

use std::sync::Once;
static START: Once = Once::new();

#[cfg(not(feature = "opencc"))]
use libc::c_void;
#[cfg(not(feature = "opencc"))]
use std::path::Path;
#[cfg(not(feature = "opencc"))]
use std::ptr::null_mut;

#[cfg(not(feature = "opencc"))]
#[derive(Debug, Copy, Clone)]
pub enum DefaultConfig {
    S2T,
    T2S,
}

#[cfg(not(feature = "opencc"))]
impl AsRef<Path> for DefaultConfig {
    fn as_ref(&self) -> &Path {
        Path::new("s2t.json")
    }
}

#[cfg(not(feature = "opencc"))]
#[allow(dead_code)]
pub struct OpenCC {
    opencc: *mut c_void,
}

#[cfg(not(feature = "opencc"))]
unsafe impl Send for OpenCC {}

#[cfg(not(feature = "opencc"))]
unsafe impl Sync for OpenCC {}

#[cfg(not(feature = "opencc"))]
impl OpenCC {
    /// Create a new OpenCC instance through a file provided by its path.
    pub fn new<P: AsRef<Path>>(_config_file_path: P) -> Result<OpenCC> {
        let opencc = null_mut();
        Ok(OpenCC { opencc })
    }
    pub fn convert<S: AsRef<str>>(&self, input: S) -> String {
        String::from(input.as_ref())
    }
}

static mut OPENCC_CONFIG: DefaultConfig = DefaultConfig::S2T;

lazy_static! {
    static ref OPENCC: Option<OpenCC> = unsafe {
        let result = panic::catch_unwind(|| OpenCC::new(OPENCC_CONFIG).unwrap());
        match result {
            Ok(v) => Some(v),
            Err(_e) => None,
        }
    };
    static ref SYMBOLS_: Mutex<FnvHashSet<char>> = Mutex::new(FnvHashSet::default());
    static ref SIMPL_MAPPING_: Mutex<FnvHashSet<char>> = Mutex::new(FnvHashSet::default());
    static ref ZH_NORM_MAPPING_: Mutex<FnvHashMap<char, String>> =
        Mutex::new(FnvHashMap::default());
}

static SEPARATE_INTEGERS: u32 = 1 << 1;
static SEPARATE_SYMBOLS: u32 = 1 << 2;
static SIMPL_TO_TRAD: u32 = 1 << 3;
static TRAD_TO_SIMPL: u32 = 1 << 4;
static ZH_NORM_MAPPING: u32 = 1 << 5;

/// Checks opencc is installed
#[cfg(feature = "opencc")]
pub fn opencc_enabled() -> bool {
    OPENCC.as_ref().is_some()
}

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        true
    } else {
        c.is_whitespace()
    }
}

/// match for numbers
fn is_number(c: char) -> bool {
    matches!(c as usize, 0x30..=0x39)
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        false
    } else {
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        c.is_other()
    }
}

/// Checks whether a character is chinese
/// This defines a "chinese character" as anything in the CJK Unicode block:
///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
///
/// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
/// despite its name. The modern Korean Hangul alphabet is a different block,
/// as is Japanese Hiragana and Katakana. Those alphabets are used to write
/// space-separated words, so they are not treated specially and handled
/// like for all of the other languages.
fn is_chinese_char(c: char) -> bool {
    match c as usize {
        0x4E00..=0x9FFF => true,
        0x3400..=0x4DBF => true,
        0x20000..=0x2A6DF => true,
        0x2A700..=0x2B73F => true,
        0x2B740..=0x2B81F => true,
        0x2B920..=0x2CEAF => true,
        0xF900..=0xFAFF => true,
        0x2F800..=0x2FA1F => true,
        _ => false,
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub struct BertNormalizer {
    /// Whether to do the bert basic cleaning:
    ///   1. Remove any control characters
    ///   2. Replace all sorts of whitespace by the classic one ` `
    pub clean_text: bool,
    /// Whether to put spaces around chinese characters so they get split
    pub handle_chinese_chars: bool,
    /// Whether to strip accents
    pub strip_accents: Option<bool>,
    /// Whether to lowercase the input
    pub lowercase: bool,
    /// Normalization Options
    pub norm_options: u32,
}

impl Default for BertNormalizer {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
            norm_options: 0,
        }
    }
}

impl BertNormalizer {
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
        norm_options: u32,
    ) -> Self {
        START.call_once(|| {
            let mut zh_norm_mapping = ZH_NORM_MAPPING_.lock().unwrap();
            let mut simpl_mapping = SIMPL_MAPPING_.lock().unwrap();
            let mut symbols = SYMBOLS_.lock().unwrap();

            if (norm_options & SIMPL_TO_TRAD) != 0 {
                unsafe {
                    OPENCC_CONFIG = DefaultConfig::S2T;
                }
            }

            if (norm_options & TRAD_TO_SIMPL) != 0 {
                unsafe {
                    OPENCC_CONFIG = DefaultConfig::T2S;
                }
            }

            if (norm_options & ZH_NORM_MAPPING) != 0 {
                zh_norm_mapping.insert(0 as char, " ".to_string());
                for line in include_str!("zh_char2str_mapping.txt").lines() {
                    let mut pair = line.split('\t');
                    let left = pair.next().unwrap().chars().next().unwrap();
                    let right = pair.next().unwrap();
                    zh_norm_mapping.insert(left, right.to_string());
                }
                for line in include_str!("s2t.txt").lines() {
                    let mut pair = line.split('\t');
                    let left = pair.next().unwrap().chars().next().unwrap();
                    simpl_mapping.insert(left);

                    let right_ = pair.next();
                    if right_ != None {
                        let right = right_.unwrap().chars().next().unwrap();
                        if !zh_norm_mapping.contains_key(&left) {
                            let value = zh_norm_mapping.get(&right);
                            if let Some(rep) = value {
                                let value2 = rep.clone();
                                zh_norm_mapping.insert(left, value2.to_string());
                            } else {
                                zh_norm_mapping.insert(left, right.to_string());
                            };
                        }
                    }
                }
            }
            if (norm_options & SEPARATE_SYMBOLS) != 0 {
                for c in include_str!("symbols.txt").chars() {
                    symbols.insert(c);
                }
            }
        });

        BertNormalizer {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
            norm_options,
        }
    }

    fn do_clean_text(&self, normalized: &mut NormalizedString) {
        normalized
            .filter(|c| !(c as usize == 0 || c as usize == 0xfffd || is_control(c)))
            .map(|c| if is_whitespace(c) { ' ' } else { c });
    }

    fn do_handle_chinese_chars(&self, normalized: &mut NormalizedString, norm_options: u32) {
        let mut new_chars: Vec<(char, isize)> = vec![];
        let sep_int = (norm_options & SEPARATE_INTEGERS) != 0;
        let sep_symbols = (norm_options & SEPARATE_SYMBOLS) != 0;
        let zh_norm = (norm_options & ZH_NORM_MAPPING) != 0;
        let symbols = SYMBOLS_.lock().unwrap();
        if zh_norm {
            let zh_norm_mapping = ZH_NORM_MAPPING_.lock().unwrap();
            normalized.for_each(|c| match zh_norm_mapping.get(&c) {
                Some(rep) => {
                    rep.chars().enumerate().for_each(|(i, c2)| {
                        if (is_chinese_char(c2))
                            || (sep_int && is_number(c2))
                            || (sep_symbols && symbols.contains(&c2))
                        {
                            new_chars.extend(&[
                                (' ', if i == 0 { 0 } else { 1 }),
                                (c2, 1),
                                (' ', 1),
                            ]);
                        } else {
                            new_chars.push((c2, if i == 0 { 0 } else { 1 }));
                        }
                    });
                }
                None => {
                    if (is_chinese_char(c))
                        || (sep_int && is_number(c))
                        || (sep_symbols && symbols.contains(&c))
                    {
                        new_chars.extend(&[(' ', 0), (c, 1), (' ', 1)]);
                    } else {
                        new_chars.push((c, 0));
                    };
                }
            });
        } else {
            normalized.for_each(|c| {
                if (is_chinese_char(c))
                    || (sep_int && is_number(c))
                    || (sep_symbols && symbols.contains(&c))
                {
                    new_chars.extend(&[(' ', 0), (c, 1), (' ', 1)]);
                } else {
                    new_chars.push((c, 0));
                };
            });
        }
        normalized.transform(new_chars.into_iter(), 0);
    }

    fn do_strip_accents(&self, normalized: &mut NormalizedString) {
        normalized.nfd().filter(|c| !c.is_mark_nonspacing());
    }

    fn do_lowercase(&self, normalized: &mut NormalizedString) {
        normalized.lowercase();
    }

    fn convert_zh(&self, normalized: &mut NormalizedString) {
        if let Some(opencc) = OPENCC.as_ref() {
            let mut need_clean_count = 0;
            let mut seen: FnvHashSet<char> = FnvHashSet::default();

            let simpl_mapping = SIMPL_MAPPING_.lock().unwrap();
            for c in normalized.get().chars() {
                if c as u32 > 12032 && simpl_mapping.contains(&c) && !seen.contains(&c) {
                    need_clean_count += 1;
                    if need_clean_count > 1 {
                        break;
                    }
                    seen.insert(c);
                }
            }

            if need_clean_count > 1 {
                let normalized_str = normalized.get();
                let normalized_str_arg: String;
                if normalized_str.chars().any(|c| c == '\u{00}') {
                    normalized_str_arg = normalized_str.replace("\u{00}", " ");
                } else {
                    normalized_str_arg = normalized_str.to_string();
                }

                normalized.set_normalized(opencc.convert(normalized_str_arg));
            }
        }
    }
}

impl Normalizer for BertNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.clean_text {
            self.do_clean_text(normalized);
        }
        if ((self.norm_options & SIMPL_TO_TRAD) != 0) || ((self.norm_options & TRAD_TO_SIMPL) != 0)
        {
            self.convert_zh(normalized)
        }
        if self.handle_chinese_chars {
            self.do_handle_chinese_chars(normalized, self.norm_options);
        }
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);
        if strip_accents {
            self.do_strip_accents(normalized);
        }
        if self.lowercase {
            self.do_lowercase(normalized);
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "opencc"))]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        if opencc_enabled() {
            let norm = BertNormalizer::new(
                true,
                true,
                Some(true),
                true,
                SEPARATE_INTEGERS | SEPARATE_SYMBOLS | SIMPL_TO_TRAD | ZH_NORM_MAPPING,
            );
            let mut input =
                NormalizedString::from("系列 聯系 « 联系 𠱁 氹 𥱊 栄 梊 𠹌 买书 <n> \u{00}");
            let _ = norm.normalize(&mut input).unwrap();
            assert_eq!(
                input.get(),
                " 系  列   聯  系  <<  聯  繫   o 氹   氹   席   榮   折  木   o 能   買  書  <n> "
            );

            input = NormalizedString::from("头部");
            let _ = norm.normalize(&mut input).unwrap();
            assert_eq!(input.get(), " 頭  部 ");
        }
    }
}
