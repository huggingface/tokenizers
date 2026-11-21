use fancy_regex::Regex;
use std::borrow::Cow;
use std::error::Error;

/// 0.23 (Oniguruma) and 1.x (`fancy-regex` -> `regex` crate, UTS#18) disagree on
/// exactly eight code points for `\w`:
///
/// - ² ³ ¹ ¼ ½ ¾ (U+00B2, U+00B3, U+00B9, U+00BC, U+00BD, U+00BE) were word
///   characters in 0.23 and are not in UTS#18.
/// - ZWNJ / ZWJ (U+200C, U+200D) were not word characters in 0.23 and are
///   `Join_Control` word characters in UTS#18.
///
/// These replacements are the old sets expressed with the current engine's
/// syntax. Set difference comes before union: `[\u{00B2}[\w--[...]]]` adds the
/// Latin-1 numbers back while removing the join controls.
const V023_WORD: &str =
    r"[\u{00B2}\u{00B3}\u{00B9}\u{00BC}\u{00BD}\u{00BE}[\w--[\u{200C}\u{200D}]]]";
const V023_NON_WORD: &str =
    r"[\u{200C}\u{200D}[\W--[\u{00B2}\u{00B3}\u{00B9}\u{00BC}\u{00BD}\u{00BE}]]]";
static V023_BOUNDARY: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!("(?:(?<={V023_WORD})(?!{V023_WORD})|(?<!{V023_WORD})(?={V023_WORD}))")
});
static V023_NON_BOUNDARY: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!("(?:(?<={V023_WORD})(?={V023_WORD})|(?<!{V023_WORD})(?!{V023_WORD}))")
});

/// Rewrites the four word-character escapes to their 0.23 equivalents.
///
/// `\b` and `\B` are word boundaries in 0.23; inside a character class they are
/// backspace and must be left alone. All other escapes (including `\\w`, a
/// literal backslash followed by `w`) pass through unchanged.
fn restore_v023_word_classes(pattern: &str) -> Cow<'_, str> {
    if !pattern.contains('\\') {
        return Cow::Borrowed(pattern);
    }

    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    let mut class_start = false;

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek().copied() {
                Some('w') => {
                    chars.next();
                    out.push_str(V023_WORD);
                }
                Some('W') => {
                    chars.next();
                    out.push_str(V023_NON_WORD);
                }
                Some('b') if !in_class => {
                    chars.next();
                    out.push_str(&V023_BOUNDARY);
                }
                Some('B') if !in_class => {
                    chars.next();
                    out.push_str(&V023_NON_BOUNDARY);
                }
                Some(next) => {
                    chars.next();
                    out.push('\\');
                    out.push(next);
                }
                None => out.push('\\'),
            }
            continue;
        }

        match c {
            '[' => {
                in_class = true;
                class_start = true;
                out.push(c);
            }
            '^' if class_start => out.push(c),
            ']' if class_start => {
                // A leading `]` is a literal, not the end of the class.
                class_start = false;
                out.push(c);
            }
            ']' => {
                in_class = false;
                out.push(c);
            }
            _ => {
                class_start = false;
                out.push(c);
            }
        }
    }

    Cow::Owned(out)
}

#[derive(Debug)]
pub struct SysRegex {
    regex: Regex,
}

impl SysRegex {
    pub(crate) fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches {
            matches: self.regex.find_iter(inside),
            previous_end: 0,
        }
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let pattern = restore_v023_word_classes(regex_str);
        Ok(Self {
            regex: Regex::new(&pattern)?,
        })
    }
}

/// The matches of a [`SysRegex`], checked to come in order without overlaps.
///
/// `Split` builds spans from these matches and its `unsafe impl PreTokenize` relies on this check.
pub struct Matches<'r, 't> {
    matches: fancy_regex::Matches<'r, 't, str>,
    previous_end: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = crate::Result<(usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        // stop if an error is encountered
        let found = self.matches.next()?.ok()?;
        let (start, end) = (found.start(), found.end());
        if start < self.previous_end || end < start {
            return Some(Err(format!(
                "the regex matched bytes {start}..{end}, a range that is reversed or overlaps the \
                 previous match, which ended at byte {}.",
                self.previous_end,
            )
            .into()));
        }
        self.previous_end = end;
        Some(Ok((start, end)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(pattern: &str, inside: &str) -> bool {
        SysRegex::new(pattern)
            .unwrap()
            .find_iter(inside)
            .next()
            .is_some()
    }

    /// The eight code points that changed between 0.23 and 1.x.
    #[test]
    fn word_class_matches_v023_character_set() {
        for c in ['²', '³', '¹', '¼', '½', '¾'] {
            assert!(
                matches(r"\w", &c.to_string()),
                "U+{:04X} must be a word char",
                c as u32
            );
            assert!(
                !matches(r"\W", &c.to_string()),
                "U+{:04X} must not be a non-word char",
                c as u32
            );
        }
        for c in ['\u{200c}', '\u{200d}'] {
            assert!(
                !matches(r"\w", &c.to_string()),
                "U+{:04X} must not be a word char",
                c as u32
            );
            assert!(
                matches(r"\W", &c.to_string()),
                "U+{:04X} must be a non-word char",
                c as u32
            );
        }
        assert!(matches(r"\w", "a"));
        assert!(!matches(r"\W", "a"));
    }

    #[test]
    fn word_class_works_in_quantifiers_and_from_json() {
        // The issue's reproducer: `a` + No chars must be one word, and the
        // join controls must split it.
        let re = SysRegex::new(r"\w+").unwrap();
        let hits: Vec<(usize, usize)> = re
            .find_iter("a²³¹¼½¾\u{200c}\u{200d}b")
            .map(|m| m.unwrap())
            .collect();
        assert_eq!(hits, vec![(0, 13), (19, 20)]);

        // The same class, nested inside a larger character class.
        assert!(matches(r"[\s\w]+", " ²"));
        assert!(!matches(r"[\s\w]+", "\u{200c}"));
    }

    #[test]
    fn word_boundaries_use_the_old_character_set() {
        let re = SysRegex::new(r"\b").unwrap();
        // a=word, ²=word, ZWNJ=non-word, b=word -> boundaries at 0, 3, 6, 7.
        let positions: Vec<usize> = re.find_iter("a²\u{200c}b").map(|m| m.unwrap().0).collect();
        assert_eq!(positions, vec![0, 3, 6, 7]);

        let re = SysRegex::new(r"\B").unwrap();
        let positions: Vec<usize> = re.find_iter("a²\u{200c}b").map(|m| m.unwrap().0).collect();
        assert_eq!(positions, vec![1]);
    }

    #[test]
    fn unrelated_escapes_are_untouched() {
        // `\\w` is a literal backslash followed by `w`, not a word class.
        assert!(matches(r"\\w", r"\w"));
        assert!(!matches(r"\\w", "a"));
        assert!(matches(r"\s+", " \t"));
        assert!(!matches(r"\s+", "a"));

        // `\b` inside a class is a backspace, so it must not be rewritten.
        assert!(matches(r"[\b]", "\u{0008}"));
        assert!(!matches(r"[\b]", "a"));
    }
}
