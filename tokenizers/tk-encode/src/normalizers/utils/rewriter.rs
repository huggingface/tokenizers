use std::borrow::Cow;
use std::ops::Range;

/// Builds a normalizer's output when only parts of the input must change.
/// Inputs that don't require changes are returned as Cow::Borrowed
pub(crate) struct Rewriter<'a> {
    input: &'a str,
    cursor: usize,
    output: Option<String>,
}

impl<'a> Rewriter<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Self {
            input,
            cursor: 0,
            output: None,
        }
    }

    /// Copies the input bytes up to `range`, skips `range`, then returns a &mut to
    /// the output so the caller can write the replacement for range into it.
    /// Ranges must come in ascending order and must not overlap.
    pub(crate) fn rewrite(&mut self, range: Range<usize>) -> &mut String {
        debug_assert!(self.cursor <= range.start && range.start <= range.end);
        let out = self
            .output
            .get_or_insert_with(|| String::with_capacity(self.input.len()));
        out.push_str(&self.input[self.cursor..range.start]);
        self.cursor = range.end;
        out
    }

    pub(crate) fn finish(self) -> Cow<'a, str> {
        if let Some(mut owned) = self.output {
            owned.push_str(&self.input[self.cursor..]);
            Cow::Owned(owned)
        } else {
            Cow::Borrowed(self.input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewriter_borrows_without_rewrite() {
        for input in ["The lazy dog jumps over", ""] {
            assert!(
                matches!(Rewriter::new(input).finish(), Cow::Borrowed(_)),
                "{input:?}"
            );
        }
    }

    #[test]
    fn rewriter_owns_after_a_rewrite() {
        let mut rewriter = Rewriter::new("The lazy dog jumps over");
        rewriter.rewrite(0.."The lazy".len()).push_str("catzz");
        assert!(matches!(rewriter.finish(), Cow::Owned(s) if s == "catzz dog jumps over"));
    }

    #[test]
    fn rewriter_splices_replacements_into_the_input() {
        type Replacements<'a> = &'a [(Range<usize>, &'a str)];
        let cases: [(&str, Replacements, &str); 6] = [
            ("hello world", &[(0..5, "HI")], "HI world"),
            ("hello world", &[(6..11, "")], "hello "),
            ("hello world", &[(0..11, "x")], "x"),
            ("hello world", &[(0..1, "HH"), (6..11, "w")], "HHello w"),
            (
                "hello world",
                &[(0..5, "a"), (5..6, "b"), (6..6, "c")],
                "abcworld",
            ),
            ("café 中文", &[(3..5, "e"), (6..9, "")], "cafe 文"),
        ];
        for (input, replacements, expected) in cases {
            let mut rewriter = Rewriter::new(input);
            for (range, text) in replacements {
                rewriter.rewrite(range.clone()).push_str(text);
            }
            assert_eq!(rewriter.finish(), expected, "{input:?} {replacements:?}");
        }
    }
}
