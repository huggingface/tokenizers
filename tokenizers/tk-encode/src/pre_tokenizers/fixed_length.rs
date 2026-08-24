use crate::pipeline;
use crate::tokenizer::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedLength {
    /// Absent from a config means 5, which is also `FixedLength::default()`.
    pub length: usize,
}

impl FixedLength {
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

// SAFETY: every offset is one `str::char_indices` yielded, or `text.len()`, and they are pushed in
// increasing order.
unsafe impl pipeline::PreTokenizer for FixedLength {
    fn pre_tokenize(
        &self,
        text: &str,
        _scratch: &mut pipeline::PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        if self.length == 0 {
            out.push(pipeline::Span {
                start: 0,
                end: text.len() as u32,
            });
            return Ok(());
        }

        // `step_by` yields the byte offset of every `length`-th char — i.e. the
        // start of each chunk. `skip(1)` turns those into the *end* of the
        // preceding chunk; the final chunk runs to `text.len()`.
        let mut start: u32 = 0;
        for (end, _) in text.char_indices().step_by(self.length).skip(1) {
            out.push(pipeline::Span {
                start,
                end: end as u32,
            });
            start = end as u32;
        }
        out.push(pipeline::Span {
            start,
            end: text.len() as u32,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(length: usize, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = FixedLength { length };
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_basic() {
        // same expectations as the legacy `basic`/`custom_length` tests
        assert_eq!(
            pretokenize(5, "Hello world"),
            vec![("Hello", (0, 5)), (" worl", (5, 10)), ("d", (10, 11))],
        );
        assert_eq!(
            pretokenize(3, "Hello world"),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("wor", (6, 9)),
                ("ld", (9, 11)),
            ],
        );
    }

    #[test]
    fn pipeline_utf8() {
        // chunks are counted in chars; offsets are bytes (👋 is 4 bytes)
        assert_eq!(
            pretokenize(3, "Hello 👋 world"),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("👋 w", (6, 12)),
                ("orl", (12, 15)),
                ("d", (15, 16)),
            ],
        );
    }

    #[test]
    fn pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize(5, ""), empty);
        // length >= char count -> one chunk
        assert_eq!(pretokenize(5, "Short"), vec![("Short", (0, 5))]);
        assert_eq!(pretokenize(10, "abc"), vec![("abc", (0, 3))]);
        // length == 0 -> whole text as a single split (no panic)
        assert_eq!(pretokenize(0, "abc"), vec![("abc", (0, 3))]);
    }
}
