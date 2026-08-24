use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::Result;
use bitsplit::classify::mask;
use bitsplit::classes::class_runs_into;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Pre tokenizes the numbers into single tokens. If individual_digits is set
/// to true, then all digits are splitted into individual tokens.
#[non_exhaustive]
pub struct Digits {
    pub individual_digits: bool,
}

impl Digits {
    pub fn new(individual_digits: bool) -> Self {
        Self { individual_digits }
    }
}

impl Default for Digits {
    fn default() -> Self {
        Self::new(false)
    }
}

// SAFETY: the spans come from an `bitsplit` fsm, which splits only at character boundaries of `text`.
// See `bitsplit` docs.
unsafe impl pipeline::PreTokenizer for Digits {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // isolate each numeric char (`individual_digits`) or keep numeric runs — bitsplit classify +
        // class-runs FSM. atom `NUMERIC` == `char::is_numeric`, so byte-exact with the scalar path.
        let individual = self.individual_digits;
        scratch.split_on_bits(
            text.as_bytes(),
            |b, t, st, fk, _, s| {
                if individual {
                    class_runs_into::<0, { mask::NUMERIC }, 0>(b, t, st, fk, s) // isolate each digit
                } else {
                    class_runs_into::<0, 0, { mask::NUMERIC }>(b, t, st, fk, s) // keep digit runs
                }
            },
            out,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(individual_digits: bool, text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = Digits::new(individual_digits);
        let mut splits = Vec::new();
        let mut scratch = PreTokenizerScratch::default();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_contiguous() {
        // individual_digits = false: a digit run is kept as one split, matching
        // the legacy `Contiguous` behavior.
        assert_eq!(
            pretokenize(false, "Hey 123 friend!"),
            vec![("Hey ", (0, 4)), ("123", (4, 7)), (" friend!", (7, 15))],
        );
        // only digits in string
        assert_eq!(
            pretokenize(false, "123, 456, 789"),
            vec![
                ("123", (0, 3)),
                (", ", (3, 5)),
                ("456", (5, 8)),
                (", ", (8, 10)),
                ("789", (10, 13))
            ],
        )
    }

    #[test]
    fn pipeline_individual() {
        // individual_digits = true: each digit is its own split (legacy `Isolated`).
        assert_eq!(
            pretokenize(true, "Hey 123 friend!"),
            vec![
                ("Hey ", (0, 4)),
                ("1", (4, 5)),
                ("2", (5, 6)),
                ("3", (6, 7)),
                (" friend!", (7, 15)),
            ],
        );

        // only digits in string
        assert_eq!(
            pretokenize(true, "123, 456, 789"),
            vec![
                ("1", (0, 1)),
                ("2", (1, 2)),
                ("3", (2, 3)),
                (", ", (3, 5)),
                ("4", (5, 6)),
                ("5", (6, 7)),
                ("6", (7, 8)),
                (", ", (8, 10)),
                ("7", (10, 11)),
                ("8", (11, 12)),
                ("9", (12, 13))
            ],
        )
    }

    #[test]
    fn pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize(false, ""), empty);
        // all digits -> one split (contiguous) / per-digit (individual)
        assert_eq!(pretokenize(false, "123"), vec![("123", (0, 3))]);
        assert_eq!(
            pretokenize(true, "123"),
            vec![("1", (0, 1)), ("2", (1, 2)), ("3", (2, 3))],
        );
        // single-char runs split at each digit/non-digit boundary
        assert_eq!(
            pretokenize(false, "1a2"),
            vec![("1", (0, 1)), ("a", (1, 2)), ("2", (2, 3))],
        );
        // multibyte: é is 2 bytes, so the non-digit run "café" spans 0..5
        assert_eq!(
            pretokenize(false, "café2"),
            vec![("café", (0, 5)), ("2", (5, 6))],
        );
    }
}
