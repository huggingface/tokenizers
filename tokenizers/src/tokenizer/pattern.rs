use crate::parallelism::get_parallelism;
use crate::utils::SysRegex;
use crate::{Offsets, Result};
use rayon::current_num_threads;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;

const MIN_CHUNK_SIZE: usize = 8 * 1024; // 8KB
const CHUNK_OVERLAP: usize = 1024; // 1KB

/// Pattern used to split a NormalizedString
pub trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>>;
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let is_char = |c: char| -> bool { c == *self };
        is_char.find_matches(inside)
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if self.is_empty() {
            // If we try to find the matches with an empty string, just don't match anything
            return Ok(vec![((0, inside.chars().count()), false)]);
        }

        let re = Regex::new(&regex::escape(self))?;
        (&re).find_matches(inside)
    }
}

impl Pattern for &String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let s: &str = self;
        s.find_matches(inside)
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.len() > 2 * MIN_CHUNK_SIZE && get_parallelism() {
            return parallel_find_matches_with_config(*self, inside, MIN_CHUNK_SIZE, CHUNK_OVERLAP);
        }
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for m in self.find_iter(inside) {
            if prev != m.start() {
                splits.push(((prev, m.start()), false));
            }
            splits.push(((m.start(), m.end()), true));
            prev = m.end();
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}

impl Pattern for &SysRegex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.len() > 2 * MIN_CHUNK_SIZE && get_parallelism() {
            return parallel_find_matches_with_config(*self, inside, MIN_CHUNK_SIZE, CHUNK_OVERLAP);
        }
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for (start, end) in self.find_iter(inside) {
            if prev != start {
                splits.push(((prev, start), false));
            }
            splits.push(((start, end), true));
            prev = end;
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut last_offset = 0;
        let mut last_seen = 0;

        let mut matches = inside
            .char_indices()
            .flat_map(|(b, c)| {
                last_seen = b + c.len_utf8();
                if self(c) {
                    let mut events = Vec::with_capacity(2);
                    if last_offset < b {
                        // We need to emit what was before this match
                        events.push(((last_offset, b), false));
                    }
                    events.push(((b, b + c.len_utf8()), true));
                    last_offset = b + c.len_utf8();
                    events
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        // Do not forget the last potential split
        if last_seen > last_offset {
            matches.push(((last_offset, last_seen), false));
        }

        Ok(matches)
    }
}

/// Invert the `is_match` flags for the wrapped Pattern. This is useful
/// for example when we use a regex that matches words instead of a delimiter,
/// and we want to match the delimiter.
pub struct Invert<P: Pattern>(pub P);
impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        Ok(self
            .0
            .find_matches(inside)?
            .into_iter()
            .map(|(offsets, flag)| (offsets, !flag))
            .collect())
    }
}

struct OverlappingChunks<'a> {
    s: &'a str,
    chunk_size: usize,
    overlap: usize,
    pos: usize,
}

struct Chunk<'a> {
    text: &'a str,
    authority_start: usize,
    authority_end: usize,
}

impl<'a> OverlappingChunks<'a> {
    fn new(s: &'a str, chunk_size: usize, overlap: usize) -> Self {
        Self {
            s,
            chunk_size,
            overlap,
            pos: 0,
        }
    }
}

impl<'a> Iterator for OverlappingChunks<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.s.len() {
            return None;
        }

        let authority_start = self.pos;

        let mut authority_end = (self.pos + self.chunk_size).min(self.s.len());
        while authority_end < self.s.len() && !self.s.is_char_boundary(authority_end) {
            authority_end += 1;
        }

        let mut chunk_end = (authority_end + self.overlap).min(self.s.len());
        while chunk_end < self.s.len() && !self.s.is_char_boundary(chunk_end) {
            chunk_end += 1;
        }

        self.pos = authority_end;

        Some(Chunk {
            text: &self.s[authority_start..chunk_end],
            authority_start,
            authority_end,
        })
    }
}

fn parallel_find_matches_with_config<P: Pattern + Sync>(
    pattern: P,
    inside: &str,
    min_chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<(Offsets, bool)>> {
    if inside.len() <= 2 * min_chunk_size {
        return pattern.find_matches(inside);
    }

    let n_chunks = current_num_threads().min(inside.len() / min_chunk_size);

    // Split the string into overlapping chunks, find matches in each chunk in parallel
    let chunks: Vec<_> =
        OverlappingChunks::new(inside, inside.len() / n_chunks, chunk_overlap).collect();
    let matches: Vec<Vec<(Offsets, bool)>> = chunks
        .par_iter()
        .map(|chunk| -> Result<Vec<(Offsets, bool)>> {
            let local_matches = pattern.find_matches(chunk.text)?;
            Ok(local_matches
                .into_iter()
                .map(|((s, e), is_match)| {
                    (
                        (s + chunk.authority_start, e + chunk.authority_start),
                        is_match,
                    )
                })
                .filter(|((s, _e), is_match)| *is_match && *s < chunk.authority_end)
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;

    // Merge results
    let matches: Vec<_> = matches.into_iter().flatten().collect();
    let mut i = 0;
    let mut merged = Vec::new();
    let mut prev_end = 0;

    while i < matches.len() {
        let (s, e) = matches[i].0;

        if s >= prev_end {
            // Normal match
            if s > prev_end {
                merged.push(((prev_end, s), false));
            }
            merged.push(matches[i]);
            prev_end = e;
            i += 1;
        } else {
            // Ghost region, skip matches that start before prev_end
            let mut max_ghost_end = 0;
            while i < matches.len() && matches[i].0 .0 < prev_end {
                max_ghost_end = max_ghost_end.max(matches[i].0 .1);
                i += 1;
            }
            // If a ghost region extends past prev_end, last match was truncated, we need to fix
            if max_ghost_end > prev_end {
                if let Some(((trunc_start, trunc_end), _)) = merged.last_mut() {
                    if let Some((_, new_end)) =
                        find_one_from(&pattern, inside, *trunc_start, chunk_overlap)?
                    {
                        *trunc_end = new_end;
                        prev_end = new_end;
                    }
                }
            }

            if i < matches.len() && matches[i].0 .0 > prev_end {
                let mut pos = prev_end;
                while pos < inside.len() {
                    match find_one_from(&pattern, inside, pos, chunk_overlap)? {
                        Some((ms, me)) => {
                            if matches[i].0 == (ms, me) {
                                break;
                            }
                            if prev_end < ms {
                                merged.push(((prev_end, ms), false));
                            }
                            merged.push(((ms, me), true));
                            prev_end = me;
                            pos = me;
                        }
                        _ => break,
                    }
                }
            }
        }
    }

    if prev_end < inside.len() {
        merged.push(((prev_end, inside.len()), false));
    }

    Ok(merged)
}

fn find_one_from<P: Pattern>(
    pattern: &P,
    inside: &str,
    from: usize,
    chunk_overlap: usize,
) -> Result<Option<(usize, usize)>> {
    for n in 1..=8 {
        let mut window_end = (from + chunk_overlap * n).min(inside.len());
        while window_end < inside.len() && !inside.is_char_boundary(window_end) {
            window_end += 1;
        }
        let window = &inside[from..window_end];
        let result = pattern
            .find_matches(window)?
            .into_iter()
            .find(|(_, is_match)| *is_match)
            .map(|((s, e), _)| (s + from, e + from));

        match result {
            Some((s, e)) if e < window_end => return Ok(Some((s, e))),
            Some(_) if window_end < inside.len() => continue,
            Some((s, e)) => return Ok(Some((s, e))),
            None if window_end < inside.len() => continue,
            None => return Ok(None),
        }
    }
    Ok(pattern
        .find_matches(&inside[from..])?
        .into_iter()
        .find(|(_, is_match)| *is_match)
        .map(|((s, e), _)| (s + from, e + from)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    macro_rules! do_test {
        ($inside: expr, $pattern: expr => @ERROR) => {
            assert!($pattern.find_matches($inside).is_err());
        };
        ($inside: expr, $pattern: expr => $result: expr) => {
            assert_eq!($pattern.find_matches($inside).unwrap(), $result);
            assert_eq!(
                Invert($pattern).find_matches($inside).unwrap(),
                $result
                    .into_iter()
                    .map(|v: (Offsets, bool)| (v.0, !v.1))
                    .collect::<Vec<_>>()
            );
        };
    }

    #[test]
    fn char() {
        do_test!("aba", 'a' => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", 'a' => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", 'a' => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", 'a' => vec![((0, 0), false)]);
        do_test!("aaa", 'b' => vec![((0, 3), false)]);
    }

    #[test]
    fn str() {
        do_test!("aba", "a" => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", "a" => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", "a" => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("aabbb", "ab" => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
        do_test!("aabbab", "ab" =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );
        do_test!("", "" => vec![((0, 0), false)]);
        do_test!("aaa", "" => vec![((0, 3), false)]);
        do_test!("aaa", "b" => vec![((0, 3), false)]);
    }

    #[test]
    fn functions() {
        let is_b = |c| c == 'b';
        do_test!("aba", is_b => vec![((0, 1), false), ((1, 2), true), ((2, 3), false)]);
        do_test!("aaaab", is_b => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("bbaaa", is_b => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", is_b => vec![((0, 0), false)]);
        do_test!("aaa", is_b => vec![((0, 3), false)]);
    }

    #[test]
    fn regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }

    #[test]
    fn sys_regex() {
        let is_whitespace = SysRegex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }

    #[test]
    fn parallel_correctness() {
        let patterns = vec![
            SysRegex::new(r"\s+").unwrap(),
            SysRegex::new(r"\w+|[^\w\s]+").unwrap(),
            SysRegex::new(
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            )
            .unwrap(),
        ];

        let long_input = "NoSpacesAtAllInThisVeryLongWord repeated "
            .repeat(20)
            .trim()
            .to_string();

        let inputs: Vec<&str> = vec![
            "hello world foo bar baz",
            "a   b   c   d   e   f   g   h   i   j",
            "Hello, world! This is a test. Numbers: 123, 456.",
            "Unicode: café résumé naïve 日本語テスト",
            "Short",
            "",
            &long_input,
        ];

        for pattern in &patterns {
            for input in &inputs {
                let sequential = pattern.find_matches(input).unwrap();
                let parallel = parallel_find_matches_with_config(pattern, input, 5, 5).unwrap();
                assert_eq!(
                    sequential,
                    parallel,
                    "Mismatch for input: '{}'",
                    &input[..input.len().min(50)]
                );
            }
        }
    }
}
