use smallvec::SmallVec;
use std::fmt::{self, Write};
use std::iter::Fuse;
use std::ops::Range;
use unicode_normalization::char::{
    canonical_combining_class, decompose_canonical, decompose_compatible,
};

#[derive(Clone)]
enum DecompositionType {
    Canonical,
    Compatible,
}

/// External iterator for a string decomposition's characters.
#[derive(Clone)]
pub struct Decompositions<I> {
    kind: DecompositionType,
    iter: Fuse<I>,

    // This buffer stores pairs of (canonical combining class, character),
    // pushed onto the end in text order.
    //
    // It's divided into up to three sections:
    // 1) A prefix that is free space;
    // 2) "Ready" characters which are sorted and ready to emit on demand;
    // 3) A "pending" block which stills needs more characters for us to be able
    //    to sort in canonical order and is not safe to emit.
    buffer: SmallVec<[(u8, char, isize); 4]>,
    ready: Range<usize>,
}

#[inline]
pub fn new_canonical<I: Iterator<Item = char>>(iter: I) -> Decompositions<I> {
    Decompositions {
        kind: self::DecompositionType::Canonical,
        iter: iter.fuse(),
        buffer: SmallVec::new(),
        ready: 0..0,
    }
}

#[inline]
pub fn new_compatible<I: Iterator<Item = char>>(iter: I) -> Decompositions<I> {
    Decompositions {
        kind: self::DecompositionType::Compatible,
        iter: iter.fuse(),
        buffer: SmallVec::new(),
        ready: 0..0,
    }
}

impl<I> Decompositions<I> {
    #[inline]
    fn push_back(&mut self, ch: char, first: bool) {
        let class = canonical_combining_class(ch);

        if class == 0 {
            self.sort_pending();
        }

        self.buffer.push((class, ch, if first { 0 } else { 1 }));
    }

    #[inline]
    fn sort_pending(&mut self) {
        // NB: `sort_by_key` is stable, so it will preserve the original text's
        // order within a combining class.
        self.buffer[self.ready.end..].sort_by_key(|k| k.0);
        self.ready.end = self.buffer.len();
    }

    #[inline]
    fn reset_buffer(&mut self) {
        // Equivalent to `self.buffer.drain(0..self.ready.end)` (if SmallVec
        // supported this API)
        let pending = self.buffer.len() - self.ready.end;
        for i in 0..pending {
            self.buffer[i] = self.buffer[i + self.ready.end];
        }
        self.buffer.truncate(pending);
        self.ready = 0..0;
    }

    #[inline]
    fn increment_next_ready(&mut self) {
        let next = self.ready.start + 1;
        if next == self.ready.end {
            self.reset_buffer();
        } else {
            self.ready.start = next;
        }
    }
}

impl<I: Iterator<Item = char>> Iterator for Decompositions<I> {
    type Item = (char, isize);

    #[inline]
    fn next(&mut self) -> Option<(char, isize)> {
        while self.ready.end == 0 {
            match (self.iter.next(), &self.kind) {
                (Some(ch), &DecompositionType::Canonical) => {
                    let mut first = true;
                    decompose_canonical(ch, |d| {
                        self.push_back(d, first);
                        first = false;
                    });
                }
                (Some(ch), &DecompositionType::Compatible) => {
                    let mut first = true;
                    decompose_compatible(ch, |d| {
                        self.push_back(d, first);
                        first = false;
                    });
                }
                (None, _) => {
                    if self.buffer.is_empty() {
                        return None;
                    } else {
                        self.sort_pending();

                        // This implementation means that we can call `next`
                        // on an exhausted iterator; the last outer `next` call
                        // will result in an inner `next` call. To make this
                        // safe, we use `fuse`.
                        break;
                    }
                }
            }
        }

        // We can assume here that, if `self.ready.end` is greater than zero,
        // it's also greater than `self.ready.start`. That's because we only
        // increment `self.ready.start` inside `increment_next_ready`, and
        // whenever it reaches equality with `self.ready.end`, we reset both
        // to zero, maintaining the invariant that:
        //      self.ready.start < self.ready.end || self.ready.end == self.ready.start == 0
        //
        // This less-than-obviously-safe implementation is chosen for performance,
        // minimizing the number & complexity of branches in `next` in the common
        // case of buffering then unbuffering a single character with each call.
        let (_, ch, size) = self.buffer[self.ready.start];
        self.increment_next_ready();
        Some((ch, size))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.iter.size_hint();
        (lower, None)
    }
}

impl<I: Iterator<Item = char> + Clone> fmt::Display for Decompositions<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c.0)?;
        }
        Ok(())
    }
}
