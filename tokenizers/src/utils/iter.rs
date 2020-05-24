//! This comes from the Rust libcore and is duplicated here because it is not exported
//! (cf https://github.com/rust-lang/rust/blob/25091ed9b7739e12466fb2490baa1e8a2815121c/src/libcore/iter/adapters/mod.rs#L2664)
//! We are now using the version from https://stackoverflow.com/questions/44544323/how-to-unzip-a-sequence-of-resulta-b-e-to-a-veca-vecb-and-stop-on-f
//! because the one from the libcore seems to cause overflowing stacks in some cases
//! It also contains a lines_with_ending that copies std::io::BufRead but keeps line endings.
use std::io::BufRead;

pub struct ResultShunt<I, E> {
    iter: I,
    error: Option<E>,
}

impl<I, T, E> ResultShunt<I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    /// Process the given iterator as if it yielded a `T` instead of a
    /// `Result<T, _>`. Any errors will stop the inner iterator and
    /// the overall result will be an error.
    pub fn process<F, U>(iter: I, mut f: F) -> Result<U, E>
    where
        F: FnMut(&mut Self) -> U,
    {
        let mut shunt = ResultShunt::new(iter);
        let value = f(shunt.by_ref());
        shunt.reconstruct(value)
    }

    fn new(iter: I) -> Self {
        ResultShunt { iter, error: None }
    }

    /// Consume the adapter and rebuild a `Result` value. This should
    /// *always* be called, otherwise any potential error would be
    /// lost.
    fn reconstruct<U>(self, val: U) -> Result<U, E> {
        match self.error {
            None => Ok(val),
            Some(e) => Err(e),
        }
    }
}

impl<I, T, E> Iterator for ResultShunt<I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => {
                self.error = Some(e);
                None
            }
            None => None,
        }
    }
}

/// Copied from std::io::BufRead but keep newline characters.
#[derive(Debug)]
pub struct Lines<B> {
    buf: B,
}

pub trait LinesWithEnding<B> {
    fn lines_with_ending(self) -> Lines<B>;
}

impl<B> LinesWithEnding<B> for B
where
    B: BufRead,
{
    fn lines_with_ending(self) -> Lines<B> {
        Lines::<B> { buf: self }
    }
}
impl<B: BufRead> Iterator for Lines<B> {
    type Item = std::io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = String::new();
        match self.buf.read_line(&mut buf) {
            Ok(0) => None,
            Ok(_n) => {
                // if buf.ends_with('\n') {
                //     buf.pop();
                //     if buf.ends_with('\r') {
                //         buf.pop();
                //     }
                // }
                Some(Ok(buf))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
