//! `Tokenizer.encode_file`: toki's streaming encode driver behind a Python iterator.
//!
//! Worker threads cut the file into ~`chunk` byte pieces where encoding restarts identically (`stream_cut`: before
//! a `\n` with an ASCII letter or digit on both sides, or before a space for a splitting `Metaspace`), read each piece with pread and encode it with
//! `encode_into` (no `Encoding`). Finished pieces wait in a reorder buffer; the iterator hands them out in file
//! order, one numpy `uint32` array each, without a copy. Workers run at most `window` pieces ahead of the
//! consumer, so memory stays bounded whatever the file size, and the GIL is released while Python waits.
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tk_encode::pipeline::{EncodeOptions, PipelineTokenizer as Pipeline};

use crate::type_hints::U32Array;

struct State {
    claimed: usize,                      // next piece a worker takes
    consumed: usize,                     // next piece the iterator hands out
    ready: BTreeMap<usize, Vec<u32>>,    // finished pieces not yet handed out
    error: Option<String>,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
}

/// Iterator over the token ids of a file, one `uint32` numpy array per piece, in file order.
#[pyclass(module = "tokenizers")]
pub struct EncodeStream {
    shared: Arc<Shared>,
    n_pieces: usize,
    workers: Vec<JoinHandle<()>>,
}

/// First cut >= x (or the end), read forward in 64 KiB steps: before a `\n` with an ASCII letter or digit on both
/// sides, or before a space.
fn cut(f: &File, size: usize, x: usize, sep: Option<u8>, scratch: &mut Vec<u8>) -> std::io::Result<usize> {
    if x == 0 {
        return Ok(0);
    }
    let Some(sep) = sep.filter(|_| x < size) else {
        return Ok(size);
    };
    let mut from = x - 1;
    loop {
        let n = (64usize << 10).min(size - from);
        scratch.resize(n, 0);
        f.read_exact_at(scratch, from as u64)?;
        for c in from + 1..from + n - 1 {
            let b = &scratch[c - from - 1..c - from + 2];
            if b[1] == sep && (sep == b' ' || (b[0].is_ascii_alphanumeric() && b[2].is_ascii_alphanumeric())) {
                return Ok(c);
            }
        }
        if from + n >= size {
            return Ok(size);
        }
        from += n - 2;
    }
}

impl EncodeStream {
    pub fn start(pipeline: &Pipeline, path: &str, options: EncodeOptions, threads: usize, chunk: usize, window: usize) -> PyResult<Self> {
        let f = Arc::new(File::open(path).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?);
        let size = f.metadata().map_err(|e| PyValueError::new_err(e.to_string()))?.len() as usize;
        let sep = pipeline.stream_cut();
        let n_pieces = if sep.is_some() { size.div_ceil(chunk).max(1) } else { 1 };
        let shared = Arc::new(Shared {
            state: Mutex::new(State { claimed: 0, consumed: 0, ready: BTreeMap::new(), error: None }),
            cv: Condvar::new(),
        });
        let options = Arc::new(options);
        let workers = (0..threads.clamp(1, n_pieces))
            .map(|_| {
                let (shared, f, pipeline, options) = (shared.clone(), f.clone(), pipeline.clone(), options.clone());
                std::thread::spawn(move || {
                    let (mut data, mut scratch, mut toks) = (Vec::new(), Vec::new(), Vec::new());
                    loop {
                        let i = {
                            let mut st = shared.state.lock().unwrap();
                            while st.claimed < n_pieces && st.claimed >= st.consumed + window && st.error.is_none() {
                                st = shared.cv.wait(st).unwrap();
                            }
                            if st.claimed >= n_pieces || st.error.is_some() {
                                return;
                            }
                            st.claimed += 1;
                            st.claimed - 1
                        };
                        let piece = (|| -> Result<Vec<u32>, String> {
                            let lo = cut(&f, size, i * chunk, sep, &mut scratch).map_err(|e| e.to_string())?;
                            let hi = if i + 1 == n_pieces { size } else { cut(&f, size, (i + 1) * chunk, sep, &mut scratch).map_err(|e| e.to_string())? };
                            toks.clear();
                            if hi > lo {
                                data.resize(hi - lo, 0);
                                f.read_exact_at(&mut data, lo as u64).map_err(|e| e.to_string())?;
                                pipeline.encode_into(&String::from_utf8_lossy(&data), &options, &mut toks).map_err(|e| e.to_string())?;
                            }
                            Ok(toks.iter().map(|t| t.id()).collect())
                        })();
                        let mut st = shared.state.lock().unwrap();
                        match piece {
                            Ok(ids) => {
                                st.ready.insert(i, ids);
                            }
                            Err(e) => st.error = Some(e),
                        }
                        shared.cv.notify_all();
                    }
                })
            })
            .collect();
        Ok(EncodeStream { shared, n_pieces, workers })
    }
}

#[pymethods]
impl EncodeStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<U32Array<'py>>> {
        let (shared, n) = (self.shared.clone(), self.n_pieces);
        let ids = py.detach(move || {
            let mut st = shared.state.lock().unwrap();
            loop {
                if let Some(e) = st.error.take() {
                    return Err(e);
                }
                if st.consumed >= n {
                    return Ok(None);
                }
                let k = st.consumed;
                if let Some(ids) = st.ready.remove(&k) {
                    st.consumed += 1;
                    shared.cv.notify_all();
                    return Ok(Some(ids));
                }
                st = shared.cv.wait(st).unwrap();
            }
        });
        match ids.map_err(PyValueError::new_err)? {
            Some(ids) => Ok(Some(U32Array(PyArray1::from_vec(py, ids)))),
            None => {
                for w in self.workers.drain(..) {
                    let _ = w.join();
                }
                Ok(None)
            }
        }
    }
}

impl Drop for EncodeStream {
    fn drop(&mut self) {
        // an abandoned iterator: stop the workers (they exit at their next claim)
        self.shared.state.lock().unwrap().error.get_or_insert_with(|| "dropped".into());
        self.shared.cv.notify_all();
    }
}
