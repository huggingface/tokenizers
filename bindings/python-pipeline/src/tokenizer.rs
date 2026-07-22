use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyStopIteration, PyTypeError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use tk_encode::Tokenizer as SpecTokenizer;
use tk_encode::pipeline::{
    Model as _, PipelineModelScratch, PipelineToken, PipelineTokenizer, Span,
};
use tk_encode::tokenizer::PostProcessor as _;
use tk_encode::utils::parallelism::get_parallelism;
use tk_train::{TokenizerTrainExt, Trainable};

use crate::added_token::{TokenInput, parse_tokens};
use crate::detached_lock::{Detached, DetachedRwLock};
use crate::error::{TokenizersError, to_pyerr};
use crate::models::{PyModel, wrap_model};
use crate::normalizers::{PyNormalizer, wrap_normalizer};
use crate::pre_tokenizers::{PyPreTokenizer, wrap_pre_tokenizer};
use crate::trainers::PyTrainer;

/// Set when the bindings actually run a rayon-parallel section, so the
/// pthread_atfork handler only disables parallelism in children of processes
/// that really used it (mirrors the v1 bindings' semantics).
pub static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);

/// The compiled encode path plus the facts about the spec the encode calls
/// need without re-locking it.
#[derive(Clone)]
struct Compiled {
    pipe: Arc<PipelineTokenizer>,
    /// Whether the spec's post-processor would add special tokens. Post-processing
    /// is not wired into the pipeline yet, so encode(add_special_tokens=True)
    /// must fail loudly instead of silently dropping them.
    post_adds_special_tokens: bool,
}

struct Inner {
    /// Source of truth: the mutable, serializable tokenizer definition.
    spec: SpecTokenizer,
    /// Memoized compilation of `spec`; invalidated by every mutation.
    compiled: Option<Compiled>,
}

fn poisoned<G>(_: std::sync::PoisonError<G>) -> PyErr {
    PyRuntimeError::new_err("tokenizer lock poisoned")
}

/// A tokenizer: a model plus its optional normalizer and pre-tokenizer.
/// Mutations apply to the serializable definition; encode runs a compiled
/// pipeline that is rebuilt automatically after any change.
// The lock/GIL ordering rule (never block on the lock while attached) is
// enforced by DetachedRwLock: guards are only reachable inside its
// detach-first `with` closure. See detached_lock.rs for the rationale and
// the residual hole.
#[pyclass(frozen, name = "Tokenizer", module = "tokenizers_pipeline")]
pub struct PyTokenizer {
    inner: DetachedRwLock<Inner>,
}

impl PyTokenizer {
    fn from_spec(spec: SpecTokenizer) -> Self {
        Self {
            inner: DetachedRwLock::new(Inner {
                spec,
                compiled: None,
            }),
        }
    }

    fn read_spec<T: Ungil + Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&SpecTokenizer) -> T + Ungil + Send,
    ) -> PyResult<T> {
        self.inner.with(py, |lock| {
            let guard = lock.read().map_err(poisoned)?;
            Ok(f(&guard.spec))
        })
    }

    /// Write access to the spec; invalidates the compiled pipeline.
    fn mutate_spec<T: Ungil + Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut SpecTokenizer) -> PyResult<T> + Ungil + Send,
    ) -> PyResult<T> {
        self.inner.with(py, |lock| {
            let mut guard = lock.write().map_err(poisoned)?;
            let result = f(&mut guard.spec)?;
            guard.compiled = None;
            Ok(result)
        })
    }
}

/// Get the compiled pipeline, building it from the spec on first use after a
/// mutation. The `Detached` parameter is the proof this runs off the GIL.
fn get_or_compile(lock: &Detached<'_, Inner>) -> PyResult<Compiled> {
    {
        let guard = lock.read().map_err(poisoned)?;
        if let Some(compiled) = &guard.compiled {
            return Ok(compiled.clone());
        }
    }
    let mut guard = lock.write().map_err(poisoned)?;
    if guard.compiled.is_none() {
        let pipe = PipelineTokenizer::try_from(&guard.spec).map_err(|e| {
            TokenizersError::new_err(format!(
                "this tokenizer cannot be compiled to an encode pipeline: {e}"
            ))
        })?;
        let post_adds_special_tokens = guard
            .spec
            .get_post_processor()
            .is_some_and(|p| p.added_tokens(false) > 0);
        guard.compiled = Some(Compiled {
            pipe: Arc::new(pipe),
            post_adds_special_tokens,
        });
    }
    Ok(guard.compiled.clone().expect("just set"))
}

fn check_special_tokens_flag(compiled: &Compiled, add_special_tokens: bool) -> PyResult<()> {
    if add_special_tokens && compiled.post_adds_special_tokens {
        return Err(PyNotImplementedError::new_err(
            "this tokenizer's post-processor adds special tokens, but post-processing is not \
             implemented in the encode pipeline yet; pass add_special_tokens=False to encode \
             without them",
        ));
    }
    Ok(())
}

fn encode_one(
    pipe: &PipelineTokenizer,
    text: &str,
    pre_tokens: &mut Vec<Span>,
    scratch: &mut PipelineModelScratch,
) -> PyResult<Vec<u32>> {
    let mut output: Vec<PipelineToken> = Vec::new();
    pipe.encode_generic::<{ PipelineTokenizer::STAGE_MODEL }>(
        text,
        pre_tokens,
        scratch,
        &mut output,
    )
    .map_err(to_pyerr)?;
    Ok(output.iter().map(|t| t.id).collect())
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model: PyRef<'_, PyModel>) -> Self {
        Self::from_spec(SpecTokenizer::new(model.inner.clone()))
    }

    /// Load a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    #[pyo3(signature = (path) -> "Tokenizer")]
    fn from_file(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let spec = py
            .detach(|| SpecTokenizer::from_file(path))
            .map_err(to_pyerr)?;
        Ok(Self::from_spec(spec))
    }

    /// Load a tokenizer from the bytes of a `tokenizer.json` file.
    #[staticmethod]
    #[pyo3(signature = (buffer) -> "Tokenizer")]
    fn from_buffer(py: Python<'_>, buffer: Vec<u8>) -> PyResult<Self> {
        let spec = py
            .detach(|| SpecTokenizer::from_bytes(&buffer))
            .map_err(to_pyerr)?;
        Ok(Self::from_spec(spec))
    }

    /// Download `tokenizer.json` from a model on the Hugging Face Hub (requires
    /// the `huggingface_hub` package) and load it.
    #[staticmethod]
    #[pyo3(signature = (identifier, *, revision = String::from("main"), token = None) -> "Tokenizer")]
    fn from_pretrained(
        py: Python<'_>,
        identifier: &str,
        revision: String,
        token: Option<String>,
    ) -> PyResult<Self> {
        let hub = py.import("huggingface_hub")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("repo_id", identifier)?;
        kwargs.set_item("filename", "tokenizer.json")?;
        kwargs.set_item("revision", revision)?;
        kwargs.set_item("token", token)?;
        let path: PathBuf = hub
            .getattr("hf_hub_download")?
            .call((), Some(&kwargs))?
            .extract()?;
        Self::from_file(py, path)
    }

    /// Serialize the tokenizer definition as a `tokenizer.json` string.
    #[pyo3(signature = (*, pretty = false))]
    fn to_str(&self, py: Python<'_>, pretty: bool) -> PyResult<String> {
        self.read_spec(py, move |spec| spec.to_string(pretty).map_err(to_pyerr))?
    }

    /// Save the tokenizer definition to a `tokenizer.json` file.
    #[pyo3(signature = (path, *, pretty = true))]
    fn save(&self, py: Python<'_>, path: PathBuf, pretty: bool) -> PyResult<()> {
        self.read_spec(py, move |spec| spec.save(path, pretty).map_err(to_pyerr))?
    }

    /// Encode `text` into token ids.
    ///
    /// Runs entirely outside the interpreter lock and returns a `numpy.uint32`
    /// array backed by the Rust output buffer (no copy).
    #[pyo3(signature = (text, *, add_special_tokens = true) -> "npt.NDArray[np.uint32]")]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        add_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let ids = self.inner.with(py, |lock| -> PyResult<Vec<u32>> {
            let compiled = get_or_compile(&lock)?;
            check_special_tokens_flag(&compiled, add_special_tokens)?;
            let mut pre_tokens = Vec::new();
            let mut scratch = compiled.pipe.get_model().init_scratch();
            encode_one(&compiled.pipe, text, &mut pre_tokens, &mut scratch)
        })?;
        Ok(ids.into_pyarray(py))
    }

    /// Encode a batch of texts, in parallel across Rust threads (respects
    /// `TOKENIZERS_PARALLELISM`), without holding the interpreter lock.
    /// Input strings are borrowed, not copied; each output is a `numpy.uint32`
    /// array backed by its Rust buffer.
    #[pyo3(signature = (texts, *, add_special_tokens = true) -> "list[npt.NDArray[np.uint32]]")]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<PyBackedStr>,
        add_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let batches = self.inner.with(py, |lock| -> PyResult<Vec<Vec<u32>>> {
            let compiled = get_or_compile(&lock)?;
            check_special_tokens_flag(&compiled, add_special_tokens)?;
            if get_parallelism() && texts.len() > 1 {
                USED_PARALLELISM.store(true, Ordering::SeqCst);
                texts
                    .par_iter()
                    .map_init(
                        || (Vec::new(), compiled.pipe.get_model().init_scratch()),
                        |(pre_tokens, scratch), text| {
                            encode_one(&compiled.pipe, text, pre_tokens, scratch)
                        },
                    )
                    .collect()
            } else {
                let mut pre_tokens = Vec::new();
                let mut scratch = compiled.pipe.get_model().init_scratch();
                texts
                    .iter()
                    .map(|text| encode_one(&compiled.pipe, text, &mut pre_tokens, &mut scratch))
                    .collect()
            }
        })?;
        let list = PyList::empty(py);
        for ids in batches {
            list.append(ids.into_pyarray(py))?;
        }
        Ok(list)
    }

    /// Not implemented yet: decoding is not part of the encode pipeline.
    #[pyo3(signature = (ids, *, skip_special_tokens = true))]
    #[allow(unused_variables)]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        Err(PyNotImplementedError::new_err(
            "decode is not implemented in the encode pipeline yet",
        ))
    }

    /// Train the model on text files (one sequence per line).
    #[pyo3(signature = (files, *, trainer = None))]
    fn train(
        &self,
        py: Python<'_>,
        files: Vec<String>,
        trainer: Option<PyRef<'_, PyTrainer>>,
    ) -> PyResult<()> {
        let explicit = trainer.map(|t| t.inner.clone());
        self.inner.with(py, |lock| {
            let mut guard = lock.write().map_err(poisoned)?;
            let mut trainer = explicit.unwrap_or_else(|| guard.spec.get_model().get_trainer());
            guard
                .spec
                .train_from_files(&mut trainer, files)
                .map_err(to_pyerr)?;
            guard.compiled = None;
            Ok(())
        })
    }

    /// Train the model from any iterator of `str`.
    ///
    /// The interpreter lock is only re-acquired to refill an internal buffer
    /// (256 sequences at a time); the training itself runs multi-threaded in
    /// Rust with the lock released.
    #[pyo3(signature = (iterator, *, trainer = None))]
    fn train_from_iterator(
        &self,
        py: Python<'_>,
        iterator: &Bound<'_, PyAny>,
        trainer: Option<PyRef<'_, PyTrainer>>,
    ) -> PyResult<()> {
        let explicit = trainer.map(|t| t.inner.clone());
        let sequences = BufferedPyIterator::new(iterator)?;
        let error = sequences.error.clone();
        self.inner.with(py, |lock| {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
            let mut guard = lock.write().map_err(poisoned)?;
            let mut trainer = explicit.unwrap_or_else(|| guard.spec.get_model().get_trainer());
            guard
                .spec
                .train(&mut trainer, sequences)
                .map_err(to_pyerr)?;
            guard.compiled = None;
            Ok::<_, PyErr>(())
        })?;
        if let Some(err) = error.lock().expect("error slot poisoned").take() {
            return Err(err);
        }
        Ok(())
    }

    /// Add tokens to the vocabulary, matched literally in the input text.
    fn add_tokens(&self, py: Python<'_>, tokens: Vec<TokenInput>) -> PyResult<usize> {
        let tokens = parse_tokens(tokens, false);
        self.mutate_spec(py, move |spec| spec.add_tokens(tokens).map_err(to_pyerr))
    }

    /// Add special tokens (never split, skipped on decode) to the vocabulary.
    fn add_special_tokens(&self, py: Python<'_>, tokens: Vec<TokenInput>) -> PyResult<usize> {
        let tokens = parse_tokens(tokens, true);
        self.mutate_spec(py, move |spec| {
            spec.add_special_tokens(tokens).map_err(to_pyerr)
        })
    }

    fn token_to_id(&self, py: Python<'_>, token: &str) -> PyResult<Option<u32>> {
        self.read_spec(py, |spec| spec.token_to_id(token))
    }

    fn id_to_token(&self, py: Python<'_>, id: u32) -> PyResult<Option<String>> {
        self.read_spec(py, move |spec| spec.id_to_token(id))
    }

    /// The whole vocabulary as a dict. This copies every entry; prefer
    /// `token_to_id` for lookups.
    #[pyo3(signature = (*, with_added_tokens = true))]
    fn get_vocab(&self, py: Python<'_>, with_added_tokens: bool) -> PyResult<HashMap<String, u32>> {
        self.read_spec(py, move |spec| spec.get_vocab(with_added_tokens))
    }

    #[pyo3(signature = (*, with_added_tokens = true))]
    fn get_vocab_size(&self, py: Python<'_>, with_added_tokens: bool) -> PyResult<usize> {
        self.read_spec(py, move |spec| spec.get_vocab_size(with_added_tokens))
    }

    /// The model in use by this tokenizer (a copy: reassign to change it).
    #[getter]
    fn model(&self, py: Python<'_>) -> PyResult<Py<PyModel>> {
        let model = self.read_spec(py, |spec| spec.get_model().clone())?;
        wrap_model(py, model)
    }

    #[setter]
    fn set_model(&self, py: Python<'_>, model: PyRef<'_, PyModel>) -> PyResult<()> {
        let model = model.inner.clone();
        self.mutate_spec(py, move |spec| {
            spec.with_model(model);
            Ok(())
        })
    }

    /// The optional normalizer in use by this tokenizer (a copy: reassign to
    /// change it).
    #[getter]
    fn normalizer(&self, py: Python<'_>) -> PyResult<Option<Py<PyNormalizer>>> {
        let normalizer = self.read_spec(py, |spec| spec.get_normalizer().cloned())?;
        normalizer.map(|n| wrap_normalizer(py, n)).transpose()
    }

    #[setter]
    fn set_normalizer(
        &self,
        py: Python<'_>,
        normalizer: Option<PyRef<'_, PyNormalizer>>,
    ) -> PyResult<()> {
        let normalizer = normalizer.map(|n| n.inner.clone());
        self.mutate_spec(py, move |spec| {
            spec.with_normalizer(normalizer).map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// The optional pre-tokenizer in use by this tokenizer (a copy: reassign
    /// to change it).
    #[getter]
    fn pre_tokenizer(&self, py: Python<'_>) -> PyResult<Option<Py<PyPreTokenizer>>> {
        let pre_tokenizer = self.read_spec(py, |spec| spec.get_pre_tokenizer().cloned())?;
        pre_tokenizer.map(|p| wrap_pre_tokenizer(py, p)).transpose()
    }

    #[setter]
    fn set_pre_tokenizer(
        &self,
        py: Python<'_>,
        pre_tokenizer: Option<PyRef<'_, PyPreTokenizer>>,
    ) -> PyResult<()> {
        let pre_tokenizer = pre_tokenizer.map(|p| p.inner.clone());
        self.mutate_spec(py, move |spec| {
            spec.with_pre_tokenizer(pre_tokenizer);
            Ok(())
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.read_spec(py, |spec| {
            format!(
                "Tokenizer(model={}, vocab_size={})",
                match spec.get_model() {
                    tk_encode::ModelWrapper::BPE(_) => "BPE",
                    tk_encode::ModelWrapper::WordPiece(_) => "WordPiece",
                    tk_encode::ModelWrapper::WordLevel(_) => "WordLevel",
                    tk_encode::ModelWrapper::Unigram(_) => "Unigram",
                },
                spec.get_vocab_size(true)
            )
        })
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let data = self.to_str(py, false)?;
        let from_buffer = py.get_type::<PyTokenizer>().getattr("from_buffer")?;
        Ok((from_buffer, (PyBytes::new(py, data.as_bytes()),)))
    }
}

/// Pulls a Python iterator of `str` from Rust threads: re-attaches to the
/// interpreter only to refill an internal buffer, `CHUNK` items at a time.
/// A conversion error stops the stream and is stashed in `error` for the
/// caller to surface once training finishes.
struct BufferedPyIterator {
    iterator: Py<PyAny>,
    buffer: std::collections::VecDeque<String>,
    finished: bool,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl BufferedPyIterator {
    const CHUNK: usize = 256;

    fn new(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            iterator: iterable.try_iter()?.unbind().into(),
            buffer: std::collections::VecDeque::with_capacity(Self::CHUNK),
            finished: false,
            error: Arc::new(Mutex::new(None)),
        })
    }

    // The vetted lock-then-GIL direction: the caller (train) holds the write
    // lock and re-attaches here. Safe because no attached thread can be
    // blocking on the lock — DetachedRwLock makes that unrepresentable.
    #[allow(clippy::disallowed_methods)]
    fn refill(&mut self) {
        let result = Python::attach(|py| -> PyResult<bool> {
            let iterator = self.iterator.bind(py);
            for _ in 0..Self::CHUNK {
                match iterator.call_method0("__next__") {
                    Ok(item) => {
                        let sequence = item.extract::<String>().map_err(|_| {
                            PyTypeError::new_err("train_from_iterator expects an iterator of str")
                        })?;
                        self.buffer.push_back(sequence);
                    }
                    Err(e) if e.is_instance_of::<PyStopIteration>(py) => return Ok(true),
                    Err(e) => return Err(e),
                }
            }
            Ok(false)
        });
        match result {
            Ok(done) => self.finished = done,
            Err(e) => {
                *self.error.lock().expect("error slot poisoned") = Some(e);
                self.finished = true;
            }
        }
    }
}

impl Iterator for BufferedPyIterator {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.buffer.is_empty() && !self.finished {
            self.refill();
        }
        self.buffer.pop_front()
    }
}
