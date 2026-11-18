use std::ffi::c_int;
use std::{mem, ptr};

use dlpark::python::ExportRequest;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

use crate::dlpack;
use crate::encoding::Encoding;

/// Which field of an `Encoding` an `EncodingBuffer` exposes.
pub(crate) enum Field {
    Ids,
    TypeIds,
    AttentionMask,
}

impl Field {
    fn values<'a>(&self, encoding: &'a Encoding) -> &'a [u32] {
        match self {
            Self::Ids => encoding.ids(),
            Self::TypeIds => encoding.type_ids(),
            Self::AttentionMask => encoding.attention_mask(),
        }
    }
}

/// One field of an `Encoding`, as a read-only buffer of `uint32`.
///
/// It implements the Python buffer protocol over the encoding's own memory, so `numpy.asarray`
/// and `memoryview` read the field without a copy. `torch.tensor` copies it to an `int64` tensor.
#[pyclass(frozen, module = "tokenizers")]
pub struct EncodingBuffer {
    encoding: Py<Encoding>,
    field: Field,
    /// The number of values. The buffer's `shape` points at it.
    len: isize,
}

/// The size of one value in bytes. The buffer's `strides` points at it.
static ITEM_SIZE: isize = mem::size_of::<u32>() as isize;

impl EncodingBuffer {
    pub(crate) fn new(encoding: Bound<'_, Encoding>, field: Field) -> Self {
        let len = field.values(encoding.get()).len() as isize;
        Self {
            encoding: encoding.unbind(),
            field,
            len,
        }
    }

    fn as_slice(&self) -> &[u32] {
        self.field.values(self.encoding.get())
    }
}

#[pymethods]
impl EncodingBuffer {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        // Writing through the buffer would change the `Encoding`, which other threads read.
        // `torch.frombuffer` and `torch.asarray` ask for unformatted bytes and return a writable
        // tensor over them, whatever `readonly` says.
        if flags & ffi::PyBUF_WRITABLE != 0 || flags & ffi::PyBUF_FORMAT == 0 {
            return Err(PyBufferError::new_err(
                "an Encoding buffer is read-only uint32: read it with numpy.asarray or memoryview, \
                 copy it with torch.tensor",
            ));
        }
        let buffer = slf.get();
        let values = buffer.as_slice();
        let shape = &raw const buffer.len;
        // SAFETY: CPython hands a valid `view` to fill. `view.obj` keeps `slf` alive while the view
        // lives, and `slf` keeps the `Encoding` alive. Both are frozen, so `values` and `shape` stay
        // valid and unchanged. `readonly` tells consumers not to write through any pointer.
        unsafe {
            (*view).buf = values.as_ptr().cast_mut().cast();
            (*view).len = mem::size_of_val(values) as isize;
            (*view).readonly = 1;
            (*view).itemsize = ITEM_SIZE;
            (*view).format = c"I".as_ptr().cast_mut();
            (*view).ndim = 1;
            (*view).shape = if flags & ffi::PyBUF_ND != 0 {
                shape.cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
                (&raw const ITEM_SIZE).cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }

    // DLPack is the protocol `torch.tensor` reads. torch ignores DLPack's read-only flag, so this
    // exports a copy, never the encoding's own memory. That answers `copy` whatever its value.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = copy;
        let request = ExportRequest::parse(stream, max_version, dl_device, None)?;
        dlpack::int64_capsule(py, self.as_slice(), &request)
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        (dlpack::DEVICE.device_type.0, dlpack::DEVICE.device_id)
    }
}
