//! Exports the fields of an `Encoding` as DLPack tensors.
//!
//! DLPack is the protocol array libraries use to hand each other tensors.
//! `torch.tensor(x)` calls `x.__dlpack__()` and reads the tensor that call returns.
//! The `dlpark` crate builds that tensor.
//!
//! Every export is a copy, widened to `int64`.
//! torch models take `int64` ids, and torch lets its user write to any tensor it receives.
//!
//! # What `__dlpack__` returns
//!
//! A DLPack tensor is a small C struct. It describes memory that lives elsewhere:
//!
//! - `data`: the address of the numbers, next to their shape, element type and device.
//! - `manager_ctx`: an opaque pointer to whatever owns that memory. Here, a boxed `Vec<i64>`.
//! - `deleter`: a function the reader calls when it is done. It frees the owner.
//!
//! The reader only uses `data`, and the deleter only uses `manager_ctx`.
//! Nothing in the struct ties the two together, so [`int64_tensor`] points both at the same `Vec`.
//!
//! Python carries the struct in a capsule, an object that wraps a C pointer.
//! The struct has two layouts: the legacy one, and the versioned one that adds a version and flags.
//! The reader picks the layout with `max_version`.

use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;

use crate::encoding::Encoding;

/// Where every export lives: host memory.
const DEVICE: dlpark::ffi::DLDevice = dlpark::ffi::DLDevice::CPU;

/// Which field of an `Encoding` a `DLPackExporter` exports.
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

/// One field of an `Encoding`, for array libraries to read through DLPack.
///
/// `torch.tensor` and `numpy.from_dlpack` each get their own `int64` copy of the field.
#[pyclass(frozen, module = "tokenizers")]
pub struct DLPackExporter {
    encoding: Py<Encoding>,
    field: Field,
}

impl DLPackExporter {
    pub(crate) fn new(encoding: Bound<'_, Encoding>, field: Field) -> Self {
        Self {
            encoding: encoding.unbind(),
            field,
        }
    }
}

#[pymethods]
impl DLPackExporter {
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let request = dlpark::python::ExportRequest::parse(stream, max_version, dl_device, copy)?;
        // The export widens to `int64`, so it cannot share the encoding's `u32` buffer.
        if request.copy() == Some(false) {
            return Err(PyBufferError::new_err(
                "copy=False is not supported: an Encoding field is always exported as a new int64 \
                 array. Use copy=None or copy=True.",
            ));
        }
        if request.device().is_some_and(|device| device != DEVICE) {
            return Err(PyBufferError::new_err(
                "dl_device is not supported: an Encoding field is always exported to CPU memory. \
                 Move the array to another device after reading it.",
            ));
        }
        let values = self.field.values(self.encoding.get());
        // `max_version=(1, 0)` or later asks for a versioned capsule.
        let capsule = match request.abi() {
            dlpark::python::ExportAbi::Versioned => {
                int64_tensor::<dlpark::ffi::DLManagedTensorVersioned>(values)?.into_pyobject(py)?
            }
            dlpark::python::ExportAbi::Legacy => {
                int64_tensor::<dlpark::ffi::DLManagedTensor>(values)?.into_pyobject(py)?
            }
        };
        Ok(capsule.unbind())
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        (DEVICE.device_type.0, DEVICE.device_id)
    }
}

/// Copies `values` into a new `int64` buffer and builds the DLPack tensor that owns it.
///
/// `M` is the layout of the tensor: legacy or versioned.
fn int64_tensor<M: dlpark::ManagedTensorBase>(values: &[u32]) -> PyResult<dlpark::Managed<M>> {
    // Copies the internal &[u32] and widen it to Vec<i64>
    let mut widened: Vec<i64> = values.iter().map(|&value| i64::from(value)).collect();
    let shape = [widened.len()];
    let strides = [1];
    // A pointer to the data of widened.
    // Read before the `Vec` moves into a box. Moving a `Vec` does not move the data buffer.
    let data_ptr = widened.as_mut_ptr();
    // `manager_ctx` is one pointer wide (void*) and a `Vec` is three, so the owner is a `Box<Vec>`.
    // The tensor's deleter drops `owner`, which frees the `Vec` and its data.
    let owner = Box::new(widened);
    let mut tensor = dlpark::metadata::Fixed::new(shape, strides)
        .initialize_as::<M>(owner)
        .map_err(|error| PyBufferError::new_err(error.to_string()))?;
    tensor
        .set_data(data_ptr.cast())
        .set_dtype(dlpark::ffi::DLDataType::of::<i64>())
        .set_device(DEVICE);
    // SAFETY: `IS_COPIED` says the reader gets memory nobody else holds.
    unsafe { tensor.set_flags_unchecked(dlpark::DlpackFlags::IS_COPIED) };
    // SAFETY: `data_ptr` points at the buffer of `owner`, which the deleter frees.
    // `shape` is that buffer's length.
    Ok(unsafe { tensor.finish() })
}
