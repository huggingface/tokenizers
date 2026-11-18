//! Exports token ids as a DLPack tensor.
//!
//! DLPack is the protocol array libraries use to hand each other tensors.
//! `torch.tensor(x)` calls `x.__dlpack__()` and expects a capsule that holds a managed tensor.
//! The `dlpark` crate builds the tensor and its capsule.
//!
//! [`int64_capsule`] exports a copy of the ids, widened to `int64`.
//! torch models take `int64` ids, and torch lets its user write to any tensor it receives.

use dlpark::ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned};
use dlpark::metadata::Fixed;
use dlpark::python::ExportRequest;
use dlpark::{DlpackFlags, Managed, ManagedTensorBase};
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;

/// Where every export lives: host memory.
pub(crate) const DEVICE: DLDevice = DLDevice::CPU;

/// Copies `ids` to an `int64` tensor and returns the DLPack capsule that owns it.
///
/// `request` holds the arguments of `__dlpack__`. It picks the capsule's ABI.
pub(crate) fn int64_capsule<'py>(
    py: Python<'py>,
    ids: &[u32],
    request: &ExportRequest<'py>,
) -> PyResult<Py<PyAny>> {
    request.export_zero_copy(
        py,
        DEVICE,
        DlpackFlags::empty(),
        || int64_tensor::<DLManagedTensor>(ids),
        || int64_tensor::<DLManagedTensorVersioned>(ids),
    )
}

fn int64_tensor<M: ManagedTensorBase>(ids: &[u32]) -> PyResult<Managed<M>> {
    let mut wide: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
    let data = wide.as_mut_ptr();
    let mut tensor = Fixed::new([wide.len()], [1])
        .initialize_as::<M>(Box::new(wide))
        .map_err(|error| PyBufferError::new_err(error.to_string()))?;
    tensor
        .set_data(data.cast())
        .set_dtype(DLDataType::of::<i64>())
        .set_device(DEVICE);
    // SAFETY: `data` points at the heap buffer of `wide`, which the tensor now owns and frees in
    // its deleter. Moving a `Vec` does not move its buffer. The shape is the buffer's length.
    Ok(unsafe { tensor.finish() })
}
