//! Exports token ids as a DLPack tensor.
//!
//! DLPack is the protocol array libraries use to hand each other tensors.
//! `torch.tensor(x)` calls `x.__dlpack__()` and expects a capsule that holds a [`DLManagedTensor`].
//! A consumer that takes the tensor renames the capsule, and later calls the tensor's `deleter`.
//!
//! [`int64_capsule`] exports a copy of the ids, widened to `int64`.
//! torch models take `int64` ids, and torch lets its user write to any tensor it receives.

use std::ffi::{CStr, c_void};
use std::ptr::{self, NonNull};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

// The four structs below follow `dlpack.h` field for field. Consumers read them as C structs.
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// The DLPack device type of host memory, `kDLCPU`.
pub(crate) const CPU: i32 = 1;

/// The DLPack type code of signed integers, `kDLInt`.
const INT: u8 = 0;

/// A consumer that takes the tensor renames the capsule to `used_dltensor`.
const CAPSULE_NAME: &CStr = c"dltensor";

/// A [`DLManagedTensor`] with the memory it points to.
struct Export {
    tensor: DLManagedTensor,
    /// `tensor.dl_tensor.data` points at its heap buffer.
    ids: Vec<i64>,
    /// `tensor.dl_tensor.shape` points at it.
    len: i64,
}

/// Copies `ids` to an `int64` tensor and returns the DLPack capsule that owns it.
pub(crate) fn int64_capsule<'py>(py: Python<'py>, ids: &[u32]) -> PyResult<Bound<'py, PyCapsule>> {
    let export = Box::into_raw(Box::new(Export {
        tensor: DLManagedTensor {
            dl_tensor: DLTensor {
                data: ptr::null_mut(),
                device: DLDevice {
                    device_type: CPU,
                    device_id: 0,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: INT,
                    bits: i64::BITS as u8,
                    lanes: 1,
                },
                shape: ptr::null_mut(),
                // DLPack reads a null `strides` as a contiguous tensor.
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: Some(delete),
        },
        ids: ids.iter().map(|&id| i64::from(id)).collect(),
        len: ids.len() as i64,
    }));
    // SAFETY: `export` comes from `Box::into_raw` just above, so it is valid and nothing else
    // reads it yet. Its address and the heap buffer of `ids` stay fixed until `delete` frees them.
    let tensor = unsafe {
        (*export).tensor.dl_tensor.data = (*export).ids.as_mut_ptr().cast();
        (*export).tensor.dl_tensor.shape = &raw mut (*export).len;
        (*export).tensor.manager_ctx = export.cast();
        &raw mut (*export).tensor
    };
    let pointer = NonNull::new(tensor.cast()).expect("a field of a boxed Export has an address");
    // SAFETY: `pointer` is a live `DLManagedTensor`, and `drop_capsule` or the consumer frees it.
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(py, pointer, CAPSULE_NAME, Some(drop_capsule))
    };
    if capsule.is_err() {
        // SAFETY: no capsule took `tensor`, so this is its only owner.
        unsafe { delete(tensor) };
    }
    capsule
}

/// Frees an [`Export`]. It holds no Python object, so it is safe to call from any thread.
///
/// # Safety
///
/// `tensor` is the tensor of an [`Export`] made by [`int64_capsule`], and this is the first call.
unsafe extern "C" fn delete(tensor: *mut DLManagedTensor) {
    // SAFETY: `manager_ctx` is the `Box<Export>` that `int64_capsule` turned into a raw pointer.
    // The caller guarantees it was not freed.
    drop(unsafe { Box::from_raw((*tensor).manager_ctx.cast::<Export>()) });
}

/// Frees the tensor of a capsule that no consumer took.
unsafe extern "C" fn drop_capsule(capsule: *mut ffi::PyObject) {
    // SAFETY: a capsule still named `CAPSULE_NAME` holds the tensor `int64_capsule` put in it.
    // A consumer that took the tensor renamed the capsule, and calls `delete` itself.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, CAPSULE_NAME.as_ptr()) != 0 {
            delete(ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr()).cast());
        }
    }
}
