//! Import Arrow string buffers without creating Python string/scalar objects.

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi_and_data_type};
use arrow_array::{Array, BinaryArray, GenericBinaryArray, LargeBinaryArray, OffsetSizeTrait};
use arrow_schema::DataType;
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section2;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

pub(crate) enum NullHandling {
    Error,
    Empty,
    Skip,
}

impl TryFrom<&str> for NullHandling {
    type Error = PyErr;

    fn try_from(value: &str) -> PyResult<Self> {
        match value {
            "error" => Ok(Self::Error),
            "empty" => Ok(Self::Empty),
            "skip" => Ok(Self::Skip),
            _ => Err(PyValueError::new_err(format!(
                "null_handling must be 'error', 'empty', or 'skip', got {value:?}"
            ))),
        }
    }
}

/// Owns the imported buffers, including the producer's release callback.
pub(crate) enum ArrowStrings {
    Empty,
    Utf8(BinaryArray),
    LargeUtf8(LargeBinaryArray),
}

impl ArrowStrings {
    pub(crate) fn from_python(input: &Bound<'_, PyAny>) -> PyResult<Self> {
        let export = input.getattr("__arrow_c_array__").map_err(|err| {
            if err.is_instance_of::<PyAttributeError>(input.py()) {
                PyTypeError::new_err("input must implement __arrow_c_array__")
            } else {
                err
            }
        })?;
        let (schema_capsule, array_capsule) = export
            .call0()?
            .extract::<(Bound<'_, PyCapsule>, Bound<'_, PyCapsule>)>()?;

        // Serialize consumption of the same capsules on free-threaded Python too.
        // No Python callbacks or detachment occur between the release checks and moves.
        let (schema, array) = with_critical_section2(
            schema_capsule.as_any(),
            array_capsule.as_any(),
            || -> PyResult<_> {
                let schema_ptr = schema_capsule
                    .pointer_checked(Some(c"arrow_schema"))?
                    .cast::<FFI_ArrowSchema>()
                    .as_ptr();
                let array_ptr = array_capsule
                    .pointer_checked(Some(c"arrow_array"))?
                    .cast::<FFI_ArrowArray>()
                    .as_ptr();
                if !schema_ptr.is_aligned() || !array_ptr.is_aligned() {
                    return Err(PyValueError::new_err("unaligned Arrow capsule pointer"));
                }
                // SAFETY: Named Arrow capsules must contain valid C Data Interface
                // structs. The capsules remain alive and are locked during the move.
                // from_raw clears their release callbacks, transferring ownership
                // exactly once; their destructors still free the outer C structs.
                unsafe {
                    if (*schema_ptr).release.is_none() || (*array_ptr).is_released() {
                        return Err(PyValueError::new_err(
                            "Arrow capsules have already been consumed",
                        ));
                    }
                    Ok((
                        FFI_ArrowSchema::from_raw(schema_ptr),
                        FFI_ArrowArray::from_raw(array_ptr),
                    ))
                }
            },
        )?;

        let data_type = DataType::try_from(&schema)
            .map_err(|err| PyValueError::new_err(format!("invalid Arrow schema: {err}")))?;
        if !matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
            return Err(PyTypeError::new_err(format!(
                "expected an Arrow string or large_string array, got {data_type}"
            )));
        }
        // Empty slices can retain a nonzero string offset. There are no values
        // to read, and the FFI importer need not infer a data-buffer length from
        // that offset (it treats empty arrays as having zero data bytes).
        if array.is_empty() {
            return Ok(Self::Empty);
        }

        if array.num_buffers() != 3 || array.buffers.is_null() {
            return Err(PyValueError::new_err(
                "an Arrow string array must have three buffers",
            ));
        }
        let offsets = array.buffer(1);
        if offsets.is_null() {
            return Err(PyValueError::new_err("missing Arrow string offsets buffer"));
        }
        // arrow-rs 59 reads the final offset through a typed pointer before its
        // alignment repair step. Reject unaligned offsets rather than invoking
        // undefined behavior on a valid but unsupported C Data buffer layout.
        let alignment = match data_type {
            DataType::Utf8 => std::mem::align_of::<i32>(),
            DataType::LargeUtf8 => std::mem::align_of::<i64>(),
            _ => unreachable!(),
        };
        if offsets.align_offset(alignment) != 0 {
            return Err(PyValueError::new_err(format!(
                "Arrow string offsets must be aligned to {alignment} bytes on this platform"
            )));
        }

        // String and binary arrays have identical buffer layouts. Import as binary
        // so validation checks offsets without requiring UTF-8 in null slots, whose
        // payload is unspecified. Non-null text is validated when borrowed below.
        let storage_type = match data_type {
            DataType::Utf8 => DataType::Binary,
            DataType::LargeUtf8 => DataType::LargeBinary,
            _ => unreachable!(),
        };
        // SAFETY: The producer supplies valid, immutable Arrow C Data Interface
        // buffers. arrow-rs imports them with shared ownership of the release callback.
        // As with other C Data consumers, allocation sizes/pointer validity are the
        // producer's responsibility; the interface does not expose buffer capacities.
        let data = unsafe { from_ffi_and_data_type(array, storage_type) }
            .map_err(|err| PyValueError::new_err(format!("invalid Arrow array: {err}")))?;
        // Check offsets and validity before constructing a typed binary array.
        data.validate_full()
            .map_err(|err| PyValueError::new_err(format!("invalid Arrow array: {err}")))?;
        match data.data_type() {
            DataType::Binary => Ok(Self::Utf8(BinaryArray::from(data))),
            DataType::LargeBinary => Ok(Self::LargeUtf8(LargeBinaryArray::from(data))),
            _ => unreachable!("the imported schema was checked above"),
        }
    }

    pub(crate) fn texts(&self, null_handling: NullHandling) -> PyResult<Vec<&str>> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::Utf8(array) => collect_texts(array, null_handling),
            Self::LargeUtf8(array) => collect_texts(array, null_handling),
        }
    }
}

fn collect_texts<O: OffsetSizeTrait>(
    array: &GenericBinaryArray<O>,
    null_handling: NullHandling,
) -> PyResult<Vec<&str>> {
    let mut texts = Vec::with_capacity(array.len());
    for (index, text) in array.iter().enumerate() {
        match text {
            Some(bytes) => {
                let text = std::str::from_utf8(bytes).map_err(|err| {
                    PyValueError::new_err(format!(
                        "invalid Arrow array: invalid UTF-8 at index {index}: {err}"
                    ))
                })?;
                texts.push(text);
            }
            None => match null_handling {
                NullHandling::Error => {
                    return Err(PyValueError::new_err(format!(
                        "Arrow input contains a null at index {index}"
                    )));
                }
                NullHandling::Empty => texts.push(""),
                NullHandling::Skip => {}
            },
        }
    }
    Ok(texts)
}
