use std::{
    ffi::{CStr, c_char},
    str::Utf8Error,
};

use crate::error::{Error, catch_panic};

/// Marks a type whose instances Rust heap-allocates and frees, and hands to C as an opaque
/// pointer.
///
/// C may use one pointer from several threads at once. The `tk_*` fns that read it borrow it as
/// `&T`, which is sound because `T: Sync`. The `tk_*_set_*` fns borrow it as `&mut T`, which is
/// sound only because their contract makes the caller the sole user for the call's duration.
pub(crate) trait RustOwned: Send + Sync {}

/// Moves `value` to the heap and returns the pointer C will hold, pairing with [`free_ptr`].
pub(crate) fn new_ptr<T: RustOwned>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// Frees the pointer in `slot` and writes NULL to it, so calling this again on the same slot is
/// a no-op instead of a double free.
///
/// # Safety
/// `slot` must be NULL, or point to a writable `*mut T` holding NULL or a pointer from
/// [`new_ptr`] that is not yet freed and that nothing else is using.
pub(crate) unsafe fn free_ptr<T: RustOwned>(slot: *mut *mut T) {
    // SAFETY: caller's obligation, documented above.
    let Some(slot) = (unsafe { slot.as_mut() }) else {
        return;
    };
    if slot.is_null() {
        return;
    }
    // SAFETY: just checked non-NULL; liveness is the caller's obligation, documented above.
    drop(unsafe { Box::from_raw(*slot) });
    *slot = std::ptr::null_mut();
}

/// A wrapper for FFI functions that output a pointer to a new Rust-owned value.
///
/// Takes care of:
/// - reporting a [`Error`] instead of writing anything, if `out` is NULL
/// - initializing the out pointer to NULL
/// - catching panic unwinds so they don't reach C code
/// - moving the output of `inner` to the heap and writing its pointer to `out`, when it succeeds
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `*mut T`, initialized or not.
pub(crate) unsafe fn wrap_in_ptr<T: RustOwned, E: std::fmt::Display>(
    out: *mut *mut T,
    inner: impl FnOnce() -> Result<T, E>,
) -> *mut Error {
    if out.is_null() {
        return Error::into_ptr("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { out.write(std::ptr::null_mut()) };
    catch_panic(move || -> Result<(), E> {
        let value = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { out.write(new_ptr(value)) };
        Ok(())
    })
}

/// A borrowed view of `len` contiguous `T`s: a pointer and a length. It owns nothing, in either
/// direction: C reads a `tk_*` result through one, and hands `tk_tokenizer_decode` its ids in
/// one. A NULL `ptr` with `len` 0 is the empty slice.
#[repr(C)]
pub(crate) struct Slice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> Slice<T> {
    pub(crate) fn new(ptr: *const T, len: usize) -> Self {
        Self { ptr, len }
    }

    pub(crate) fn null() -> Self {
        Self::new(std::ptr::null(), 0)
    }

    /// Borrows this slice's data as a `&[T]`, without copying. A NULL `ptr` is the empty slice
    /// when `len` is 0, and an error otherwise.
    ///
    /// # Safety
    /// `ptr` must be NULL, or point to `len` readable `T`s that are not modified for as long as
    /// the returned slice is used.
    pub(crate) unsafe fn as_slice<'a>(&self) -> Result<&'a [T], &'static str> {
        if self.ptr.is_null() {
            return if self.len == 0 {
                Ok(&[])
            } else {
                Err("slice has a NULL ptr but a non-zero len")
            };
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }
}

impl<T> From<&[T]> for Slice<T> {
    fn from(value: &[T]) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// A wrapper for FFI functions that output a [`Slice`].
///
/// Takes care of:
/// - reporting a [`Error`] instead of writing anything, if `out` is NULL
/// - initializing the out pointer to an empty slice
/// - catching panic unwinds so they don't reach C code
/// - writing `inner`'s slice to the out pointer, when it succeeds
///
/// A missing value (e.g. an encoding with no type ids) is a `body` that returns `Ok(&[])`: a
/// [`Slice`] can't tell "absent" from "empty" apart, and nothing needs it to.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a [`Slice<T>`], initialized or not.
pub(crate) unsafe fn wrap_in_slice<'a, T: 'a, E: std::fmt::Display>(
    out: *mut Slice<T>,
    inner: impl FnOnce() -> Result<&'a [T], E>,
) -> *mut Error {
    if out.is_null() {
        return Error::into_ptr("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { out.write(Slice::null()) };
    catch_panic(move || -> Result<(), E> {
        let slice = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { out.write(Slice::from(slice)) };
        Ok(())
    })
}

/// Borrows a NUL-terminated C string as a UTF-8 `&str`, without copying.
///
/// # Safety
/// `c_str` must be non-NULL and point to a NUL-terminated string that is neither freed nor
/// modified for as long as the returned `&str` is used.
pub(crate) unsafe fn convert_c_str<'a>(c_str: *const c_char) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let c_str = unsafe { CStr::from_ptr(c_str) };
    c_str.to_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Dummy;
    impl RustOwned for Dummy {}

    #[test]
    fn slice_as_slice_borrows_the_underlying_data() {
        let data = [1u32, 2, 3];
        let slice = Slice::from(&data[..]);
        assert_eq!(unsafe { slice.as_slice() }.unwrap(), &data);
    }

    #[test]
    fn slice_as_slice_null_ptr_with_zero_len_is_empty() {
        let slice = Slice::<u32>::new(std::ptr::null(), 0);
        assert_eq!(unsafe { slice.as_slice() }.unwrap(), &[] as &[u32]);
    }

    #[test]
    fn slice_as_slice_null_ptr_with_nonzero_len_is_an_error() {
        let slice = Slice::<u32>::new(std::ptr::null(), 5);
        assert!(unsafe { slice.as_slice() }.is_err());
    }

    #[test]
    fn convert_c_str_accepts_valid_utf8() {
        let bytes = b"hello\0";
        let result = unsafe { convert_c_str(bytes.as_ptr() as *const c_char) };
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn convert_c_str_rejects_invalid_utf8() {
        let bytes = b"\xff\xfe\0";
        let result = unsafe { convert_c_str(bytes.as_ptr() as *const c_char) };
        assert!(result.is_err());
    }

    #[test]
    fn free_ptr_outer_null_is_a_no_op() {
        // Doesn't crash -- that's the whole test.
        unsafe { free_ptr::<Dummy>(std::ptr::null_mut()) };
    }

    #[test]
    fn free_ptr_frees_and_nulls_the_slot() {
        let mut slot = new_ptr(Dummy);
        assert!(!slot.is_null());
        unsafe { free_ptr(&mut slot) };
        assert!(slot.is_null());
    }

    #[test]
    fn free_ptr_is_a_no_op_the_second_time_on_the_same_slot() {
        let mut slot = new_ptr(Dummy);
        unsafe { free_ptr(&mut slot) };
        unsafe { free_ptr(&mut slot) }; // must not double-free
        assert!(slot.is_null());
    }

    #[test]
    fn wrap_in_ptr_out_null_reports_an_error_without_calling_inner() {
        let called = Cell::new(false);
        let result = unsafe {
            wrap_in_ptr::<Dummy, &str>(std::ptr::null_mut(), || {
                called.set(true);
                Ok(Dummy)
            })
        };
        assert!(!result.is_null());
        assert!(!called.get(), "inner must not run when out is NULL");
    }

    #[test]
    fn wrap_in_ptr_resets_out_to_null_even_when_inner_fails() {
        let mut out: *mut Dummy = std::ptr::dangling_mut(); // poison: some non-null value
        let result = unsafe { wrap_in_ptr(&mut out, || -> Result<Dummy, &str> { Err("nope") }) };
        assert!(!result.is_null());
        assert!(out.is_null());
    }

    #[test]
    fn wrap_in_slice_out_null_reports_an_error_without_calling_inner() {
        let called = Cell::new(false);
        let result = unsafe {
            wrap_in_slice::<u32, &str>(std::ptr::null_mut(), || {
                called.set(true);
                Ok(&[][..])
            })
        };
        assert!(!result.is_null());
        assert!(!called.get(), "inner must not run when out is NULL");
    }

    #[test]
    fn wrap_in_slice_resets_out_to_empty_even_when_inner_fails() {
        let mut out = Slice::<u32> {
            ptr: std::ptr::dangling(),
            len: 99,
        }; // poison
        let result = unsafe { wrap_in_slice(&mut out, || -> Result<&[u32], &str> { Err("nope") }) };
        assert!(!result.is_null());
        assert!(out.ptr.is_null());
        assert_eq!(out.len, 0);
    }

    #[test]
    fn wrap_in_slice_writes_the_slice_on_success() {
        let data = [1u32, 2, 3];
        let mut out = Slice::<u32>::null();
        let result =
            unsafe { wrap_in_slice(&mut out, || -> Result<&[u32], &str> { Ok(&data[..]) }) };
        assert!(result.is_null());
        assert_eq!(
            unsafe { std::slice::from_raw_parts(out.ptr, out.len) },
            &data
        );
    }
}
