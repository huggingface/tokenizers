use crate::utils::{RustOwned, Slice, free_ptr};

pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Option<Vec<u8>>,
}
impl RustOwned for Encoding {}

/// Frees `encoding` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `encoding` must be NULL, or point to a writable `TkEncoding *` holding NULL or an encoding
/// that is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_free(encoding: *mut *mut Encoding) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(encoding) }
}

/// Returns `encoding`'s token ids, or the empty slice if `encoding` is NULL. The slice points
/// into `encoding` and is valid until `tk_encoding_free`.
///
/// # Safety
/// `encoding` must be NULL, or an encoding that is not freed while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_ids(encoding: Option<&Encoding>) -> Slice<u32> {
    encoding.map_or(Slice::empty(), |encoding| Slice::from(&encoding.ids[..]))
}

/// Returns `encoding`'s per-token type ids, or the empty slice if the tokenizer doesn't produce
/// them or `encoding` is NULL. The slice points into `encoding` and is valid until
/// `tk_encoding_free`.
///
/// # Safety
/// `encoding` must be NULL, or an encoding that is not freed while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_type_ids(encoding: Option<&Encoding>) -> Slice<u8> {
    encoding.map_or(Slice::empty(), |encoding| {
        Slice::from(encoding.type_ids.as_deref().unwrap_or(&[]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding() -> Encoding {
        Encoding {
            ids: vec![1, 2, 3],
            type_ids: Some(vec![0, 0, 1]),
        }
    }

    #[test]
    fn ids_borrow_the_encoding() {
        let encoding = encoding();
        let ids = unsafe { tk_encoding_ids(Some(&encoding)) };
        assert_eq!(unsafe { ids.as_slice() }.unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn type_ids_borrow_the_encoding() {
        let encoding = encoding();
        let type_ids = unsafe { tk_encoding_type_ids(Some(&encoding)) };
        assert_eq!(unsafe { type_ids.as_slice() }.unwrap(), &[0, 0, 1]);
    }

    #[test]
    fn type_ids_are_empty_when_the_tokenizer_produces_none() {
        let encoding = Encoding {
            ids: vec![1],
            type_ids: None,
        };
        let type_ids = unsafe { tk_encoding_type_ids(Some(&encoding)) };
        assert!(unsafe { type_ids.as_slice() }.unwrap().is_empty());
    }

    #[test]
    fn reads_on_null_return_the_empty_slice() {
        let ids = unsafe { tk_encoding_ids(None) };
        assert!(ids.ptr.is_null());
        assert_eq!(ids.len, 0);
        let type_ids = unsafe { tk_encoding_type_ids(None) };
        assert!(type_ids.ptr.is_null());
        assert_eq!(type_ids.len, 0);
    }
}
