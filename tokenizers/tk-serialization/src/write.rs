//! Writing a `.tok`. Feature-gated behind `write`: an inference binary only ever reads one.

use core::mem::size_of;

use crate::{Header, MAGIC, SECTION_ALIGN, Section, VERSION};

// The reader casts file bytes straight to these types, so their layout *is* the format. Pin it,
// and pin that none of them has implicit padding — `as_bytes` reads every byte of one.
const _: () = assert!(size_of::<Header>() == 16);
const _: () = assert!(size_of::<Section>() == 16);
const _: () = assert!(size_of::<crate::Config>() == 48);
const _: () = assert!(size_of::<crate::Entry>() == 12);
const _: () = assert!(size_of::<crate::AddedEntry>() == 16);

/// Lays sections out back to back at [`SECTION_ALIGN`], patching the header and table in at the
/// end. Sections are written in whatever order you push them and sorted by kind on `finish`.
#[derive(Default)]
pub struct Writer {
    payload: Vec<u8>,
    table: Vec<Section>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a section holding `data`. Skips empty sections — an absent section reads back as an
    /// empty slice, so writing one would only cost a descriptor.
    pub fn push<T: Copy>(&mut self, kind: u32, data: &[T]) {
        if data.is_empty() {
            return;
        }
        self.push_bytes(kind, as_bytes(data));
    }

    /// Append a section holding exactly one `T`.
    pub fn push_one<T: Copy>(&mut self, kind: u32, value: &T) {
        self.push_bytes(kind, as_bytes(core::slice::from_ref(value)));
    }

    fn push_bytes(&mut self, kind: u32, data: &[u8]) {
        // Offsets are relative to the file start, which is not known until the table is sized, so
        // record the payload-relative offset and shift every descriptor once in `finish`.
        let offset = self.payload.len();
        self.payload.extend_from_slice(data);
        self.payload
            .resize(self.payload.len().next_multiple_of(SECTION_ALIGN), 0);
        self.table.push(Section {
            kind,
            offset: offset as u32,
            len: data.len() as u32,
            _pad: 0,
        });
    }

    /// Serialise. The returned bytes are a complete `.tok` file.
    pub fn finish(mut self) -> Vec<u8> {
        let header_end = size_of::<Header>() + self.table.len() * size_of::<Section>();
        let base = header_end.next_multiple_of(SECTION_ALIGN);

        self.table.sort_unstable_by_key(|s| s.kind);
        for section in &mut self.table {
            section.offset += base as u32;
        }

        let mut out = vec![0u8; base];
        out.extend_from_slice(&self.payload);

        let header = Header {
            magic: MAGIC,
            n_sections: self.table.len() as u16,
            version: VERSION,
            file_len: out.len() as u32,
            _reserved: 0,
        };
        out[..size_of::<Header>()].copy_from_slice(as_bytes(core::slice::from_ref(&header)));
        out[size_of::<Header>()..header_end].copy_from_slice(as_bytes(&self.table));
        out
    }
}

/// Every section element is `#[repr(C)]` over plain integers with no padding (asserted above), so
/// its byte image is exactly what the reader casts back.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a `#[repr(C)]` integer aggregate with no padding bytes, so every byte of the
    // slice is initialised and readable as `u8`.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
}
