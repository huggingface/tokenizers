use crate::Offsets;

struct VocabStore {
    blob: Vec<u8>,
    offsets: Offsets,
    byte_to_id: Vec<u32>,
}
