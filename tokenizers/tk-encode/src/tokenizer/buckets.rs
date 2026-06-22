#[derive(Clone, Hash, Default, Copy)]
pub struct Bucket {
    pub prefix: [u8; 4],
    pub prefix_len: u8,
    pub start: u32,
    pub end: u32,
}
