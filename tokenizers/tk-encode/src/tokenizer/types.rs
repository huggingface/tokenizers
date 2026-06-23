#[derive(Clone, Hash, Default, Copy)]
pub struct Bucket {
    pub prefix: [u8; 4],
    pub prefix_len: u8,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone)]
pub struct AddedTokenFlags {
    pub special: bool,
    pub normalized: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

#[derive(Clone)]
pub struct TokenId(pub u32);

#[derive(Clone)]
pub struct TokenMetadata {
    pub data_offset: u32,
    pub len: u8,
    pub id: TokenId,
    pub flags: AddedTokenFlags,
}
