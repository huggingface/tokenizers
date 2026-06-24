#[derive(Clone, Hash, Default, Copy, Debug)]
pub struct Bucket {
    pub prefix: [u8; 4],
    pub prefix_len: u8,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct AddedTokenFlags {
    pub special: bool,
    pub normalized: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub struct TokenId(pub u32);

#[derive(Clone, PartialEq, Debug)]
pub struct TokenMetadata {
    pub data_offset: u32,
    pub len: u8,
    pub id: TokenId,
    pub flags: AddedTokenFlags,
}
