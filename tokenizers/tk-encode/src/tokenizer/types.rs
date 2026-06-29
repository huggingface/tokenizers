use crate::vocab_store::VocabStore;

#[derive(Clone, PartialEq, Debug)]
pub struct AddedTokenFlags {
    pub special: bool,
    pub normalized: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// We implement a 2-level check:
/// 1. on the next byte after the prefix.
/// 2. on the lenght of the added tokens.
/// 3. on the hashed bytes.
#[derive(Clone, Debug)]
pub struct Bucket {
    // longest common prefix of the bucket
    pub prefix: Box<[u8]>,
    pub next_byte_to_length_id: [u16; 256], // post-prefix byte -> sub-list index; 0xFFFF = none
    pub length_list: Box<[Box<[u16]>]>,
}

/// Buckets are responsible of finding matches.
/// lo / hi nibble
pub struct Buckets {
    first_byte_to_bucket_id: [u8; 256],
    // we use optimized SIMD to scan potential special tokens when there are more than 1 bucket.
    // this is taken from http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html
    lo16: [u8; 16],
    hi16: [u8; 16],
    can_use_nibbling: bool, // we can't if there are >8 high nibbles (very rare)
    buckets: Box<[Bucket]>,
    inner: VocabStore,
}

impl Buckets {
    pub fn new() -> Self {
        Self {
            first_byte_to_bucket_id: [0; 256],
            lo16: [0; 16],
            hi16: [0; 16],
            can_use_nibbling: true,
            buckets: Box::default(),
            inner: VocabStore::new(),
        }
    }
    //   ┌───────────────────────────────╴
    //   │ 0 1 2 3 4 5 6 7 8 9 a b c d e f
    // ╶─┼───────────────────────────────╴
    // 0 │ . . . . . . . . . . . . . . . .
    // 1 │ . . . . . . . . . . . . . . . .
    // 2 │ . . . . . . . . . . . . . . . .
    // 3 │ . . . . . . . . . . . . . . . .
    // 4 │ . . . . . . . . . . . . . . . .
    // 5 │ . . . . . . . . . . . . . . . .
    // 6 │ . . . . . . . . . . . . . . . .
    // 7 │ . . . . . . . . . . . . . . . .
    // 8 │ . . . . . . . . . . . . . . . .
    // 9 │ x . . . . . . . . . . . . . . .
    // a │ x . . . . . . . . . . . . . . .
    // b │ . . . . . x . . . . . . . . . .
    // c │ . . . x . . . . . . . . . . . .
    // d │ . . . . . . . . . . . . . . . .
    // e │ . . . . . . . . . . . . . . . .
    // f ╵ . . . . . . . . . . . . . . . .
    // If you have the following first bytes
    // "<"  = 0x3c  -> col 3, row c
    // "["  = 0x5b  -> col 5, row b
    // "\t" = 0x09  -> col 0, row 9
    // "\n" = 0x0a  -> col 0, row a
    // The reason we have lo and hi is because for SIMD
    // we can't have the index be 16 bits, so lo has 8bits length, hi as well.
    fn build_nibble_table(&mut self) {
        // we build the nibble table from the candidate first_byte_to_bucket_id
        let candidates: [bool; 256] = self.first_byte_to_bucket_id & true;
        // for each valid candidate (a u8) we build the 16x16 lookup tabllet (mut next, mut bit, mut has) = (0u32, [0u8;16], [false;16]);
        let (mut lo, mut hi) = ([0u8; 16], [0u8; 16]);
        let (mut next, mut bit, mut has) = (0u32, [0u8; 16], [false; 16]);
        for h in 0..16 {
            // h<<4 | l is just computing 0x{h}{l} indexed into the candidates
            // here for each match, we update the count and fill the table with the
            // unique id associated with that match.
            if (0..16).any(|l| candidates[(h << 4) | l]) {
                if next >= 8 {
                    return None;
                } // >8 high nibbles -> caller uses scalar TODO:
                let counter = 1u8 << next;
                next += 1;
                hi[h] = counter;
                bit[h] = counter; // will be used to fill lo16 next
                has[h] = true;
            }
        }
        for h in 0..16 {
            if has[h] {
                // fast rejections
                for l in 0..16 {
                    // we use | because we don't want to erase the rest of the column
                    if candidates[(h << 4) | l] {
                        lo[l] |= bit[h];
                    }
                }
            }
        }
        (self.lo16, self.hi16) = (lo, hi);
    }
    #[cfg(target_arch = "aarch64")]
    fn nibble_match_bytes(&self, bytes: &[u8]) -> Option<u32> {
        // lane index   0     1     2     3     4    ...  (16 lanes, one register)
        // input byte  'a'   'b'   '<'   'd'   'e'   ...
        // hex         61    62    3C    64    65    ...
        //
        // 1) low_nibbles  = v & 0x0F   →   1     2     C     4     5   ...
        // 2) high_nibbles = v >> 4     →   6     6     3     6     6   ...
        //
        // 3) lo_hits = lo_table[low]   →  lo[1] lo[2] lo[C] lo[4] ...  = 0   0  0b01  0  ...
        // 4) hi_hits = hi_table[high]  →  hi[6] hi[6] hi[3] hi[6] ...  = 0   0  0b01  0  ...
        //
        // 5) matched = lo_hits & hi_hits → 0    0   0b01   0   ...   ← lane 2 nonzero = '<' ∈ S
        // 6) first nonzero lane = 2  →  candidate at offset+2   (or "all zero → advance 16")
        //
        // The key is that instead of "for each of the 16 bytes: load it, index a 256-table, branch,"
        // you do "classify all 16, branchless, from registers".
        //
        // if there are more than 1 bucket, we would waste time checking first_byte_to_bucket_id
        // first. So we start with optimized SIMD 16-byte checks, then if a byte matches, then
        // we get the bucket id via first_byte_to_bucket_id.
        use core::arch::aarch64::*;
        let (n, mut i) = (bytes.len(), 0);
        // SAFETY: NEON is baseline on aarch64; the only memory op is the 16-byte load below,
        // kept in bounds by the `i + 16 <= n` guard.
        unsafe {
            let lov = vld1q_u8(self.lo16.as_ptr()); // load a 16bytes array in registers
            let hiv = vld1q_u8(self.hi16.as_ptr());
            let m0f = vdupq_n_u8(0x0f); // prepare 0 mask used to get low nibbles
            while i + 16 <= n {
                let v = vld1q_u8(bytes.as_ptr().add(i)); // in bounds: i + 16 <= n
                let lon = vandq_u8(v, m0f); // 1) low nibbles
                let hin = vshrq_n_u8(v, 4); // 2) high nibbles
                let lm = vqtbl1q_u8(lov, lon); // 3) lo_tbl[low]
                let hm = vqtbl1q_u8(hiv, hin); // 4) hi_tbl[high]
                let m = vandq_u8(lm, hm); // nonzero lane == match
                                          // Here we need to get the index of the match, that's gonna hardware dependant.
                                          // NEON has no PMOVMSKB: emulate movemask, 4 bits per lane, via shrn-by-4
                let t = vreinterpretq_u16_u8(vtstq_u8(m, m));
                let packed = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(t, 4)), 0);
                if packed != 0 {
                    return Some(i + (packed.trailing_zeros() as usize >> 2));
                }
                i += 16;
            }
        } // end unsafe (NEON region)
        while i < n {
            if self.first_byte_to_bucket_id[bytes[i] as usize] != 0xFF {
                return Some(i);
            }
            i += 1;
        } // scalar tail
        return None;
    }

    fn longest_first_match(&self, bytes: &[u8], bucket_id: u32) -> Option<(u32, u32)> {
        // 1. common prefix check
        if !bytes.starts_with(&self.buckets[bucket_id as usize].prefix) {
            return None;
        }
        // 2. byte AFTER the common prefix -> which length sub-list (0xFFFF = none)
        let bucket = &self.buckets[bucket_id as usize];
        let disc = match bytes.get(bucket.prefix.len()) {
            Some(&b) => b,
            None => return None, // input ends at the prefix (a token == prefix would need an `exact` flag)
        };
        let length_id = bucket.next_byte_to_length_id[disc as usize];
        if length_id == 0xFFFF {
            return None;
        }
        // 3. probe each candidate length, longest-first (first hit = longest match)
        for &len in bucket.length_list[length_id as usize].iter() {
            let len = len as usize;
            if len <= bytes.len() {
                if let Some(id) = self.inner.get_bytes(&bytes[..len]) {
                    return Some((id, len));
                }
            }
        }
        None
    }
    /// returns token_id, match_position, match_len
    pub fn match_bytes(&self, bytes: &[u8]) -> Option<(u32, u32, u32)> {
        let mut best: Option<(u32, u32, u32)> = None;
        let search = 0;
        // return the end of match index and the id of the match token if any.
        // 1. quick scan of the bytes with fast rejection
        let (bucket, cutoff) = match self.buckets.len() {
            // single bucket, fast memchr scan on the first byte of the common prefix
            1 => match memchr::memchr(self.buckets[0].prefix[0], &bytes) {
                Some(id) => (0, id),
                None => return best,
            },
            2 => {
                match memchr::memchr2(self.buckets[0].prefix[0], self.buckets[1].prefix[0], &bytes)
                {
                    Some(id) => (self.first_byte_to_bucket_id[bytes[id as usize]], id),
                    None => return best,
                }
            }
            // memchr has optimized path for 2 and 3 prefix
            3 => {
                match memchr::memchr3(
                    self.buckets[0].prefix[0],
                    self.buckets[1].prefix[0],
                    self.buckets[2].prefix[0],
                    &bytes,
                ) {
                    Some(id) => (self.first_byte_to_bucket_id[bytes[id as usize]], id),
                    None => return best,
                }
            }

            _ => match self.nibble_match_bytes(bytes) {
                Some((id, cutoff)) => {
                    (id, cutoff);
                }
                None => return best,
            },
        };
        let (token_id, len) = self.longest_first_match(&bytes[cutoff..], bucket);
        (token_id, cutoff, len)
    }
}
