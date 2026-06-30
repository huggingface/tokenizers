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
#[derive(Clone)]
pub struct Buckets {
    first_byte_to_bucket_id: [u8; 256],
    // we use optimized SIMD to scan potential special tokens when there are more than 1 bucket.
    // this is taken from http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html
    lo16: [u8; 16],
    hi16: [u8; 16],
    #[allow(dead_code)] // TODO: honor in the scan — fall back to scalar when false
    can_use_nibbling: bool, // we can't if there are >8 high nibbles (very rare)
    buckets: Box<[Bucket]>,
    inner: VocabStore,
}

impl Buckets {
    /// Empty matcher (no added tokens yet).
    pub fn new() -> Self {
        Self {
            first_byte_to_bucket_id: [0xFF; 256], // 0xFF = no bucket for this first byte
            lo16: [0; 16],
            hi16: [0; 16],
            can_use_nibbling: true,
            buckets: Box::default(),
            inner: VocabStore::new(),
        }
    }
    /// Build the matcher from sorted `tokens`, the first-byte->bucket table, and the buckets.
    pub fn build(
        tokens: Vec<(Vec<u8>, u32)>,
        first_byte_to_bucket_id: [u8; 256],
        buckets: Box<[Bucket]>,
    ) -> Self {
        // tokens should be sorted
        let inner = VocabStore::build(tokens);
        let mut new = Self {
            first_byte_to_bucket_id: first_byte_to_bucket_id,
            lo16: [0; 16],
            hi16: [0; 16],
            can_use_nibbling: true,
            buckets: buckets,
            inner: inner,
        };
        new.build_nibble_table();
        new
    }
    ///  - group token indices by first byte,
    ///  - per group: longest-common-prefix capped at `shortest_len - 1` (so every token has a
    ///    discriminating byte and the "token == prefix" case needs no special handling),
    ///  - build the post-prefix disc table + per-disc distinct lengths (longest first),
    /// then `build()` does the VocabStore (hash pass) + nibble table.
    pub fn from_tokens(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        if tokens.is_empty() {
            return Self::new();
        }
        // group token indices by first byte
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); 256];
        for (i, (bytes, _)) in tokens.iter().enumerate() {
            if let Some(&first_b) = bytes.first() {
                groups[first_b as usize].push(i as u32);
            }
        }
        let mut first_byte_to_bucket_id = [0xFFu8; 256];
        let mut buckets: Vec<Bucket> = Vec::new();
        for first_b in 0..256 {
            if groups[first_b].is_empty() {
                continue;
            }
            // longest common prefix of the group + shortest token length
            let mut lcp: &[u8] = &tokens[groups[first_b][0] as usize].0;
            let mut min_len = lcp.len();
            for &i in &groups[first_b] {
                // for each token index we start with the longest token
                // and we iterate over the lcp's bytes until theirs are
                // no longer equal to the current. At this point we shortest
                // the lcp and go to the next. The tokens are sorted by size and
                // longest prefix.
                let t = &tokens[i as usize].0;
                min_len = min_len.min(t.len());
                let common = lcp.iter().zip(t).take_while(|(a, b)| a == b).count();
                lcp = &lcp[..common];
            }
            // cap so prefix.len() < every token's len => every token has a byte at prefix.len()
            let plen = lcp.len().min(min_len.saturating_sub(1));
            let prefix: Box<[u8]> = lcp[..plen].to_vec().into_boxed_slice();
            // disc table (byte after prefix -> sub-list) + per-disc distinct lengths, longest first
            let mut next_byte_to_length_id = [0xFFFFu16; 256];
            let mut length_list: Vec<Vec<u16>> = Vec::new();
            for &i in &groups[first_b] {
                let t = &tokens[i as usize].0;
                let disc = t[plen] as usize;
                let li = if next_byte_to_length_id[disc] == 0xFFFF {
                    next_byte_to_length_id[disc] = length_list.len() as u16;
                    length_list.push(Vec::new());
                    length_list.len() - 1
                } else {
                    next_byte_to_length_id[disc] as usize
                };
                let l = t.len() as u16;
                if !length_list[li].contains(&l) {
                    length_list[li].push(l);
                }
            }
            for list in length_list.iter_mut() {
                list.sort_unstable_by(|a, b| b.cmp(a)); // longest first
            }
            debug_assert!(buckets.len() < 0xFF, "more than 254 distinct first bytes");
            first_byte_to_bucket_id[first_b] = buckets.len() as u8;
            buckets.push(Bucket {
                prefix,
                next_byte_to_length_id,
                length_list: length_list.into_iter().map(Vec::into_boxed_slice).collect(),
            });
        }
        Self::build(tokens, first_byte_to_bucket_id, buckets.into_boxed_slice())
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
    // The key is also that we don't store the 16x16 grid, just 2x16 u8 (since u8 is up to 8 values)
    // So the lo[3] = 0b00100100 ->
    //                  ..x..x..
    // store the entire row in the u8 format. (0-255)
    fn build_nibble_table(&mut self) {
        // we build the nibble table from the candidate first_byte_to_bucket_id
        let candidates: [bool; 256] = self.first_byte_to_bucket_id.map(|v| v != 0xFF);
        let (mut lo, mut hi) = ([0u8; 16], [0u8; 16]);
        let (mut next, mut bit, mut has) = (0u32, [0u8; 16], [false; 16]);
        for h in 0..16 {
            // h<<4 | l is just computing 0x{h}{l} indexed into the candidates
            // here for each match, we update the count and fill the table with the
            // unique id associated with that match.
            if (0..16).any(|l| candidates[(h << 4) | l]) {
                if next >= 8 {
                    return;
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
    fn nibble_match_bytes(&self, bytes: &[u8]) -> Option<usize> {
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
                    return Some((id, len as u32));
                }
            }
        }
        None
    }
    /// returns token_id, match_position, match_len
    pub fn match_bytes(&self, bytes: &[u8]) -> Option<(u32, u32, u32)> {
        // 1. quick scan of the bytes with fast rejection
        let (bucket, cutoff) = match self.buckets.len() {
            // single bucket, fast memchr scan on the first byte of the common prefix
            1 => match memchr::memchr(self.buckets[0].prefix[0], &bytes) {
                Some(id) => (0, id),
                None => return None,
            },
            2 => {
                match memchr::memchr2(self.buckets[0].prefix[0], self.buckets[1].prefix[0], &bytes)
                {
                    Some(id) => (
                        self.first_byte_to_bucket_id[bytes[id as usize] as usize],
                        id,
                    ),
                    None => return None,
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
                    Some(id) => (
                        self.first_byte_to_bucket_id[bytes[id as usize] as usize],
                        id,
                    ),
                    None => return None,
                }
            }

            _ => match self.nibble_match_bytes(bytes) {
                Some(id) => (
                    self.first_byte_to_bucket_id[bytes[id as usize] as usize],
                    id,
                ),
                None => return None,
            },
        };
        if let Some((token_id, len)) = self.longest_first_match(&bytes[cutoff..], bucket as u32) {
            Some((token_id, cutoff as u32, len))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// The bucket metadata (prefix / disc table / length lists), one entry per first-byte group.
    pub fn get_buckets(&self) -> &[Bucket] {
        &self.buckets
    }
    /// first byte -> bucket index (0xFF = none). The builder reads this to extend buckets.
    pub fn first_byte_to_bucket_id(&self) -> &[u8; 256] {
        &self.first_byte_to_bucket_id
    }
    /// All added-token byte strings + ids (e.g. to rebuild the vocab when adding tokens).
    pub fn get_vocab_bytes(&self) -> Vec<(Vec<u8>, u32)> {
        self.inner.get_vocab_bytes()
    }
    /// All added-token strings + ids.
    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.inner.get_vocab()
    }
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

impl Default for Buckets {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_nibble_table() {
        let mut first_byte_to_bucket = [0xFFu8; 256];
        // spell "toad": t=0x74, o=0x6F, a=0x61, d=0x64
        //   TOP    collision: o, a, d all have high nibble 6 (same row -> same hi bit)
        //   BOTTOM collision: t and d share low nibble 4 (lo16[4] ends up with two bits)
        first_byte_to_bucket[b't' as usize] = 0;
        first_byte_to_bucket[b'o' as usize] = 1;
        first_byte_to_bucket[b'a' as usize] = 2;
        first_byte_to_bucket[b'd' as usize] = 3;

        let mut expected_hi16 = [0u8; 16];
        let mut expected_lo16 = [0u8; 16];

        // filling hi goes from 0..16. First match gets the first id.
        expected_hi16[(b'o' >> 4) as usize] = 0b01; // row 6  (o, a, d all share index)
        expected_hi16[(b't' >> 4) as usize] = 0b10; // row 7  (t)

        // each byte ORs in ITS ROW's bit at its low nibble
        expected_lo16[(b'o' & 0x0f) as usize] |= 0b01; // o -> lo16[15]
        expected_lo16[(b'a' & 0x0f) as usize] |= 0b01; // a -> lo16[1]
        expected_lo16[(b'd' & 0x0f) as usize] |= 0b01; // d -> lo16[4]
        expected_lo16[(b't' & 0x0f) as usize] |= 0b10; // t -> lo16[4]  => lo16[4] = 0b11  (bottom collision!)
        let mut fake_bucket = Buckets::build(
            vec![("ha".as_bytes().to_vec(), 0)],
            first_byte_to_bucket,
            Box::new([]),
        );
        fake_bucket.build_nibble_table();
        assert_eq!(fake_bucket.lo16, expected_lo16);
        assert_eq!(fake_bucket.hi16, expected_hi16);
        assert!(fake_bucket.nibble_match_bytes("pardis".as_bytes()) == Some(1));
    }
}
