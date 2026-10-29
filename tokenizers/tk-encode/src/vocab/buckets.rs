use crate::Result;

use super::bucket_vocab_store::BucketVocabStore;

#[derive(Clone, PartialEq, Debug, Default)]
pub struct AddedTokenFlags {
    pub special: bool,
    pub normalized: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// The key to have a fast byte matching alrogithm is to skip failures fast and reject quickly
/// potential candidates. What we do here:
/// 1. memchr / nibble SIMD scan of potential candidates. `pshufb` does parallel lookup in a 16-byte
///    register, using 4-bit indices. `nibbles` are 4 bits
/// 2. on candidate bytes, check the Bucket that corresponds to that byte (1 in 255). If there is a
///    bucket we check if there is in that bucket's next_byte_to_length_id a valid length_list. If
///    not we don't have to check anything
/// 3. We check the prefix interpreted as a u64 (so we take the first 8 bytes). This is a fast &
/// 4. We iterate over the length list, hash the bytes cropped to the length -> longest first match.
#[derive(Clone, Debug)]
pub struct Bucket {
    // longest common prefix of the bucket
    pub prefix: Box<[u8]>,
    // prefix[0..min(len,8)] packed native-endian + a per-byte masking on invalid bytes, so the hot-path reject is a
    // single `(chunk ^ prefix_word) & prefix_mask` instead of walking the boxed prefix slice.
    pub prefix_word: u64,
    pub prefix_mask: u64,
    // post-prefix byte -> sub-list index; 0xFFFF = none. We use u16 because proba of having > 65K
    // different lists is close to 0. We could use u4 safely AFAIK :).
    pub next_byte_to_length_id: [u16; 256],
    // per-post-prefix byte indexed by next_byte_to_length_id. We build this once so Box (per byte)
    // Box (an array of length) and IDK any token are gonna > u16::MAX.
    pub length_list: Box<[Box<[u16]>]>,
}

impl Bucket {
    /// Pack the prefix into `prefix_word`/`prefix_mask` for the fast rejection.
    pub fn new(
        prefix: Box<[u8]>,
        next_byte_to_length_id: [u16; 256],
        length_list: Box<[Box<[u16]>]>,
    ) -> Self {
        // native-endian packing so the reject's `from_ne_bytes` read is a pure reinterpret on any
        // arch (no byte-swap); the equality compare is endian-agnostic as long as both sides match.
        let mut word_buf = [0u8; 8];
        let mut mask_buf = [0u8; 8];
        let plen = prefix.len().min(8);
        word_buf[..plen].copy_from_slice(&prefix[..plen]);
        for b in mask_buf[..plen].iter_mut() {
            *b = 0xFF;
        }
        let prefix_word = u64::from_ne_bytes(word_buf);
        let prefix_mask = u64::from_ne_bytes(mask_buf);
        // we create a mask over the valid bytes as the prefix can be shorter
        // than 8 bytes, interpreting 8 bytes + mask is better than slicing.
        Bucket {
            prefix,
            prefix_word,
            prefix_mask,
            next_byte_to_length_id,
            length_list,
        }
    }
}

/// Buckets store added tokens and the stuff needed to quickly find them in bytes.
/// This is equivalent to a HashMap<token, id> + DoubleArrayAhoCorasick on all token but at a
/// fraction of the memory cost for the same performance in worst cases (dense inputs) and much
/// faster when there are no candidates.
#[derive(Clone)]
pub struct Buckets {
    // is there a bucket for that byte. This is used to build the nibble matcher and extract
    // on match the bucket corresponding to the byte that was scanned.
    first_byte_to_bucket_id: [u8; 256],
    // we use optimized SIMD instructions (main one being `pshufb`) to scan potential special tokens when there are multiple unique starting bytes, noting that we have one bucket per starting byte.
    // The algo is taken from http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html, specifically the "Special case 1 - small sets"
    lo16: [u8; 16],
    hi16: [u8; 16],
    buckets: Box<[Bucket]>,
    // efficient AHashMap equivalent build on the premise that we know in advance all the keys: we
    // are in a close addressing problem.
    vocab: BucketVocabStore,
}

impl Buckets {
    pub fn new() -> Self {
        Self {
            first_byte_to_bucket_id: [0xFF; 256], // 0xFF = no bucket for this first byte
            lo16: [0; 16],
            hi16: [0; 16],
            buckets: Box::default(),
            vocab: BucketVocabStore::new(),
        }
    }
    /// Build the matcher from `tokens`, the first-byte->bucket table, and the buckets.
    pub fn build(
        tokens: Vec<(Vec<u8>, u32)>,
        first_byte_to_bucket_id: [u8; 256],
        buckets: Box<[Bucket]>,
    ) -> Result<Self> {
        let vocab = BucketVocabStore::build(tokens)?;
        let mut new = Self {
            first_byte_to_bucket_id,
            lo16: [0; 16],
            hi16: [0; 16],
            buckets,
            vocab,
        };
        new.build_nibble_table();
        Ok(new)
    }

    /// Used by the AddedVocabulary when a new token is added, we recreate the entire structure.
    pub fn from_tokens(tokens: Vec<(Vec<u8>, u32)>) -> Result<Self> {
        if tokens.is_empty() {
            return Ok(Self::new());
        }
        // First we group tokens that have the same starting byte.
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
            // longest common prefix of the group + compute token length_list
            let mut lcp: &[u8] = &tokens[groups[first_b][0] as usize].0;
            let mut min_len = lcp.len();
            for &i in &groups[first_b] {
                // for each token index we start with the longest token
                // and we iterate over the lcp's bytes until there are
                // no longer equal to the current. At this point we shorten
                // the lcp and go to the next.
                let t = &tokens[i as usize].0;
                min_len = min_len.min(t.len());
                let common = lcp.iter().zip(t).take_while(|(a, b)| a == b).count();
                lcp = &lcp[..common];
            }
            // cap so prefix.len() < every token's len => every token has a byte at prefix.len()
            let plen = lcp.len().min(min_len.saturating_sub(1));
            let prefix: Box<[u8]> = lcp[..plen].to_vec().into_boxed_slice();
            // build the byte after prefix -> sub-list + per-byte distinct lengths, longest first
            let mut next_byte_to_length_id = [0xFFFFu16; 256];
            let mut length_list: Vec<Vec<u16>> = Vec::new();
            for &i in &groups[first_b] {
                let t = &tokens[i as usize].0;
                let post_byte = t[plen] as usize;
                let list_index = if next_byte_to_length_id[post_byte] == 0xFFFF {
                    next_byte_to_length_id[post_byte] = length_list.len() as u16;
                    length_list.push(Vec::new());
                    length_list.len() - 1
                } else {
                    next_byte_to_length_id[post_byte] as usize
                };
                let l = t.len() as u16;
                // Insert straight into place, longest first. The scan that used to only answer
                // "is `l` already in here?" now also says where `l` belongs, which retires the
                // descending sort that followed -- and with it a whole `[u16]` copy of
                // core::slice::sort in the binary. These lists are one entry per distinct token
                // length behind one byte, so a handful of entries at most.
                let list = &mut length_list[list_index];
                match list.iter().position(|&x| x <= l) {
                    Some(p) if list[p] == l => {} // already present
                    Some(p) => list.insert(p, l),
                    None => list.push(l),
                }
            }
            debug_assert!(buckets.len() < 0xFF, "more than 254 distinct first bytes");
            first_byte_to_bucket_id[first_b] = buckets.len() as u8;
            buckets.push(Bucket::new(
                prefix,
                next_byte_to_length_id,
                length_list.into_iter().map(Vec::into_boxed_slice).collect(),
            ));
        }
        Self::build(tokens, first_byte_to_bucket_id, buckets.into_boxed_slice())
    }
    // If you have the following first bytes
    // "<"  = 0x3c  -> col 3, row c
    // "["  = 0x5b  -> col 5, row b
    // "\t" = 0x09  -> col 0, row 9
    // "\n" = 0x0a  -> col 0, row a
    // `pshufb` maps bytes to bytes, so to have a 16bit row  bitmap, we need to split it into two 16x8bit bitmaps. We need two calls to `pshufb` to pull one row of the bitmap.
    // So the lo[0xc] = 0b00010000 ->
    //                  ...x....
    // stores half the row in the u8 format. (0-255)
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
                } // >8 high nibbles -> fallback to the general implementation TODO:
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
    #[cfg(all(test, target_arch = "aarch64"))] // test-only helper for the NEON nibble matcher
    fn nibble_match_bytes(&self, bytes: &[u8]) -> Option<usize> {
        let tables = NibbleTables::new(&self.lo16, &self.hi16);
        let mut base = 0;
        while let Some(window) = bytes[base..].first_chunk::<16>() {
            let mask = tables.candidates(window);
            if mask != 0 {
                return Some(base + (mask.trailing_zeros() >> NibbleTables::LANE_SHIFT) as usize);
            }
            base += 16;
        }
        (base..bytes.len()).find(|&i| self.first_byte_to_bucket_id[bytes[i] as usize] != 0xFF)
    }

    /// Reduced per-candidate reject. The first byte already matched (that's why `pos` is a
    /// candidate), so verify the rest of the prefix as ONE masked `u64` compare — no boxed-slice
    /// walk, no loop, `prefix[0]` folded in for free. Only the rare survivor reaches the disc table
    /// and the hash. This is the work that dominates dense input, so it has to be tiny.
    #[inline(always)]
    fn match_fast(&self, bytes: &[u8], pos: usize, bucket_id: u32) -> Option<(u32, u32)> {
        let bucket = &self.buckets[bucket_id as usize];
        let n = bytes.len();
        // 1. prefix: single masked u64 compare over prefix[0..min(len,8)].
        if pos + 8 <= n {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[pos..pos + 8]);
            let chunk = u64::from_ne_bytes(buf);
            if (chunk ^ bucket.prefix_word) & bucket.prefix_mask != 0 {
                return None;
            }
            // 1.b. prefix longer than 8 bytes (rare): verify the remainder.
            if bucket.prefix.len() > 8 && !bytes[pos + 8..].starts_with(&bucket.prefix[8..]) {
                return None;
            }
        } else if !bytes[pos..].starts_with(&bucket.prefix) {
            return None; // near the end of the buffer: too few bytes for a word load
        }
        // 2. byte after the prefix -> length sub-list (0xFFFF = none).
        let post_byte = match bytes.get(pos + bucket.prefix.len()) {
            Some(&b) => b,
            None => return None,
        };
        let length_id = bucket.next_byte_to_length_id[post_byte as usize];
        if length_id == 0xFFFF {
            return None;
        }
        // 3. probe candidate lengths, longest first; confirm via the hash.
        for &len in bucket.length_list[length_id as usize].iter() {
            let len = len as usize;
            if pos + len <= n
                && let Some(id) = self.vocab.get_bytes(&bytes[pos..pos + len])
            {
                return Some((id, len as u32));
            }
        }
        None
    }

    /// Find the leftmost added-token match in `bytes`. Returns (token_id, match_position, match_len).
    /// Dispatch: 1 bucket -> memchr on the shared first byte; >=2 -> NEON nibble, mask-iterated.
    pub fn match_bytes(&self, bytes: &[u8]) -> Option<(u32, u32, u32)> {
        match self.buckets.len() {
            0 => None,
            1 => {
                // needle = the bucket's shared first byte. Assumes a non-empty prefix
                // (false only if a lone 1-byte token is the sole holder of its first byte); store
                // the first byte explicitly if that case ever appears.
                // Search for the bucket's whole shared prefix, not just its first byte.
                //
                // One byte of a long needle is a poor filter on text that is dense in that byte and
                // holds no match: `<|endoftext|>` over a corpus full of `<|xs0|>` stops at every `<`
                // and dies at the third byte. Restarting `memchr` on `&bytes[pos + 1..]` per
                // candidate then pays its prologue again each time. Measured over the same
                // one-bucket vocabulary in one process, that cost 0.58 ns/B on a 9.8%-`<` corpus
                // that matches nothing, against 0.12 for `memmem`, which picks a *rare* byte of the
                // needle instead. Where candidates are already sparse it costs at most 0.015 ns/B.
                //
                // `nibble_mask_match` already avoids this for two or more buckets and calls it the
                // restart penalty. Same positions in the same order, so the leftmost match is
                // unchanged and `match_fast` still confirms the length sub-list.
                for pos in memchr::memmem::find_iter(bytes, &self.buckets[0].prefix) {
                    if let Some((id, len)) = self.match_fast(bytes, pos, 0) {
                        return Some((id, pos as u32, len));
                    }
                }
                None
            }
            _ => self.nibble_mask_match(bytes),
        }
    }

    /// Two or more buckets: classify each 16-byte window once with [`NibbleTables`], then run
    /// [`Self::match_fast`] on every candidate lane before moving to the next window. Returning on
    /// the first candidate would reload the window after each false one (the restart penalty).
    /// Bytes past the last full window, and all bytes on targets without [`NibbleTables`], go
    /// through [`Self::scalar_match`].
    fn nibble_mask_match(&self, bytes: &[u8]) -> Option<(u32, u32, u32)> {
        #[cfg(not(target_arch = "aarch64"))]
        let start = 0;
        #[cfg(target_arch = "aarch64")]
        let start = {
            let tables = NibbleTables::new(&self.lo16, &self.hi16);
            // Keep few values live around the inlined `match_fast`: a chunk iterator or a `window`
            // index here made LLVM spill to the stack, up to 2.6x slower.
            let mut base = 0;
            while let Some(window) = bytes[base..].first_chunk::<16>() {
                let mut mask = tables.candidates(window);
                while mask != 0 {
                    let lane = (mask.trailing_zeros() >> NibbleTables::LANE_SHIFT) as usize;
                    let pos = base + lane;
                    let bucket = self.first_byte_to_bucket_id[bytes[pos] as usize] as u32;
                    if let Some((id, len)) = self.match_fast(bytes, pos, bucket) {
                        return Some((id, pos as u32, len));
                    }
                    mask &= mask - 1;
                }
                base += 16;
            }
            base
        };
        self.scalar_match(bytes, start)
    }

    /// Looks up every byte of `bytes[start..]` in `first_byte_to_bucket_id`.
    fn scalar_match(&self, bytes: &[u8], start: usize) -> Option<(u32, u32, u32)> {
        for (i, &byte) in bytes[start..].iter().enumerate() {
            let pos = start + i;
            let bucket = self.first_byte_to_bucket_id[byte as usize];
            if bucket != 0xFF
                && let Some((id, len)) = self.match_fast(bytes, pos, bucket as u32)
            {
                return Some((id, pos as u32, len));
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }
    /// first byte -> bucket index (0xFF = none). The builder reads this to extend buckets.
    pub fn first_byte_to_bucket_id(&self) -> &[u8; 256] {
        &self.first_byte_to_bucket_id
    }
    /// All added-token byte strings + ids (e.g. to rebuild the vocab when adding tokens).
    pub fn get_vocab_bytes(&self) -> Vec<(Vec<u8>, u32)> {
        self.vocab.byte_content()
    }
    /// All added-token strings + ids.
    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.vocab.content()
    }
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.token_to_id(token)
    }
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
    }
}

/// [`Buckets::lo16`] and [`Buckets::hi16`] loaded into NEON registers once per scan.
#[cfg(target_arch = "aarch64")]
struct NibbleTables {
    lo: core::arch::aarch64::uint8x16_t,
    hi: core::arch::aarch64::uint8x16_t,
}

#[cfg(target_arch = "aarch64")]
impl NibbleTables {
    /// Lane `i` of a window is reported by bit `i << LANE_SHIFT` of [`Self::candidates`].
    const LANE_SHIFT: u32 = 2;

    fn new(lo16: &[u8; 16], hi16: &[u8; 16]) -> Self {
        use core::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64, and each load reads exactly its 16-byte array.
        unsafe {
            Self {
                lo: vld1q_u8(lo16.as_ptr()),
                hi: vld1q_u8(hi16.as_ptr()),
            }
        }
    }

    /// One set bit per candidate byte of `window`, at [`Self::LANE_SHIFT`] spacing.
    ///
    /// ```text
    /// lane index    0     1     2     3     4    ...  (16 lanes, one register)
    /// input byte   'a'   'b'   '<'   'd'   'e'   ...
    /// hex           61    62    3C    64    65   ...
    ///
    /// low nibbles   v & 0x0F    →   1     2     C     4     5   ...
    /// high nibbles  v >> 4      →   6     6     3     6     6   ...
    ///
    /// lo hits       lo16[low]   →   0     0   0b01    0     0   ...
    /// hi hits       hi16[high]  →   0     0   0b01    0     0   ...
    ///
    /// lo & hi                   →   0     0   0b01    0     0   ...  lane 2 is a candidate
    /// ```
    fn candidates(&self, window: &[u8; 16]) -> u64 {
        use core::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64, and the load reads exactly the 16-byte `window`.
        unsafe {
            let v = vld1q_u8(window.as_ptr());
            let lo = vqtbl1q_u8(self.lo, vandq_u8(v, vdupq_n_u8(0x0f)));
            let hi = vqtbl1q_u8(self.hi, vshrq_n_u8(v, 4));
            let m = vandq_u8(lo, hi);
            // NEON has no movemask. Shifting each u16 right by 4 and narrowing packs 4 bits per lane.
            let t = vreinterpretq_u16_u8(vtstq_u8(m, m));
            vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(t, 4)), 0) & 0x1111_1111_1111_1111
        }
    }
}

impl Default for Buckets {
    fn default() -> Self {
        Self::new()
    }
}

/// We add the bench as test to test private func
#[cfg(test)]
mod bench {
    // Throughput: optimized `match_bytes` (memchr for 1 bucket, NEON nibble for >=2 buckets) vs a
    // naive scalar first-byte scan vs daachorse. All three find every special-token match + advance.
    //   cargo test --release -p tk-encode bench_match_bytes -- --ignored --nocapture
    use super::*;
    use daachorse::DoubleArrayAhoCorasick;
    use std::hint::black_box;
    use std::time::Instant;

    // Optimized: drive `match_bytes` over the input the way `extract_and_normalize` does.
    fn scan_opt(b: &Buckets, bytes: &[u8]) -> usize {
        let (mut hits, mut search) = (0usize, 0usize);
        while search < bytes.len() {
            match b.match_bytes(&bytes[search..]) {
                Some((_, start, len)) if len > 0 => {
                    hits += 1;
                    search += start as usize + len as usize;
                }
                _ => break,
            }
        }
        hits
    }
    // Naive baseline, IREE-style: tokens grouped by first byte, each group sorted longest-first.
    // At a candidate position we linearly scan that group and byte-compare (`starts_with`) — no
    // perfect hash, no disc table, no SIMD. This is what IREE-style matching reduces to: scan the
    // candidate tokens and compare bytes (IREE pre-filters on a capped prefix; the first-byte group
    // is that filter at one byte). Isolates the win of the bucket + disc + MPHF path over a scan.
    struct NaiveMatcher {
        by_first_byte: Vec<Vec<(Vec<u8>, u32)>>, // [256]; each group sorted longest-first
    }
    impl NaiveMatcher {
        fn new(tokens: &[(Vec<u8>, u32)]) -> Self {
            let mut by_first_byte = vec![Vec::new(); 256];
            for (tok, id) in tokens {
                if let Some(&fb) = tok.first() {
                    by_first_byte[fb as usize].push((tok.clone(), *id));
                }
            }
            for g in by_first_byte.iter_mut() {
                g.sort_by_key(|(v, _)| std::cmp::Reverse(v.len())); // longest first -> first hit is longest
            }
            Self { by_first_byte }
        }
        #[inline]
        fn longest_match(&self, bytes: &[u8], i: usize) -> Option<(u32, u32)> {
            for (tok, id) in &self.by_first_byte[bytes[i] as usize] {
                if bytes[i..].starts_with(tok) {
                    return Some((*id, tok.len() as u32));
                }
            }
            None // empty group (no token starts with this byte) lands here too -> cheap reject
        }
    }

    fn scan_naive(m: &NaiveMatcher, bytes: &[u8]) -> usize {
        let (mut hits, mut i) = (0usize, 0usize);
        while i < bytes.len() {
            if let Some((_, len)) = m.longest_match(bytes, i) {
                hits += 1;
                i += len as usize;
            } else {
                i += 1;
            }
        }
        hits
    }
    fn scan_daachorse(bytes: &[u8], pma: &DoubleArrayAhoCorasick<usize>) -> usize {
        pma.find_iter(bytes).count()
    }

    // Hoist-safe timer: black_box the input on every call, accumulate every result so LLVM
    // can't hoist the (pure) scan out of the loop and report a fictitious 0 ns/byte.
    fn time<F: Fn() -> usize>(bytes_len: usize, iters: u32, f: F) -> f64 {
        for _ in 0..2 {
            black_box(f());
        }
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..iters {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        t.elapsed().as_nanos() as f64 / (iters as usize * bytes_len) as f64
    }

    fn run(name: &str, b: &Buckets, pma: &DoubleArrayAhoCorasick<usize>, bytes: &[u8]) {
        let m = NaiveMatcher::new(&b.get_vocab_bytes()); // built once, outside the timed loop
        let (o, n) = (scan_opt(b, bytes), scan_naive(&m, bytes));
        let cand = bytes
            .iter()
            .filter(|&&x| b.first_byte_to_bucket_id[x as usize] != 0xFF)
            .count();
        let iters = 20u32;
        let opt = time(bytes.len(), iters, || scan_opt(b, black_box(bytes)));
        let naive = time(bytes.len(), iters, || scan_naive(&m, black_box(bytes)));
        let daach = time(bytes.len(), iters, || scan_daachorse(black_box(bytes), pma));
        println!(
            "=== {name} ({:.1} MB, {:.2}% candidate bytes, {o} matches{}) ===",
            bytes.len() as f64 / 1e6,
            100.0 * cand as f64 / bytes.len() as f64,
            if o == n {
                ""
            } else {
                " — OPT/NAIVE DISAGREE"
            }
        );
        println!(
            "  match_bytes : {opt:.3} ns/byte ({:>5.0} MB/s)",
            1000.0 / opt
        );
        println!(
            "  naive scan  : {naive:.3} ns/byte ({:>5.0} MB/s)  match_bytes {:.2}x",
            1000.0 / naive,
            naive / opt
        );
        println!(
            "  daachorse   : {daach:.3} ns/byte ({:>5.0} MB/s)  match_bytes {:.2}x\n",
            1000.0 / daach,
            daach / opt
        );
    }

    // Build `len` bytes with ~`density` candidate first-bytes ('<', evenly spread via a Bresenham
    // accumulator). Those '<' are FALSE candidates ("<a" fails starts_with("<|") -> cheap reject);
    // a real `<|eos|>` is injected every ~10k bytes so hits stay sparse and the scan cost dominates.
    fn make_input(len: usize, density: f64) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 8);
        let mut acc = 0.0f64;
        while out.len() < len {
            acc += density;
            if acc >= 1.0 {
                acc -= 1.0;
                out.push(b'<');
            } else {
                out.push(b'a');
            }
            if out.len() % 10_000 == 0 {
                out.extend_from_slice(b"<|eos|>");
            }
        }
        out
    }

    #[test]
    #[ignore] // slow; explicit run only
    fn bench_match_bytes_vs_naive() {
        let specials: Vec<(Vec<u8>, u32)> = ["<|bos|>", "<|eos|>", "<|pad|>", "[CLS]", "[SEP]"]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().to_vec(), i as u32))
            .collect();
        let b = Buckets::from_tokens(specials.clone()).unwrap();
        let patterns: Vec<&Vec<u8>> = specials.iter().map(|(v, _)| v).collect();
        let pma: DoubleArrayAhoCorasick<usize> = DoubleArrayAhoCorasick::new(patterns).unwrap();

        // Sweep candidate-byte density. The SIMD skip wins big when candidates are sparse (real
        // text); it erodes as they get dense because there is nothing left to skip and per-candidate
        // extraction costs more than the naive flat scan's well-predicted branch.
        for density in [0.01f64, 0.05, 0.25, 0.50, 0.90] {
            let input = make_input(4_000_000, density);
            run(
                &format!("density {:>4.0}%", density * 100.0),
                &b,
                &pma,
                &input,
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")] // calls the NEON-only nibble_match_bytes
    #[test]
    fn test_build_nibble_table() {
        let mut first_byte_to_bucket = [0xFFu8; 256];
        // spell "toad": t=0x74, o=0x6F, a=0x61, d=0x64
        //   TOP    collision: o, a, d all have high nibble 6 (same row -> same hi nibble)
        //   BOTTOM collision: t and d share low nibble 4 (lo16[4] ends up with two bits)
        first_byte_to_bucket[b't' as usize] = 0;
        first_byte_to_bucket[b'o' as usize] = 1;
        first_byte_to_bucket[b'a' as usize] = 2;
        first_byte_to_bucket[b'd' as usize] = 3;

        let mut expected_hi16 = [0u8; 16];
        let mut expected_lo16 = [0u8; 16];

        // filling hi goes from 0..=0xF. First match gets the first id.
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
        )
        .unwrap();
        fake_bucket.build_nibble_table();
        assert_eq!(fake_bucket.lo16, expected_lo16);
        assert_eq!(fake_bucket.hi16, expected_hi16);
        assert!(fake_bucket.nibble_match_bytes("pardis".as_bytes()) == Some(1));
        assert_eq!(
            fake_bucket.nibble_match_bytes("where is toad".as_bytes()),
            Some(9)
        );
    }

    #[test]
    fn test_match_bytes() {
        let mut first_byte_to_bucket = [0xFFu8; 256];
        first_byte_to_bucket[b'<' as usize] = 0;
        let mut next_byte_to_length_id = [0xFFFFu16; 256];
        next_byte_to_length_id[b'|' as usize] = 0;
        let fake_vocab = Buckets::build(
            vec![("<|eos|>".as_bytes().to_vec(), 0)],
            first_byte_to_bucket,
            Box::new([Bucket::new(
                Box::from(*b"<"),
                next_byte_to_length_id,
                Box::new([Box::new([7])]),
            )]),
        )
        .unwrap();
        assert_eq!(
            fake_vocab.match_bytes(b"This should be kwown<s><|eos|>"),
            Some((0, 23, 7))
        );
        assert_eq!(fake_vocab.match_bytes(b"><|eos|>"), Some((0, 1, 7)));
        // now nible match
        first_byte_to_bucket[b'<' as usize] = 0;
        first_byte_to_bucket[b'[' as usize] = 1;
        first_byte_to_bucket[b'|' as usize] = 2;
        first_byte_to_bucket[b']' as usize] = 3;
        let mut next_byte_to_length_id = [0xFFFFu16; 256];
        // we re-use it because we assume a single token
        next_byte_to_length_id[b'|' as usize] = 0;
        next_byte_to_length_id[b'C' as usize] = 0;
        next_byte_to_length_id[b'S' as usize] = 0;
        next_byte_to_length_id[b'F' as usize] = 0;
        let fake_vocab = Buckets::build(
            vec![
                ("<|eos|>".as_bytes().to_vec(), 0),
                ("[CLS]".as_bytes().to_vec(), 1),
                ("|SLS]".as_bytes().to_vec(), 2),
                ("]FLS]".as_bytes().to_vec(), 3),
            ],
            first_byte_to_bucket,
            Box::new([
                Bucket::new(
                    Box::from(*b"<"),
                    next_byte_to_length_id,
                    Box::new([Box::new([7])]),
                ),
                Bucket::new(
                    Box::from(*b"["),
                    next_byte_to_length_id,
                    Box::new([Box::new([5])]),
                ),
                Bucket::new(
                    Box::from(*b"|"),
                    next_byte_to_length_id,
                    Box::new([Box::new([5])]),
                ),
                Bucket::new(
                    Box::from(*b"]"),
                    next_byte_to_length_id,
                    Box::new([Box::new([5])]),
                ),
            ]),
        )
        .unwrap();

        assert_eq!(fake_vocab.buckets.len(), 4);
        assert_eq!(fake_vocab.match_bytes(b"><|eos|>"), Some((0, 1, 7)));
        assert_eq!(fake_vocab.match_bytes(b"|SLS]>"), Some((2, 0, 5)));
        // if ]] then it will exit early to ask to move the pointer?
        assert_eq!(fake_vocab.match_bytes(b"]FLS]>"), Some((3, 0, 5)));
        assert_eq!(fake_vocab.match_bytes(b"]]FLS]>"), Some((3, 1, 5)));
    }
}
