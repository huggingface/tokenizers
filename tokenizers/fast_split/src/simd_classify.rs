use super::classify::Atom;
/// ┌───────────────────────── OWNER: SIMD path ─────────────────────────┐
/// ASCII nibble-shuffle → atom; 2-byte via `vqtbl` bitmap membership (`LETTER2`/`NUMBER2`/`MARK2`/…);
/// CJK by lead-byte range; 3-byte-non-CJK via the branchless SIMD range kernel `Σ (cp ≥ tᵢ)`.
#[cfg(all(target_arch = "aarch64"))]
unsafe fn classify_neon(text: &[u8], tags: &mut [u8]) {
    use core::arch::aarch64::*;
    let n = text.len();
    let mut i = 0;
    unsafe {
        while i + 32 < n {
            // we iterate 16 bytes by 16 but because of multbyte, we need to load the content of the
            // next chunk of 16. Thus we take a larger window.
            // 1. Load into registers
            let bytes = vld1q_u8(text.as_ptr().add(i));
            // 2. ASCII fast path:
            let out = if vmaxvq_u8(bytes) < 0x80 {
                // Letters in utf-8 start with different ranges.
                // We load the default value in out, which will be the tags
                let mut out = vdupq_n_u8(Atom::SymOther as u8);
                // first range of printable chars
                // from '!' to '~' are all
                // printable, we set them to be punctuation
                let printable = vandq_u8(
                    vcgeq_u8(bytes, vdupq_n_u8(0x21)),
                    vcleq_u8(bytes, vdupq_n_u8(0x7e)),
                );
                out = vbslq_u8(printable, vdupq_n_u8(Atom::Punct as u8), out);
                // Now '\n' and '\r'
                let new_line = vandq_u8(
                    vceqq_u8(bytes, vdupq_n_u8(0x0A)),
                    vceqq_u8(bytes, vdupq_n_u8(0x0D)),
                );
                out = vbslq_u8(new_line, vdupq_n_u8(Atom::Newline as u8), out);

                // we change the value for the space: 0x20
                out = vbslq_u8(
                    vceqq_u8(bytes, vdupq_n_u8(0x20)),
                    vdupq_n_u8(Atom::Space as u8),
                    out,
                );
                out
            } else {
                let next_bytes = vld1q_u8(text.as_ptr().add(i + 16));
                let bytes_2 = vextq_u8::<1>(bytes, next_bytes);

                todo!();
            };
            vst1q_u8(tags.as_mut_ptr().add(i), out);
            i += 16;
        }
    }
}
//out = vbsl(v==0x0A | v==0x0D,            Newline,    out)
// out = vbsl(v==0x09 | v==0x0B | v==0x0C,  WsOther,    out)
// out = vbsl(v==0x20,                      Space,      out)
// out = vbsl(digit(0x30..0x39),            NumWord,    out)
// out = vbsl((v|0x20) in 0x61..0x7A,       Letter,     out)   // case-fold trick for a-z
// out = vbsl(v==0x5F,                      Connector,  out)   // '_'
// out = vbsl(v==0x27,                      Apostrophe, out)   // '\''
// 4 different SIMD function to implement. Each take text and tags and produce a tag value per char
//   in the tect right?
// Continuation-byte sentinel is written to every non-lead byte.
// There are 15 different ATOMS. which are combined to create the different rules.
// The tags is the buffer where we'll right the atom stream.
// Let's start with the ascii nibble-shuffle
// It is the first that will run on all lanes.
// out = ascii_classify(v)                 // all 16 lanes get an ASCII atom (wrong for non-ASCII lanes)
// out = vbsl(is_2byte_mask, twobyte(v), out)   // overwrite ONLY the 2-byte-lead lanes
// out = vbsl(is_cjk_mask,   cjk(v),     out)   // overwrite ONLY the CJK lanes
// out = vbsl(is_cont_mask,  CONT,       out)   // overwrite continuation bytes
// if vmaxvq_u8(is_2byte_mask) != 0 { ...the 2-byte work + blend... }
// this is scalar but locality of script means it predicts fairly well.
// the only ambiguous locations are script boundaries: from english to say chinese and etc, but
// they happen for a single data stream.
// The is_2byte mask — yes, cheap, just not literally one op
//
// A 2-byte lead is 110xxxxx (0xC2–0xDF), so the test is (v & 0xE0) == 0xC0:
// is2 = vceqq_u8(vandq_u8(v, vdupq_n_u8(0xE0)), vdupq_n_u8(0xC0))   // vand + vceq = 2 ops
// → a per-lane mask (0xFF where it's a 2-byte lead, 0x00 elsewhere). Your intuition holds: a couple of elementwise ops → a mask. Same shape for the others:
// - CJK lead range: vcgeq(v,0xE3) & vcleq(v,0xED) (range = 2 compares + 1 and).
// - continuation byte: (v & 0xC0) == 0x80.
//
// All cheap, all produce per-lane masks you feed to vbsl.
//
// "Lane-wise" for 2/3-byte — yes, but you need vext first
//
// Here's the subtlety. The classification math is lane-wise (elementwise: vand, vceq, vcge, vqtbl, vshl — each lane independent). But a multibyte character spans several lanes: the lead is in lane i, its continuation byte(s) in lanes i+1 (and i+2 for 3-byte). To classify the char at the lead lane, that lane needs the continuation bytes — which live in adjacent lanes.
//
// That cross-lane move is vext (a byte-shift of the whole vector):
// let b2 = vextq_u8(v, vn, 1);   // at each lane i, this holds byte[i+1]  (the 1st continuation)
// let b3 = vextq_u8(v, vn, 2);   // at each lane i, this holds byte[i+2]  (the 2nd continuation)
// Now at lane i you have v[i] (lead), b2[i] (cont1), b3[i] (cont2) all in the same lane, and every classification op after that is pure lane-wise:
// - 2-byte bitmap: index ci computed from v and b2, then vqtbl + vshl + vtst — all elementwise.
// - CJK range: vceq/vcge on v, b2, b3 — elementwise.
// Counting can potentially be done using xor operations to go form 100001000 to 011110000 -> fills
// the whole. this is potentially what we'll use in some of the fast fsm to detect boundary changes?
// checking ascii is just checking if there is a header:  0x8080808080808080 with 8 bytes. find the
// simd for this.
// UTF8 validator defines 12 different categories. SIMD JSOM only finds a few
// characters that are in a set, which is a very small set of ASCII (5 different characters) We
// need much more than that, we need to classify in one of the 16 different categories
// Every single byte has a different category. there are 256 possible bytes, and 12 categories for
// utf8 validators. SIMD registers take at least 16bytes.
//
