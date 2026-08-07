//! Converting a pretoken into internal symbols, in the shape the merge engine that will process
//! it (`merge_multipass` or `merge_hot_cold_queue`) wants to start from.
use crate::models::bpe::At;
use crate::models::bpe::merge_hot_cold_queue::Entry;
use crate::models::bpe::model::{Atoms, PipelineBPE};
use crate::models::bpe::tables::{BpeTables, ID_MASK, RANK_MASK};
use crate::vocab::bucket_vocab_store::BucketVocabStore;

/// Set only for the few models that decorate their atoms: `end_of_word_suffix` (CLIP, openai-gpt,
/// XLM) and `continuing_subword_prefix`. A character's atom then depends on its position in the
/// word, so those models take a slow path that looks each decorated character up in the vocab.
pub(super) struct Affixes {
    pub(super) prefix: String,
    pub(super) suffix: String,
    /// Dense `external vocab id -> internal symbol id`, `u32::MAX` where there is none. Dense
    /// beats a hash here because external ids are `0..vocab_size`: 4 bytes a slot and one load,
    /// against 8-16 for any map. It is the array `BpeTables::build` makes anyway.
    pub(super) to_internal: Box<[u32]>,
}

/// Longest `prefix + one character + suffix` the stack buffer holds.
pub(super) const AFFIX_BUF: usize = 64;

/// UTF-8 sequence length by lead byte.
const UTF8_LEN: [u8; 256] = {
    let mut l = [1u8; 256];
    let mut b = 0xC0usize;
    while b < 0xE0 {
        l[b] = 2;
        b += 1;
    }
    while b < 0xF0 {
        l[b] = 3;
        b += 1;
    }
    while b < 0xF8 {
        l[b] = 4;
        b += 1;
    }
    l
};

/// Collects the converted symbols of a sequence into whatever the engine that merges it needs.
/// The implementing type picks which: [`MultipassSink`] or [`QueueSink`]. The mode is a type
/// rather than a runtime flag so the conversion loops are compiled once per engine, without a
/// per-symbol test.
trait SinkMode {
    /// Records the looked-up value of the pair the last two symbols form.
    fn record_pair(&mut self, merge: u64, previous: u32, symbol: u32);
    /// Records the symbol itself.
    fn push_symbol(&mut self, symbol: u32);
}

/// The flat symbol array plus the lowest-ranked adjacent pair, which is the first merge multipass
/// applies.
struct MultipassSink<'a> {
    symbols: &'a mut Vec<u32>,
    /// `ranks[i]` is the value of the pair `(symbols[i], symbols[i + 1])`.
    ///
    /// Conversion already looks every pair up, so seeding this costs one store per pair and no
    /// extra lookup. It is what lets the merge passes stop re-ranking pairs they did not touch.
    ranks: &'a mut Vec<u32>,
    /// The product id of each pair, kept apart from its rank so the merge loop's search array stays
    /// a dense `u32` of ranks alone.
    prods: &'a mut Vec<u32>,
    lowest_merge: u64,
}

impl SinkMode for MultipassSink<'_> {
    #[inline(always)]
    fn record_pair(&mut self, merge: u64, _previous: u32, _symbol: u32) {
        self.ranks.push((merge >> 32) as u32);
        self.prods.push((merge & ID_MASK) as u32);
        self.lowest_merge = self.lowest_merge.min(merge);
    }
    #[inline(always)]
    fn push_symbol(&mut self, symbol: u32) {
        self.symbols.push(symbol);
    }
}

/// The pair entries and cold queue keys, built as the symbols are produced, so the hot/cold queue
/// needs no intermediate array to read back.
struct QueueSink<'a> {
    entries: &'a mut Vec<Entry>,
    cold: &'a mut Vec<u64>,
}

impl SinkMode for QueueSink<'_> {
    #[inline(always)]
    fn record_pair(&mut self, merge: u64, previous: u32, symbol: u32) {
        let index = self.entries.len() as u32;
        self.entries.push(Entry {
            rank: (merge >> 32) as u32,
            prod: (merge & ID_MASK) as u32,
            a: previous,
            b: symbol,
            l: index.wrapping_sub(1), // u32::MAX at index 0, which is NONE
            r: index + 1,             // the final entry is patched in `convert_queue`
        });
        if merge != u64::MAX {
            self.cold.push((merge & RANK_MASK) | index as u64);
        }
    }
    #[inline(always)]
    fn push_symbol(&mut self, _symbol: u32) {}
}

/// Feeds each converted symbol to the [`SinkMode`], with its pair looked up exactly once: both
/// engines want that same value, multipass for the minimum and the queue for the pair's rank and
/// product.
struct SymbolSink<M> {
    mode: M,
    previous_symbol: u32,
}

impl<M: SinkMode> SymbolSink<M> {
    // inlining here is very important
    #[inline(always)]
    fn push(&mut self, tables: &BpeTables, symbol: u32) {
        if self.previous_symbol != u32::MAX {
            let merge = tables.get_value(&self.previous_symbol, &symbol);
            self.mode.record_pair(merge, self.previous_symbol, symbol);
        }
        self.previous_symbol = symbol;
        self.mode.push_symbol(symbol);
    }

    /// Pushes `character` as its bytes, each mapped to a symbol through `byte_symbols`.
    #[inline(always)]
    fn push_char_bytes(&mut self, tables: &BpeTables, byte_symbols: &[u32; 256], character: char) {
        let mut buf = [0u8; 4];
        for &byte in character.encode_utf8(&mut buf).as_bytes() {
            self.push(tables, byte_symbols.at(byte as usize));
        }
    }
}

impl PipelineBPE {
    /// Converts one pretoken to internal IDs, returning the lowest-ranked adjacent pair,
    /// `u64::MAX` when no pair merges.
    pub(super) fn convert_multipass(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
        ranks: &mut Vec<u32>,
        prods: &mut Vec<u32>,
    ) -> u64 {
        symbols.clear();
        ranks.clear();
        prods.clear();
        // a word never has more symbols than bytes, so one reserve covers every push
        symbols.reserve(sequence.len());
        ranks.reserve(sequence.len());
        prods.reserve(sequence.len());
        let mut sink = SymbolSink {
            mode: MultipassSink {
                symbols,
                ranks,
                prods,
                lowest_merge: u64::MAX,
            },
            previous_symbol: u32::MAX,
        };
        self.convert(sequence, &mut sink);
        sink.mode.lowest_merge
    }

    /// Converts one pretoken to pair entries and cold queue keys for the hot/cold queue.
    ///
    /// A pretoken of fewer than two symbols has no pairs and so no entries; its single symbol is
    /// left in `symbols` instead, and the queue engine sees an empty entry list.
    pub(super) fn convert_queue(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
        entries: &mut Vec<Entry>,
        cold: &mut Vec<u64>,
    ) {
        symbols.clear();
        entries.clear();
        cold.clear();
        // a word never has more symbols than bytes, so one reserve covers every push
        entries.reserve(sequence.len());
        cold.reserve(sequence.len());
        let mut sink = SymbolSink {
            mode: QueueSink { entries, cold },
            previous_symbol: u32::MAX,
        };
        self.convert(sequence, &mut sink);
        let last = sink.previous_symbol;
        match entries.last_mut() {
            Some(entry) => entry.r = u32::MAX, // NONE: nothing right of the final pair
            None if last != u32::MAX => symbols.push(last),
            None => {}
        }
    }

    fn convert<M: SinkMode>(&self, sequence: &str, sink: &mut SymbolSink<M>) {
        if let Some(affixes) = &self.affixes {
            convert_affixed(
                &self.tables,
                &self.vocab,
                &self.atoms,
                affixes,
                sequence,
                sink,
            );
        } else {
            match &self.atoms {
                Atoms::Bytes => convert_bytes(&self.tables, sequence.as_bytes(), sink),
                Atoms::Chars {
                    byte_fallback,
                    unk_token,
                    fuse_unk,
                } => convert_chars(
                    &self.tables,
                    byte_fallback.as_ref(),
                    *unk_token,
                    *fuse_unk,
                    sequence,
                    sink,
                ),
            }
        }
    }
}

fn convert_bytes<M: SinkMode>(tables: &BpeTables, bytes: &[u8], sink: &mut SymbolSink<M>) {
    let byte_symbols = &tables.byte_internal[..];
    let mut pos = 0usize;
    while pos < bytes.len() {
        // An ASCII character is exactly one symbol whether or not it folds, so this loop needs
        // no fold branch. `get` gives the bounds check and the byte in one step.
        while let Some(&ascii) = bytes.get(pos) {
            if ascii >= 0x80 {
                break;
            }
            sink.push(tables, tables.fold.get_ascii(ascii));
            pos += 1;
        }
        if pos >= bytes.len() {
            break; // the run ran to the end rather than stopping on a lead byte
        }
        let lead = bytes.at(pos);
        let char_len = UTF8_LEN[lead as usize] as usize;
        let folded = tables.fold.get_bytes(bytes, pos, lead, char_len);
        if folded != u32::MAX {
            sink.push(tables, folded);
        } else {
            for offset in 0..char_len {
                let byte = bytes.at(pos + offset) as usize;
                sink.push(tables, byte_symbols.at(byte));
            }
        }
        pos += char_len;
    }
}

/// Character-level conversion, for models without a byte-level pretokenizer: every vocab token
/// of one character has a fold entry, so there is no byte decomposition to do here.
fn convert_chars<M: SinkMode>(
    tables: &BpeTables,
    byte_fallback: Option<&[u32; 256]>,
    unk_token: Option<u32>,
    fuse_unk: bool,
    sequence: &str,
    sink: &mut SymbolSink<M>,
) {
    let mut in_unk_run = false;
    for character in sequence.chars() {
        let symbol = tables.fold.get_char(character);
        if symbol != u32::MAX {
            in_unk_run = false;
            sink.push(tables, symbol);
            continue;
        }
        if let Some(fallback) = byte_fallback {
            sink.push_char_bytes(tables, fallback, character);
            in_unk_run = false;
            continue;
        }
        if let Some(unk) = unk_token {
            // with fuse_unk the run already emitted its unk, so this character adds nothing
            if !(fuse_unk && in_unk_run) {
                sink.push(tables, unk);
            }
            in_unk_run = true;
        }
    }
}

/// Slow path for models that decorate their atoms: `continuing_subword_prefix` on every
/// character but the first, `end_of_word_suffix` on the last. The decorated form is assembled
/// in a stack buffer and looked up in the vocab, which costs a hash per character -- these
/// models are rare enough that it is not worth a second fold table to avoid it.
fn convert_affixed<M: SinkMode>(
    tables: &BpeTables,
    vocab: &BucketVocabStore,
    atoms: &Atoms,
    affixes: &Affixes,
    sequence: &str,
    sink: &mut SymbolSink<M>,
) {
    let mut buf = [0u8; AFFIX_BUF];
    let mut chars = sequence.chars().peekable();
    let mut is_first = true;
    while let Some(character) = chars.next() {
        let is_last = chars.peek().is_none();
        let mut len = 0;
        if !is_first {
            let bytes = affixes.prefix.as_bytes();
            buf[len..len + bytes.len()].copy_from_slice(bytes);
            len += bytes.len();
        }
        len += character.encode_utf8(&mut buf[len..]).len();
        if is_last {
            let bytes = affixes.suffix.as_bytes();
            buf[len..len + bytes.len()].copy_from_slice(bytes);
            len += bytes.len();
        }
        is_first = false;

        let symbol = std::str::from_utf8(&buf[..len])
            .ok()
            .and_then(|token| vocab.token_to_id(token))
            .and_then(|external| affixes.to_internal.get(external as usize).copied())
            .filter(|&symbol| symbol != u32::MAX);
        match symbol {
            Some(symbol) => sink.push(tables, symbol),
            None => push_unknown(tables, atoms, character, sink),
        }
    }
}

/// A character with no atom of its own: bytes if the model has `byte_fallback`, else `unk`.
fn push_unknown<M: SinkMode>(
    tables: &BpeTables,
    atoms: &Atoms,
    character: char,
    sink: &mut SymbolSink<M>,
) {
    match atoms {
        Atoms::Bytes => sink.push_char_bytes(tables, &tables.byte_internal, character),
        Atoms::Chars {
            byte_fallback,
            unk_token,
            ..
        } => {
            if let Some(fallback) = byte_fallback {
                sink.push_char_bytes(tables, fallback, character);
            } else if let Some(unk) = unk_token {
                sink.push(tables, *unk);
            }
        }
    }
}
