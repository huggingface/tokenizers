//! Converting a pretoken into internal symbols, in the shape the merge engine that will process
//! it (`merge_multipass` or `merge_hot_cold_queue`) wants to start from.
use crate::models::bpe::At;
use crate::models::bpe::merge_hot_cold_queue::{Entry, MergeQueue, QueueScratch};
use crate::models::bpe::model::{Atoms, BPE};
use crate::models::bpe::tables::{BpeTables, ID_MASK, NO_SYMBOL, RANK_MASK};

/// Set only for the few models that decorate their atoms: `end_of_word_suffix` (CLIP, openai-gpt,
/// XLM) and `continuing_subword_prefix`. An atom's symbol then depends on its position in the
/// word, so those models take a slow path that looks each decorated atom up in the vocab.
pub(super) struct Affixes {
    /// `continuing_subword_prefix` mapped to raw bytes
    pub(super) prefix: Box<[u8]>,
    /// `end_of_word_suffix` mapped to raw bytes
    pub(super) suffix: Box<[u8]>,
    /// Dense `external vocab id -> internal symbol id`, [`NO_SYMBOL`] where there is none. Dense
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
    lowest_merge: u64,
}

impl SinkMode for MultipassSink<'_> {
    #[inline(always)]
    fn record_pair(&mut self, merge: u64, _previous: u32, _symbol: u32) {
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
    queue: &'a mut MergeQueue,
}

impl SinkMode for QueueSink<'_> {
    #[inline(always)]
    fn record_pair(&mut self, merge: u64, previous: u32, symbol: u32) {
        let index = self.entries.len() as u32;
        self.entries.push(Entry {
            rank: (merge >> 32) as u32,
            merged_symbol: (merge & ID_MASK) as u32,
            left_symbol: previous,
            right_symbol: symbol,
            left_pair_index: index.wrapping_sub(1), // u32::MAX at index 0, which is NONE
            right_pair_index: index + 1,            // the final entry is patched in `convert_queue`
        });
        if merge != u64::MAX {
            self.queue.push_cold((merge & RANK_MASK) | index as u64);
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
        // This byte is not in the vocabulary: skip it
        if symbol == NO_SYMBOL {
            return;
        }
        if self.previous_symbol != NO_SYMBOL {
            let merge = tables.get_value(&self.previous_symbol, &symbol);
            self.mode.record_pair(merge, self.previous_symbol, symbol);
        }
        self.previous_symbol = symbol;
        self.mode.push_symbol(symbol);
    }

    /// Pushes each of `bytes` as the symbol `byte_symbols` maps it to.
    #[inline(always)]
    fn push_bytes(&mut self, tables: &BpeTables, byte_symbols: &[u32; 256], bytes: &[u8]) {
        for &byte in bytes {
            self.push(tables, byte_symbols.at(byte as usize));
        }
    }
}

impl BPE {
    /// Converts one pretoken to internal IDs, returning the lowest-ranked adjacent pair,
    /// `u64::MAX` when no pair merges.
    pub(super) fn convert_multipass(&self, sequence: &str, symbols: &mut Vec<u32>) -> u64 {
        symbols.clear();
        // a word never has more symbols than bytes, so one reserve covers every push
        symbols.reserve(sequence.len());
        let mut sink = SymbolSink {
            mode: MultipassSink {
                symbols,
                lowest_merge: u64::MAX,
            },
            previous_symbol: NO_SYMBOL,
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
        scratch: &mut QueueScratch,
    ) {
        let QueueScratch { entries, queue } = scratch;
        symbols.clear();
        entries.clear();
        // a word never has more symbols than bytes, so one reserve covers every push
        entries.reserve(sequence.len());
        queue.clear(sequence.len());
        let mut sink = SymbolSink {
            mode: QueueSink { entries, queue },
            previous_symbol: NO_SYMBOL,
        };
        self.convert(sequence, &mut sink);
        let last = sink.previous_symbol;
        match entries.last_mut() {
            Some(entry) => entry.right_pair_index = u32::MAX, // NONE: nothing right of the final pair
            None if last != NO_SYMBOL => symbols.push(last),
            None => {}
        }
    }

    fn convert<M: SinkMode>(&self, sequence: &str, sink: &mut SymbolSink<M>) {
        if let Some(affixes) = &self.affixes {
            convert_affixed(self, affixes, sequence, sink);
        } else {
            match &self.atoms {
                Atoms::Bytes {
                    missing_bytes: true,
                } if self.unk_token.is_some() => {
                    let mut in_unk_run = false;
                    for &byte in sequence.as_bytes() {
                        let symbol = self.tables.byte_internal[byte as usize];
                        if symbol == NO_SYMBOL {
                            self.push_unk(&mut in_unk_run, sink);
                        } else {
                            in_unk_run = false;
                            sink.push(&self.tables, symbol);
                        }
                    }
                }
                Atoms::Bytes { .. } => convert_bytes(&self.tables, sequence.as_bytes(), sink),
                Atoms::Chars { .. } => convert_chars(self, sequence, sink),
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
        if folded != NO_SYMBOL {
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
fn convert_chars<M: SinkMode>(model: &BPE, sequence: &str, sink: &mut SymbolSink<M>) {
    let mut in_unk_run = false;
    let sequence_bytes = sequence.as_bytes();
    for (start, character) in sequence.char_indices() {
        let symbol = model.tables.fold.get_char(character);
        if symbol != NO_SYMBOL {
            in_unk_run = false;
            sink.push(&model.tables, symbol);
        } else {
            let atom_bytes = &sequence_bytes[start..start + character.len_utf8()];
            model.push_unknown(&mut in_unk_run, atom_bytes, sink);
        }
    }
}

/// Slow path for models that decorate their atoms: `continuing_subword_prefix` on every atom but
/// the first, `end_of_word_suffix` on the last. An atom is a byte in a byte-level model, since its
/// words reach the model spelled one character per byte, and a character otherwise. The decorated
/// form is assembled in a stack buffer and looked up in the vocab, which costs a hash per atom --
/// these models are rare enough that it is not worth a second fold table to avoid it.
fn convert_affixed<M: SinkMode>(
    model: &BPE,
    affixes: &Affixes,
    sequence: &str,
    sink: &mut SymbolSink<M>,
) {
    let sequence_bytes = sequence.as_bytes();
    let mut buffer = [0u8; AFFIX_BUF];
    let mut in_unk_run = false;
    let mut pos = 0;
    while pos < sequence_bytes.len() {
        let atom_len = match model.atoms {
            Atoms::Bytes { .. } => 1,
            Atoms::Chars { .. } => UTF8_LEN[sequence_bytes[pos] as usize] as usize,
        };
        let atom = &sequence_bytes[pos..pos + atom_len];
        let mut len = 0;
        if pos > 0 {
            // Inside the sequence: insert the continuing subword prefix
            len = affixes.prefix.len();
            buffer[..len].copy_from_slice(&affixes.prefix);
        }
        buffer[len..len + atom_len].copy_from_slice(atom);
        len += atom_len;
        pos += atom_len;

        if pos == sequence_bytes.len() {
            // We've reached the end of the sequence: insert the end of word suffix
            let suffix = &affixes.suffix;
            buffer[len..len + suffix.len()].copy_from_slice(suffix);
            len += suffix.len();
        }

        let symbol = model
            .vocab
            .get_bytes(&buffer[..len])
            .and_then(|external| affixes.to_internal.get(external as usize).copied())
            .filter(|&symbol| symbol != NO_SYMBOL);
        match symbol {
            Some(symbol) => {
                in_unk_run = false;
                sink.push(&model.tables, symbol);
            }
            None => model.push_unknown(&mut in_unk_run, atom, sink),
        }
    }
}

impl BPE {
    fn push_unknown<M: SinkMode>(
        &self,
        in_unk_run: &mut bool,
        atom: &[u8],
        sink: &mut SymbolSink<M>,
    ) {
        match &self.atoms {
            Atoms::Chars {
                byte_fallback: Some(fallback),
            } => {
                sink.push_bytes(&self.tables, fallback, atom);
                *in_unk_run = false;
            }
            Atoms::Bytes { .. } if self.unk_token.is_none() => {
                sink.push_bytes(&self.tables, &self.tables.byte_internal, atom);
            }
            _ => self.push_unk(in_unk_run, sink),
        }
    }

    #[inline(always)]
    fn push_unk<M: SinkMode>(&self, in_unk_run: &mut bool, sink: &mut SymbolSink<M>) {
        if let Some(unk) = self.unk_token {
            if !(self.fuse_unk && *in_unk_run) {
                sink.push(&self.tables, unk);
            }
            *in_unk_run = true;
        }
    }
}
