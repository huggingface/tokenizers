//! Turning a pretokenized word into merge ranks, which are then processed in `multipass` or
//! `two_tier_merge`.
use crate::models::bpe::pipeline_bpe::{AFFIX_BUF, Atoms, PipelineBPE};
use crate::models::bpe::tables::{At, BpeTables, UTF8_LEN};
use crate::models::bpe::two_tier_merge::Entry;

/// Collects the converted ranks of a sequence into whatever the engine that merges it needs.
/// `MULTIPASS` picks which: a flat rank array plus the lowest-ranked adjacent pair, which is the
/// first merge multipass applies, or the pair entries and cold queue keys built as the ranks are
/// produced, so the two-tier queue needs no intermediate array to read back.
///
/// Either way the pair is looked up exactly once: both engines want that same value, multipass for
/// the minimum and the queue for the pair's rank and product.
struct SymbolSink<'a, const MULTIPASS: bool> {
    symbols: &'a mut Vec<u32>,
    entries: &'a mut Vec<Entry>,
    cold: &'a mut Vec<u64>,
    previous_symbol: u32,
    lowest_merge: u64,
}

impl<const MULTIPASS: bool> SymbolSink<'_, MULTIPASS> {
    // inlining here is very important
    #[inline(always)]
    fn push(&mut self, tables: &BpeTables, symbol: u32) {
        if self.previous_symbol != u32::MAX {
            let merge = tables.get_value(&self.previous_symbol, &symbol);
            if MULTIPASS {
                if merge < self.lowest_merge {
                    self.lowest_merge = merge;
                }
            } else {
                let index = self.entries.len() as u32;
                self.entries.push(Entry {
                    rank: (merge >> 32) as u32,
                    prod: merge as u32,
                    a: self.previous_symbol,
                    b: symbol,
                    l: index.wrapping_sub(1), // u32::MAX at index 0, which is NONE
                    r: index + 1,             // the final entry is patched in `convert`
                });
                if merge != u64::MAX {
                    self.cold.push((merge & 0xFFFF_FFFF_0000_0000) | index as u64);
                }
            }
        }
        self.previous_symbol = symbol;
        if MULTIPASS {
            self.symbols.push(symbol);
        }
    }
}

impl PipelineBPE {
    /// Converts one pretoken to internal IDs, returning the lowest-ranked adjacent pair when
    /// `MULTIPASS` and `u64::MAX` otherwise.
    ///
    /// Without `MULTIPASS` a pretoken of fewer than two ranks has no pairs and so no entries; its
    /// single rank is left in `symbols` instead, and the queue engine sees an empty entry list.
    pub(super) fn convert<const MULTIPASS: bool>(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
        entries: &mut Vec<Entry>,
        cold: &mut Vec<u64>,
    ) -> u64 {
        symbols.clear();
        entries.clear();
        cold.clear();
        // a word never has more ranks than bytes, so one reserve covers every push
        if MULTIPASS {
            symbols.reserve(sequence.len());
        } else {
            entries.reserve(sequence.len());
            cold.reserve(sequence.len());
        }
        let mut sink = SymbolSink::<MULTIPASS> {
            symbols,
            entries,
            cold,
            previous_symbol: u32::MAX,
            lowest_merge: u64::MAX,
        };
        if self.affixes.is_some() {
            self.convert_affixed(sequence, &mut sink);
        } else if matches!(self.atoms, Atoms::Bytes { .. }) {
            self.convert_bytes(sequence.as_bytes(), &mut sink);
        } else {
            self.convert_chars(sequence, &mut sink);
        }
        let last = sink.previous_symbol;
        let lowest = sink.lowest_merge;
        if !MULTIPASS {
            match entries.last_mut() {
                Some(entry) => entry.r = u32::MAX, // NONE: nothing right of the final pair
                None if last != u32::MAX => symbols.push(last),
                None => {}
            }
        }
        lowest
    }

    fn convert_bytes<const MULTIPASS: bool>(
        &self,
        bytes: &[u8],
        sink: &mut SymbolSink<'_, MULTIPASS>,
    ) {
        let byte_symbols = &self.tables.byte_internal[..];
        let mut pos = 0usize;
        while pos < bytes.len() {
            // An ASCII character is exactly one symbol whether or not it folds, so this loop needs
            // no fold branch. `get` gives the bounds check and the byte in one step.
            while let Some(&ascii) = bytes.get(pos) {
                if ascii >= 0x80 {
                    break;
                }
                sink.push(&self.tables, self.tables.fold.get_ascii(ascii));
                pos += 1;
            }
            if pos >= bytes.len() {
                break; // the run ran to the end rather than stopping on a lead byte
            }
            let lead = bytes.at(pos);
            let char_len = UTF8_LEN[lead as usize] as usize;
            let folded = self.tables.fold.get_bytes(bytes, pos, lead, char_len);
            if folded != u32::MAX {
                sink.push(&self.tables, folded);
            } else {
                for offset in 0..char_len {
                    let byte = bytes.at(pos + offset) as usize;
                    sink.push(&self.tables, byte_symbols.at(byte));
                }
            }
            pos += char_len;
        }
    }

    /// Character-level conversion, for models without a byte-level pretokenizer: every vocab token
    /// of one character has a fold entry, so there is no byte decomposition to do here.
    fn convert_chars<const MULTIPASS: bool>(
        &self,
        sequence: &str,
        sink: &mut SymbolSink<'_, MULTIPASS>,
    ) {
        let Atoms::Chars {
            byte_fallback,
            unk_token,
            fuse_unk,
        } = &self.atoms
        else {
            return;
        };
        let mut in_unk_run = false;
        for character in sequence.chars() {
            let symbol = self.tables.fold.get_char(character);
            if symbol != u32::MAX {
                in_unk_run = false;
                sink.push(&self.tables, symbol);
                continue;
            }
            if let Some(fallback) = byte_fallback {
                let mut buf = [0u8; 4];
                for &byte in character.encode_utf8(&mut buf).as_bytes() {
                    sink.push(&self.tables, fallback.at(byte as usize));
                }
                in_unk_run = false;
                continue;
            }
            if let Some(unk) = unk_token {
                // with fuse_unk the run already emitted its unk, so this character adds nothing
                if !(*fuse_unk && in_unk_run) {
                    sink.push(&self.tables, *unk);
                }
                in_unk_run = true;
            }
        }
    }
}

impl PipelineBPE {
    /// Slow path for models that decorate their atoms: `continuing_subword_prefix` on every
    /// character but the first, `end_of_word_suffix` on the last. The decorated form is assembled
    /// in a stack buffer and looked up in the vocab, which costs a hash per character -- these
    /// models are rare enough that it is not worth a second fold table to avoid it.
    fn convert_affixed<const MULTIPASS: bool>(
        &self,
        sequence: &str,
        sink: &mut SymbolSink<'_, MULTIPASS>,
    ) {
        let Some(affixes) = self.affixes.as_ref() else {
            return;
        };
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
                .and_then(|token| self.vocab.token_to_id(token))
                .and_then(|external| affixes.to_internal.get(external as usize).copied())
                .filter(|&symbol| symbol != u32::MAX);
            match symbol {
                Some(symbol) => sink.push(&self.tables, symbol),
                None => self.push_unknown(character, sink),
            }
        }
    }

    /// A character with no atom of its own: bytes if the model has `byte_fallback`, else `unk`.
    fn push_unknown<const MULTIPASS: bool>(
        &self,
        character: char,
        sink: &mut SymbolSink<'_, MULTIPASS>,
    ) {
        match &self.atoms {
            Atoms::Bytes { .. } => {
                let mut buf = [0u8; 4];
                for &byte in character.encode_utf8(&mut buf).as_bytes() {
                    sink.push(&self.tables, self.tables.byte_internal.at(byte as usize));
                }
            }
            Atoms::Chars {
                byte_fallback,
                unk_token,
                ..
            } => {
                if let Some(fallback) = byte_fallback {
                    let mut buf = [0u8; 4];
                    for &byte in character.encode_utf8(&mut buf).as_bytes() {
                        sink.push(&self.tables, fallback.at(byte as usize));
                    }
                } else if let Some(unk) = unk_token {
                    sink.push(&self.tables, *unk);
                }
            }
        }
    }
}
