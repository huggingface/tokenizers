//! Turning a pretokenized word into merge ranks, which are then processed in `multipass` or
//! `two_tier_merge`.
use crate::models::bpe::pipeline_bpe::{AFFIX_BUF, Atoms, PipelineBPE};
use crate::models::bpe::tables::{At, BpeTables, UTF8_LEN};

/// Collects the converted ranks of a sequence.`TRACK_MIN` triggers
/// lowest-ranked adjacent pair computation as it will be the first merge multipass applies.
struct SymbolSink<'a, const TRACK_MIN: bool> {
    symbols: &'a mut Vec<u32>,
    previous_symbol: u32,
    lowest_merge: u64,
}

impl<const TRACK_MIN: bool> SymbolSink<'_, TRACK_MIN> {
    // inlining here is very important
    #[inline(always)]
    fn push(&mut self, tables: &BpeTables, symbol: u32) {
        if TRACK_MIN && self.previous_symbol != u32::MAX {
            let merge = tables.get_value(&self.previous_symbol, &symbol);
            if merge < self.lowest_merge {
                self.lowest_merge = merge;
            }
        }
        self.previous_symbol = symbol;
        self.symbols.push(symbol);
    }
}

impl PipelineBPE {
    /// Converts one pretoken to internal IDs, returning the lowest-ranked adjacent pair when `TRACK_MIN`
    /// and `u64::MAX` otherwise.
    pub(super) fn convert<const TRACK_MIN: bool>(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
    ) -> u64 {
        symbols.clear();
        symbols.reserve(sequence.len()); // a word never has more symbols than bytes
        let mut sink = SymbolSink::<TRACK_MIN> {
            symbols,
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
        sink.lowest_merge
    }

    fn convert_bytes<const TRACK_MIN: bool>(
        &self,
        bytes: &[u8],
        sink: &mut SymbolSink<'_, TRACK_MIN>,
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
    fn convert_chars<const TRACK_MIN: bool>(
        &self,
        sequence: &str,
        sink: &mut SymbolSink<'_, TRACK_MIN>,
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
    fn convert_affixed<const TRACK_MIN: bool>(
        &self,
        sequence: &str,
        sink: &mut SymbolSink<'_, TRACK_MIN>,
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
    fn push_unknown<const TRACK_MIN: bool>(
        &self,
        character: char,
        sink: &mut SymbolSink<'_, TRACK_MIN>,
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
