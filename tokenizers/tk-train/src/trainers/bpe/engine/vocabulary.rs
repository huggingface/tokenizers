//! Vocabulary identity and canonical output strings, independent of position storage.
use super::{BpeTrainer, WORD_SEPARATOR_ID};
use crate::progress::{TrainingProgress, WorkProgress};
use crate::trainers::bpe::word_counts::WordCountsView;
use ahash::RandomState;
use compact_str::CompactString;
use indexmap::IndexSet;
use rayon::prelude::*;
use tk_encode::{
    Result,
    models::bpe::{Merges, Pair, Vocab},
};

pub(super) struct Vocabulary {
    // Append-only insertion indices are token IDs. Store each string once while
    // supporting both text lookup and direct lookup by ID with the same hasher.
    tokens: IndexSet<CompactString, RandomState>,
    active: Vec<bool>,
    prefix: Option<String>,
    suffix: Option<String>,
    plain_ids_resolved: bool,
}
pub(super) struct MergeToken {
    pub(super) existing_id: Option<u32>,
    text: CompactString,
}
pub(super) struct MergeIdentity {
    pub(super) id: u32,
    pub(super) reused_active_id: bool,
}
pub(super) struct InitialTokenIds {
    characters: Vec<u32>,
    decorated: Vec<[u32; 3]>,
    prefix: bool,
    suffix: bool,
    complete_alphabet: bool,
    // Conservative bound on IDs scan_symbols can emit, including decorations
    // and unused single-character IDs, but excluding the separator sentinel.
    maximum_initial_id: u32,
}
impl Vocabulary {
    pub(super) fn initialize(
        trainer: &BpeTrainer,
        word_counts: WordCountsView<'_>,
        workers: usize,
        progress: &TrainingProgress,
        retained_alphabet: &mut Option<Vec<char>>,
    ) -> Result<Self> {
        let mut vocabulary = Self {
            tokens: IndexSet::with_capacity_and_hasher(trainer.vocab_size, RandomState::default()),
            active: Vec::new(),
            prefix: trainer.continuing_subword_prefix.clone(),
            suffix: trainer.end_of_word_suffix.clone(),
            plain_ids_resolved: false,
        };
        for token in &trainer.special_tokens {
            vocabulary.intern(token.content.as_str())?;
        }
        let work = progress.stage("Compute alphabet", word_counts.len());
        if trainer.limit_alphabet.is_some() {
            // Preserve the existing frequency-tie selector for limited alphabets.
            let characters =
                retained_alphabet.get_or_insert_with(|| trainer.select_alphabet(word_counts));
            for &character in characters.iter() {
                let mut utf8 = [0; 4];
                vocabulary.intern(character.encode_utf8(&mut utf8))?;
            }
            work.complete(word_counts.len());
        } else {
            let words: Vec<_> = word_counts.keys().collect();
            let chunk = words.len().div_ceil(workers).max(1);
            let bitmaps: Vec<_> = words
                .par_chunks(chunk)
                .map(|chunk| {
                    let mut present = vec![0_u64; 0x110000 / 64];
                    for word in chunk {
                        for character in word.chars() {
                            let codepoint = character as usize;
                            present[codepoint / 64] |= 1_u64 << (codepoint % 64);
                        }
                    }
                    work.complete(chunk.len());
                    present
                })
                .collect();
            let mut present = vec![0_u64; 0x110000 / 64];
            for bitmap in bitmaps {
                for (merged, bits) in present.iter_mut().zip(bitmap) {
                    *merged |= bits;
                }
            }
            let observed = present.clone();
            for &character in &trainer.initial_alphabet {
                let codepoint = character as usize;
                present[codepoint / 64] |= 1_u64 << (codepoint % 64);
            }
            for (word, mut bits) in present.into_iter().enumerate() {
                while bits != 0 {
                    let codepoint = word * 64 + bits.trailing_zeros() as usize;
                    let character = char::from_u32(codepoint as u32)
                        .expect("alphabet bits come from valid characters");
                    let mut utf8 = [0; 4];
                    let id = vocabulary.intern(character.encode_utf8(&mut utf8))?;
                    vocabulary.active[id as usize] =
                        observed[codepoint / 64] & (1_u64 << (codepoint % 64)) != 0;
                    bits &= bits - 1;
                }
            }
            vocabulary.plain_ids_resolved = true;
        }
        Ok(vocabulary)
    }
    fn intern(&mut self, text: &str) -> Result<u32> {
        if let Some(id) = self.tokens.get_index_of(text) {
            return Ok(id as u32);
        }
        self.insert_new_token(CompactString::from(text))
    }
    // The caller has established that this text is absent. No vocabulary
    // mutation can intervene before this insertion on the coordinator.
    fn insert_new_token(&mut self, token: CompactString) -> Result<u32> {
        let id = u32::try_from(self.tokens.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
        if id == WORD_SEPARATOR_ID {
            return Err("BPE token ID collides with the word separator".into());
        }
        let (index, inserted) = self.tokens.insert_full(token);
        debug_assert!(inserted && index == id as usize);
        self.active.push(false);
        Ok(id)
    }
    pub(super) fn initial_ids(
        &mut self,
        word_counts: WordCountsView<'_>,
        work: &WorkProgress,
    ) -> Result<InitialTokenIds> {
        let mut ids = InitialTokenIds {
            characters: vec![WORD_SEPARATOR_ID; 0x110000],
            decorated: Vec::new(),
            prefix: self.prefix.as_deref().is_some_and(|p| !p.is_empty()),
            suffix: self.suffix.as_deref().is_some_and(|s| !s.is_empty()),
            complete_alphabet: self.plain_ids_resolved,
            maximum_initial_id: 0,
        };
        for (id, token) in self.tokens.iter().enumerate() {
            let mut chars = token.chars();
            if let Some(character) = chars.next()
                && chars.next().is_none()
            {
                ids.characters[character as usize] = id as u32;
                ids.maximum_initial_id = ids.maximum_initial_id.max(id as u32);
            }
        }
        if ids.prefix || ids.suffix {
            ids.decorated
                .resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);
        }
        if !ids.prefix && !ids.suffix && self.plain_ids_resolved {
            work.complete(word_counts.len());
            return Ok(ids);
        }
        self.active.fill(false);
        let mut decorated = String::new();
        // Allocate decorated IDs in the input view's traversal before sorting weighted words.
        for (index, word) in word_counts.keys().enumerate() {
            for (byte, character) in word.char_indices() {
                let plain_id = ids.characters[character as usize];
                if plain_id == WORD_SEPARATOR_ID {
                    continue;
                }
                let flags = usize::from(ids.prefix && byte != 0)
                    | (usize::from(ids.suffix && byte + character.len_utf8() == word.len()) << 1);
                let id = if flags == 0 {
                    plain_id
                } else {
                    let cached = ids.decorated[plain_id as usize][flags - 1];
                    if cached != WORD_SEPARATOR_ID {
                        cached
                    } else {
                        decorated.clear();
                        if flags & 1 != 0 {
                            decorated.push_str(
                                self.prefix
                                    .as_deref()
                                    .expect("prefix flag requires a prefix"),
                            );
                        }
                        decorated.push(character);
                        if flags & 2 != 0 {
                            decorated.push_str(
                                self.suffix
                                    .as_deref()
                                    .expect("suffix flag requires a suffix"),
                            );
                        }
                        let id = self.intern(&decorated)?;
                        ids.decorated[plain_id as usize][flags - 1] = id;
                        id
                    }
                };
                ids.maximum_initial_id = ids.maximum_initial_id.max(id);
                self.active[id as usize] = true;
            }
            if (index + 1) % 1024 == 0 {
                work.complete(1024);
            }
        }
        work.complete(word_counts.len() % 1024);
        Ok(ids)
    }
    pub(super) fn initial_spans(&self) -> Vec<u64> {
        self.active
            .iter()
            .map(|&active| u64::from(active))
            .collect()
    }
    pub(super) fn reuses_active_id(&self, token: &MergeToken) -> bool {
        token.existing_id.is_some_and(|id| self.active[id as usize])
    }
    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }
    pub(super) fn merge_token(&self, pair: Pair) -> MergeToken {
        let left = self.tokens[pair.0 as usize].as_str();
        let right = self.tokens[pair.1 as usize].as_str();
        let right = self
            .prefix
            .as_deref()
            .and_then(|p| right.strip_prefix(p))
            .unwrap_or(right);
        let mut text = CompactString::with_capacity(left.len() + right.len());
        text.push_str(left);
        text.push_str(right);
        MergeToken {
            existing_id: self.tokens.get_index_of(text.as_str()).map(|id| id as u32),
            text,
        }
    }
    pub(super) fn resolve_merge(&mut self, token: MergeToken) -> Result<MergeIdentity> {
        let id = match token.existing_id {
            Some(id) => id,
            // PERF: merge_token already checked this key. Consume its complete
            // string instead of probing again and copying another owned string.
            None => self.insert_new_token(token.text)?,
        };
        let reused_active_id = self.active[id as usize];
        self.active[id as usize] = true;
        Ok(MergeIdentity {
            id,
            reused_active_id,
        })
    }
    pub(super) fn into_model_parts(self, merges: Vec<Pair>) -> (Vocab, Merges) {
        let merges = merges
            .into_iter()
            .map(|(left, right)| {
                (
                    self.tokens[left as usize].to_string(),
                    self.tokens[right as usize].to_string(),
                )
            })
            .collect();
        (
            self.tokens
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), id as u32))
                .collect(),
            merges,
        )
    }
}
impl InitialTokenIds {
    pub(super) fn compact_pair_keys(&self) -> bool {
        self.maximum_initial_id <= u16::MAX as u32
    }
    /// Interpret filtering and decorations against original UTF-8 coordinates.
    /// The plain path chooses its loop once and avoids per-symbol affix checks.
    #[inline]
    pub(super) fn scan_symbols(
        &self,
        word: &str,
        byte_offset: usize,
        mut emit: impl FnMut(u32) -> std::ops::ControlFlow<()>,
    ) {
        use std::ops::ControlFlow;
        if self.plain() {
            for character in word[byte_offset..].chars() {
                if let Some(id) = self.plain_id(character)
                    && emit(id).is_break()
                {
                    break;
                }
            }
        } else {
            for (offset, character) in word[byte_offset..].char_indices() {
                let byte = byte_offset + offset;
                if let Some(id) = self.id(
                    character,
                    byte == 0,
                    byte + character.len_utf8() == word.len(),
                ) && emit(id) == ControlFlow::Break(())
                {
                    break;
                }
            }
        }
    }
    pub(super) fn symbol_count(&self, text: &str) -> usize {
        if self.complete_alphabet {
            text.chars().count()
        } else {
            text.chars()
                .filter(|&ch| self.characters[ch as usize] != WORD_SEPARATOR_ID)
                .count()
        }
    }
    pub(super) fn plain(&self) -> bool {
        !self.prefix && !self.suffix
    }
    pub(super) fn plain_id(&self, character: char) -> Option<u32> {
        let id = self.characters[character as usize];
        (id != WORD_SEPARATOR_ID).then_some(id)
    }
    pub(super) fn id(&self, character: char, first: bool, last: bool) -> Option<u32> {
        let plain = self.characters[character as usize];
        if plain == WORD_SEPARATOR_ID {
            return None;
        }
        let flags = usize::from(self.prefix && !first) | (usize::from(self.suffix && last) << 1);
        Some(if flags == 0 {
            plain
        } else {
            self.decorated[plain as usize][flags - 1]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_keys_follow_emitted_ids_including_decorations() {
        use std::ops::ControlFlow;
        for reserved_len in [u16::MAX as usize, u16::MAX as usize + 1] {
            let mut vocabulary = Vocabulary {
                tokens: IndexSet::with_hasher(RandomState::default()),
                active: Vec::new(),
                prefix: None,
                suffix: None,
                plain_ids_resolved: true,
            };
            assert_eq!(vocabulary.intern("a").unwrap(), 0);
            for index in 1..reserved_len {
                vocabulary.intern(&format!("<reserved{index}>")).unwrap();
            }
            let words = ahash::AHashMap::from_iter([(CompactString::from("aa"), 1_u64)]);
            let progress =
                TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                    .unwrap();
            let work = progress.stage("Initial IDs", words.len());
            let ids = vocabulary
                .initial_ids(WordCountsView::from_map(&words), &work)
                .unwrap();
            // Reserved vocabulary size alone must not force full-width records.
            assert!(ids.compact_pair_keys());
            vocabulary.prefix = Some("##".into());
            let work = progress.stage("Decorated IDs", words.len());
            let ids = vocabulary
                .initial_ids(WordCountsView::from_map(&words), &work)
                .unwrap();
            let mut emitted = Vec::new();
            ids.scan_symbols("aa", 0, |id| {
                emitted.push(id);
                ControlFlow::Continue(())
            });
            assert_eq!(emitted, [0, reserved_len as u32]);
            assert_eq!(ids.compact_pair_keys(), reserved_len <= u16::MAX as usize);
        }
    }
}
