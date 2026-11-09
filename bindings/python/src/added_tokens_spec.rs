use std::collections::HashMap;

use serde_json::Value;
use tk::vocab::bucket_added_vocabulary::AddedToken;

use crate::added_token;

/// The `added_tokens` array of a `tokenizer.json`, with a lookup by content for [`Self::add`].
///
/// Loading assigns the real ids again, in `id` order, so an `id` in the array is only a sort key.
/// The entries are checked at `compile()`.
#[derive(Clone)]
pub(crate) struct AddedTokensSpec {
    /// Always an array.
    entries: Value,
    /// The position of each content in `entries`. When the file repeats a content, the first one.
    position_by_content: HashMap<String, usize>,
    /// One more than the highest `id` in `entries`, so a new token sorts after the others.
    next_sort_key: u64,
}

impl AddedTokensSpec {
    pub(crate) fn new(entries: Vec<Value>) -> Self {
        let mut position_by_content = HashMap::new();
        let mut next_sort_key = 0;
        for (position, entry) in entries.iter().enumerate() {
            if let Some(content) = entry["content"].as_str() {
                position_by_content
                    .entry(content.to_owned())
                    .or_insert(position);
            }
            if let Some(id) = entry["id"].as_u64() {
                next_sort_key = next_sort_key.max(id.saturating_add(1));
            }
        }
        Self {
            entries: Value::Array(entries),
            position_by_content,
            next_sort_key,
        }
    }

    /// The `added_tokens` array.
    pub(crate) fn json(&self) -> &Value {
        &self.entries
    }

    /// Adds `tokens`, or updates the flags of those with the same content, and returns how many
    /// it added or updated.
    pub(crate) fn add(&mut self, tokens: impl IntoIterator<Item = AddedToken>) -> usize {
        let Value::Array(entries) = &mut self.entries else {
            unreachable!("new only stores an array");
        };
        let mut added_or_updated = 0;
        for token in tokens {
            if token.content.is_empty() {
                continue;
            }
            match self.position_by_content.get(&token.content) {
                Some(&position) => {
                    let existing = &mut entries[position];
                    let updated = added_token::json_entry(token, existing["id"].clone());
                    if *existing == updated {
                        continue;
                    }
                    *existing = updated;
                }
                None => {
                    self.position_by_content
                        .insert(token.content.clone(), entries.len());
                    entries.push(added_token::json_entry(token, self.next_sort_key.into()));
                    self.next_sort_key += 1;
                }
            }
            added_or_updated += 1;
        }
        added_or_updated
    }
}
