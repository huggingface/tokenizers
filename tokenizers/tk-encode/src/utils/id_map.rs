use std::collections::BTreeMap;

/// Direct indexing for compact ID ranges, with storage proportional to the entry count
/// even when a vocabulary assigns widely separated IDs.
#[derive(Clone, Debug)]
pub(crate) enum IdMap<T> {
    Dense(Box<[T]>),
    Sparse(BTreeMap<u32, T>),
}

impl<T> IdMap<T> {
    pub(crate) fn new(max_id: Option<u32>, entries: usize, hole: T) -> Self
    where
        T: Clone,
    {
        let slots = max_id.map_or(0, |id| u64::from(id) + 1);
        match usize::try_from(slots) {
            Ok(slots) if slots <= entries.saturating_mul(2) => {
                Self::Dense(vec![hole; slots].into_boxed_slice())
            }
            _ => Self::Sparse(BTreeMap::new()),
        }
    }

    pub(crate) fn insert(&mut self, id: u32, value: T) {
        match self {
            Self::Dense(slots) => slots[id as usize] = value,
            Self::Sparse(slots) => {
                slots.insert(id, value);
            }
        }
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<&T> {
        match self {
            Self::Dense(slots) => slots.get(id as usize),
            Self::Sparse(slots) => slots.get(&id),
        }
    }

    /// Ascending IDs. The dense representation also yields its hole values.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        let (dense, sparse) = match self {
            Self::Dense(slots) => (slots.as_ref(), None),
            Self::Sparse(slots) => (&[][..], Some(slots)),
        };
        dense
            .iter()
            .enumerate()
            .map(|(id, value)| (id as u32, value))
            .chain(sparse.into_iter().flatten().map(|(&id, value)| (id, value)))
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(slots) => slots.len(),
            Self::Sparse(slots) => slots.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IdMap;

    #[test]
    fn dense_and_sparse_tables_keep_the_last_value_for_an_id() {
        for high in [3, 128] {
            let mut slots = IdMap::new(Some(high), 2, None);
            slots.insert(high, Some("old"));
            slots.insert(0, Some(""));
            slots.insert(high, Some("new"));

            let occupied: Vec<_> = slots
                .iter()
                .filter_map(|(id, value)| value.map(|value| (id, value)))
                .collect();
            assert_eq!(occupied, [(0, ""), (high, "new")]);
            assert_eq!(slots.get(high).copied().flatten(), Some("new"));
            assert_eq!(slots.get(1).copied().flatten(), None);
        }
    }
}
