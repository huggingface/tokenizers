//! A trie over label sequences, in the double-array layout Darts uses: one `u32` per slot.
//!
//! A node is a slot, and its children sit at `slot ^ offset ^ label`. A walk for a byte is
//! therefore one load and one label compare, with no child list to scan -- which is what the
//! Viterbi's per-position search pays for on every word the word cache misses.

use std::collections::VecDeque;

/// Set on the unit that holds a value, so a value cell can never pass for a label.
const VALUE: u32 = 1 << 31;

/// One slot of the array.
#[derive(Clone, Copy, Default)]
struct Unit(u32);

impl Unit {
    #[inline]
    fn label(self) -> usize {
        (self.0 & (VALUE | 0xFF)) as usize
    }

    #[inline]
    fn has_value(self) -> bool {
        (self.0 >> 8) & 1 == 1
    }

    /// The slot this node's children XOR from.
    #[inline]
    fn offset(self) -> usize {
        (self.0 >> 10) as usize
    }

    /// The label, the value bit and the offset, in one unit.
    #[inline]
    fn new(label: usize, has_value: bool, offset: usize) -> Self {
        Self(label as u32 | (has_value as u32) << 8 | ((offset as u32) << 10))
    }
}

pub struct TrieBuilder<Label> {
    nodes: Vec<BuildNode<Label>>,
}

impl<Label: Copy + Eq> Default for TrieBuilder<Label> {
    fn default() -> Self {
        Self {
            nodes: vec![BuildNode::default()],
        }
    }
}

impl<Label: Copy + Eq> TrieBuilder<Label> {
    pub fn push_with_value(&mut self, element: &[Label], value: u32) {
        let mut node_index = 0;
        for &label in element {
            let child_index = self.nodes[node_index]
                .children
                .iter()
                .find_map(|&(edge, child)| (edge == label).then_some(child));
            node_index = match child_index {
                Some(index) => index,
                None => {
                    let index = self.nodes.len();
                    self.nodes.push(BuildNode::default());
                    self.nodes[node_index].children.push((label, index));
                    index
                }
            };
        }
        self.nodes[node_index].value = Some(value);
    }

    pub fn build(self) -> Trie<Label>
    where
        Label: Copy + Ord + Into<usize>,
    {
        let nodes = self.nodes;
        // Slot 0 is the root and is always taken, so a base search never lands on it.
        let mut units: Vec<Unit> = vec![Unit::default()];
        let mut used: Vec<bool> = vec![true];
        // A base is claimed by one node only. Two nodes sharing a base would each answer for the
        // other's labels, since a slot and the label it was reached by determine the base.
        let mut bases: Vec<bool> = vec![false];
        // The lowest slot still worth trying as a base: every placement below it lost.
        let mut next = 1usize;

        // Breadth first, so a node is placed before its children need slots.
        let mut queue = VecDeque::from([(0usize, 0usize, 0usize)]);
        while let Some((index, slot, label)) = queue.pop_front() {
            let node = &nodes[index];
            let mut children: Vec<(usize, usize)> = node
                .children
                .iter()
                .map(|&(child_label, child)| (child_label.into(), child))
                .collect();
            children.sort_unstable();
            debug_assert!(
                children.iter().all(|&(child_label, _)| child_label < 0x100),
                "a label above 255 does not fit a unit"
            );

            // The smallest base no other node holds and whose slots -- the value cell and every
            // child -- are free. A base belongs to one node: a slot and the label it was reached
            // by determine the base, so two nodes sharing one would each answer for the other's
            // labels.
            let mut base = next;
            loop {
                let free = !is_used(&bases, base)
                    && (node.value.is_none() || !is_used(&used, base))
                    && children
                        .iter()
                        .all(|&(child, _)| !is_used(&used, base ^ child));
                if free {
                    break;
                }
                base += 1;
            }
            next = next.max(base);

            grow(&mut units, &mut used, base + 1);
            grow_bases(&mut bases, base + 1);
            bases[base] = true;
            units[slot] = Unit::new(label, node.value.is_some(), slot ^ base);
            if let Some(value) = node.value {
                units[base] = Unit(value | VALUE);
                used[base] = true;
            }
            for &(child_label, child) in &children {
                let child_slot = base ^ child_label;
                grow(&mut units, &mut used, child_slot + 1);
                used[child_slot] = true;
                queue.push_back((child, child_slot, child_label));
            }
        }

        Trie {
            units,
            marker: std::marker::PhantomData,
        }
    }
}

#[inline]
fn is_used(used: &[bool], slot: usize) -> bool {
    used.get(slot).copied().unwrap_or(false)
}

/// Make sure `slot` exists in both arrays.
#[inline]
fn grow(units: &mut Vec<Unit>, used: &mut Vec<bool>, len: usize) {
    if units.len() < len {
        units.resize(len, Unit::default());
        used.resize(len, false);
    }
}

#[inline]
fn grow_bases(bases: &mut Vec<bool>, len: usize) {
    if bases.len() < len {
        bases.resize(len, false);
    }
}

#[derive(Clone)]
pub struct Trie<Label> {
    units: Vec<Unit>,
    marker: std::marker::PhantomData<Label>,
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Self {
            units: vec![Unit::default()],
            marker: std::marker::PhantomData,
        }
    }
}

impl<Label: Copy + Ord + Into<usize>> Trie<Label> {
    pub fn common_prefix_search_values<T>(&self, iterator: T) -> TrieValueIterator<'_, Label, T>
    where
        T: Iterator<Item = Label>,
    {
        let root = self.units.first().copied().unwrap_or_default();
        TrieValueIterator {
            trie: self,
            base: root.offset(),
            length: 0,
            iterator,
            finished: false,
        }
    }
}

pub struct TrieValueIterator<'a, Label, T> {
    trie: &'a Trie<Label>,
    /// The slot the next label XORs from.
    base: usize,
    length: usize,
    iterator: T,
    finished: bool,
}

impl<Label, T> Iterator for TrieValueIterator<'_, Label, T>
where
    Label: Copy + Ord + Into<usize>,
    T: Iterator<Item = Label>,
{
    type Item = (usize, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let Some(label) = self.iterator.next() else {
                self.finished = true;
                return None;
            };
            let label = label.into();
            self.length += 1;
            let slot = self.base ^ label;
            let Some(unit) = self.trie.units.get(slot).copied() else {
                self.finished = true;
                return None;
            };
            if unit.label() != label {
                self.finished = true;
                return None;
            }
            self.base = slot ^ unit.offset();
            if unit.has_value() {
                let Some(value) = self.trie.units.get(self.base).copied() else {
                    self.finished = true;
                    return None;
                };
                return Some((self.length, value.0 & !VALUE));
            }
        }
    }
}

struct BuildNode<Label> {
    value: Option<u32>,
    children: Vec<(Label, usize)>,
}

impl<Label> Default for BuildNode<Label> {
    fn default() -> Self {
        Self {
            value: None,
            children: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    trait PrefixLookup {
        fn matches(&self, query: &[u8], output: &mut Vec<(usize, u32)>);
        fn estimated_storage(&self) -> usize;
    }

    impl PrefixLookup for Trie<u8> {
        fn matches(&self, query: &[u8], output: &mut Vec<(usize, u32)>) {
            output.clear();
            output.extend(self.common_prefix_search_values(query.iter().copied()));
        }

        fn estimated_storage(&self) -> usize {
            size_of::<Trie<u8>>() + self.units.capacity() * size_of::<Unit>()
        }
    }

    #[derive(Default)]
    struct HashNode {
        value: Option<u32>,
        children: AHashMap<u8, HashNode>,
    }

    #[derive(Default)]
    struct HashTrie {
        root: HashNode,
    }

    impl HashTrie {
        fn insert(&mut self, key: &[u8], value: u32) {
            let mut node = &mut self.root;
            for label in key {
                node = node.children.entry(*label).or_default();
            }
            node.value = Some(value);
        }

        fn from_vocab(vocab: &[Vec<u8>]) -> Self {
            let mut trie = Self::default();
            for (id, token) in vocab.iter().enumerate() {
                trie.insert(token, id as u32);
            }
            trie
        }

        fn child_storage(node: &HashNode) -> usize {
            node.children.capacity() * size_of::<(u8, HashNode)>()
                + node
                    .children
                    .values()
                    .map(Self::child_storage)
                    .sum::<usize>()
        }
    }

    impl PrefixLookup for HashTrie {
        fn matches(&self, query: &[u8], output: &mut Vec<(usize, u32)>) {
            output.clear();
            let mut node = &self.root;
            for (index, label) in query.iter().enumerate() {
                let Some(child) = node.children.get(label) else {
                    break;
                };
                node = child;
                if let Some(value) = node.value {
                    output.push((index + 1, value));
                }
            }
        }

        fn estimated_storage(&self) -> usize {
            size_of::<HashTrie>() + Self::child_storage(&self.root)
        }
    }

    #[derive(Default)]
    struct VectorTrie {
        root: VectorNode,
    }

    #[derive(Default)]
    struct VectorNode {
        value: Option<u32>,
        children: Vec<(u8, VectorNode)>,
    }

    impl VectorTrie {
        fn insert(&mut self, key: &[u8], value: u32) {
            let mut node = &mut self.root;
            for &label in key {
                let index = node.children.iter().position(|(edge, _)| *edge == label);
                let index = match index {
                    Some(index) => index,
                    None => {
                        node.children.push((label, VectorNode::default()));
                        node.children.len() - 1
                    }
                };
                node = &mut node.children[index].1;
            }
            node.value = Some(value);
        }

        fn sort_children(node: &mut VectorNode) {
            node.children.sort_unstable_by_key(|(label, _)| *label);
            for (_, child) in &mut node.children {
                Self::sort_children(child);
            }
        }

        fn from_vocab(vocab: &[Vec<u8>]) -> Self {
            let mut trie = Self::default();
            for (id, token) in vocab.iter().enumerate() {
                trie.insert(token, id as u32);
            }
            Self::sort_children(&mut trie.root);
            trie
        }

        fn child_storage(node: &VectorNode) -> usize {
            node.children.capacity() * size_of::<(u8, VectorNode)>()
                + node
                    .children
                    .iter()
                    .map(|(_, child)| Self::child_storage(child))
                    .sum::<usize>()
        }
    }

    impl PrefixLookup for VectorTrie {
        fn matches(&self, query: &[u8], output: &mut Vec<(usize, u32)>) {
            output.clear();
            let mut node = &self.root;
            for (index, &label) in query.iter().enumerate() {
                let Ok(child_index) = node
                    .children
                    .binary_search_by_key(&label, |(edge, _)| *edge)
                else {
                    break;
                };
                node = &node.children[child_index].1;
                if let Some(value) = node.value {
                    output.push((index + 1, value));
                }
            }
        }

        fn estimated_storage(&self) -> usize {
            size_of::<VectorTrie>() + Self::child_storage(&self.root)
        }
    }

    fn test_vocab(count: usize) -> Vec<Vec<u8>> {
        let prefixes = ["a", "pre", "un", "##", "inter", "漢", "こんにちは", "x"];
        (0..count)
            .map(|index| {
                format!(
                    "{}{:05x}{}",
                    prefixes[index % prefixes.len()],
                    index,
                    ["", "ing", "ed", "s"][index % 4]
                )
                .into_bytes()
            })
            .collect()
    }

    fn measure_lookup<T: PrefixLookup>(
        trie: &T,
        queries: &[Vec<u8>],
        rounds: usize,
    ) -> (Duration, usize) {
        let mut output = Vec::with_capacity(16);
        let mut checksum = 0usize;
        let start = Instant::now();
        for _ in 0..rounds {
            for query in queries {
                trie.matches(query, &mut output);
                checksum += output
                    .iter()
                    .map(|&(length, value)| length + value as usize)
                    .sum::<usize>();
            }
        }
        (start.elapsed(), checksum)
    }

    #[test]
    fn common_prefix_search_values_returns_lengths_and_values() {
        let mut builder = TrieBuilder::default();
        builder.push_with_value(b"a", 3);
        builder.push_with_value(b"ab", 7);
        builder.push_with_value(b"abc", 11);
        let trie = builder.build();

        assert_eq!(
            trie.common_prefix_search_values(b"abcd".iter().copied())
                .collect::<Vec<_>>(),
            vec![(1, 3), (2, 7), (3, 11)]
        );
        assert_eq!(
            trie.common_prefix_search_values(b"ax".iter().copied())
                .collect::<Vec<_>>(),
            vec![(1, 3)]
        );
        assert_eq!(
            trie.common_prefix_search_values(b"x".iter().copied())
                .collect::<Vec<_>>(),
            Vec::<(usize, u32)>::new()
        );
    }

    #[test]
    fn prefix_iterator_stays_finished_after_a_miss() {
        let mut builder = TrieBuilder::default();
        builder.push_with_value(b"a", 1);
        builder.push_with_value(b"ab", 2);
        let trie = builder.build();

        let mut iter = trie.common_prefix_search_values(b"axb".iter().copied());
        assert_eq!(iter.next(), Some((1, 1)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut root_miss = trie.common_prefix_search_values(b"xab".iter().copied());
        assert_eq!(root_miss.next(), None);
        assert_eq!(root_miss.next(), None);
    }

    #[test]
    fn duplicate_keys_replace_values_and_empty_keys_are_not_matches() {
        let mut builder = TrieBuilder::default();
        builder.push_with_value(b"", 1);
        builder.push_with_value(b"token", 3);
        builder.push_with_value(b"token", 7);
        let trie = builder.build();

        assert_eq!(
            trie.common_prefix_search_values(b"tokenizer".iter().copied())
                .collect::<Vec<_>>(),
            vec![(5, 7)]
        );
        assert_eq!(
            trie.common_prefix_search_values(std::iter::empty())
                .collect::<Vec<_>>(),
            Vec::<(usize, u32)>::new()
        );
    }

    #[test]
    fn vector_layouts_match_hash_trie() {
        let vocab = test_vocab(2_000);
        let queries: Vec<_> = vocab
            .iter()
            .map(|token| {
                let mut query = token.clone();
                query.extend_from_slice(b"tail");
                query
            })
            .collect();

        let hash_trie = HashTrie::from_vocab(&vocab);
        let vector_trie = VectorTrie::from_vocab(&vocab);
        let mut flat_builder = TrieBuilder::default();
        for (id, token) in vocab.iter().enumerate() {
            flat_builder.push_with_value(token, id as u32);
        }
        let flat_trie = flat_builder.build();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        for query in queries {
            hash_trie.matches(&query, &mut expected);
            vector_trie.matches(&query, &mut actual);
            assert_eq!(actual, expected);
            flat_trie.matches(&query, &mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[ignore = "run explicitly with --release --ignored --nocapture for layout timings"]
    fn benchmark_trie_layouts() {
        for vocab_size in [2_000, 20_000, 80_000] {
            let vocab = test_vocab(vocab_size);
            let queries: Vec<_> = vocab
                .iter()
                .step_by(2)
                .map(|token| {
                    let mut query = token.clone();
                    query.extend_from_slice(b"tail");
                    query
                })
                .collect();
            let rounds = (1_000_000 / queries.len()).max(1);

            let start = Instant::now();
            let hash_trie = HashTrie::from_vocab(&vocab);
            let hash_build = start.elapsed();

            let start = Instant::now();
            let vector_trie = VectorTrie::from_vocab(&vocab);
            let vector_build = start.elapsed();

            let start = Instant::now();
            let mut flat_builder = TrieBuilder::default();
            for (id, token) in vocab.iter().enumerate() {
                flat_builder.push_with_value(token, id as u32);
            }
            let flat_trie = flat_builder.build();
            let flat_build = start.elapsed();

            let (hash_lookup, hash_checksum) = measure_lookup(&hash_trie, &queries, rounds);
            let (vector_lookup, vector_checksum) = measure_lookup(&vector_trie, &queries, rounds);
            let (flat_lookup, flat_checksum) = measure_lookup(&flat_trie, &queries, rounds);
            assert_eq!(hash_checksum, vector_checksum);
            assert_eq!(hash_checksum, flat_checksum);

            println!(
                "trie layouts: vocab={} queries={} rounds={rounds}",
                vocab.len(),
                queries.len()
            );
            println!(
                "hash: build={:.3}s lookup={:.3}s estimated_payload_bytes={}",
                hash_build.as_secs_f64(),
                hash_lookup.as_secs_f64(),
                hash_trie.estimated_storage()
            );
            println!(
                "sorted-vec: build={:.3}s lookup={:.3}s estimated_payload_bytes={}",
                vector_build.as_secs_f64(),
                vector_lookup.as_secs_f64(),
                vector_trie.estimated_storage()
            );
            println!(
                "flat-arena: build={:.3}s lookup={:.3}s estimated_payload_bytes={}",
                flat_build.as_secs_f64(),
                flat_lookup.as_secs_f64(),
                flat_trie.estimated_storage()
            );
        }
    }
}
