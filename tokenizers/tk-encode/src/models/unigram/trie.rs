/// Labels that can be used as a direct 256-entry root lookup table. Only byte labels
/// benefit; every other label type keeps an empty table and uses the generic path.
pub(crate) trait RootLookup {
    fn table_size() -> usize {
        0
    }
    fn to_index(_label: Self) -> usize
    where
        Self: Sized,
    {
        0
    }
}
impl RootLookup for u8 {
    fn table_size() -> usize {
        256
    }
    fn to_index(label: Self) -> usize
    where
        Self: Sized,
    {
        label as usize
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
        Label: Ord + RootLookup + Into<usize>,
    {
        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut edges = Vec::new();
        for mut node in self.nodes {
            node.children.sort_unstable_by_key(|(label, _)| *label);
            let first_edge = edges.len();
            let edge_count = node.children.len();
            edges.extend(node.children);
            nodes.push(Node {
                value: node.value,
                first_edge,
                edge_count,
            });
        }
        // Direct root table for byte tries: avoids a binary/linear search on the root's
        // (usually large) edge list for every searched position.
        let table_size = Label::table_size();
        let mut root_table = vec![0usize; table_size];
        if table_size > 0 {
            let root = &nodes[0];
            for i in 0..root.edge_count {
                let (label, child) = edges[root.first_edge + i];
                root_table[Label::to_index(label)] = child;
            }
        }
        Trie {
            nodes,
            edges,
            root_table,
        }
    }
}

#[derive(Clone)]
pub struct Trie<Label> {
    nodes: Vec<Node>,
    edges: Vec<(Label, usize)>,
    root_table: Vec<usize>,
}

impl<Label: Copy + Ord + RootLookup + Into<usize>> Trie<Label> {
    pub fn common_prefix_search_values<T>(&self, iterator: T) -> TrieValueIterator<'_, Label, T>
    where
        T: Iterator<Item = Label>,
    {
        TrieValueIterator {
            trie: self,
            node_index: 0,
            length: 0,
            iterator,
            finished: false,
        }
    }
}

pub struct TrieValueIterator<'a, Label, T> {
    trie: &'a Trie<Label>,
    node_index: usize,
    length: usize,
    iterator: T,
    finished: bool,
}

impl<Label, T> Iterator for TrieValueIterator<'_, Label, T>
where
    Label: Copy + Ord + RootLookup + Into<usize>,
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
            self.length += 1;
            if self.node_index == 0 && !self.trie.root_table.is_empty() {
                let child = self.trie.root_table[Label::to_index(label)];
                if child == 0 {
                    self.finished = true;
                    return None;
                }
                self.node_index = child;
            } else {
                let node = &self.trie.nodes[self.node_index];
                let edges = &self.trie.edges[node.first_edge..node.first_edge + node.edge_count];
                let edge_index = if edges.len() <= 4 {
                    edges.iter().position(|(edge, _)| *edge == label)
                } else {
                    edges.binary_search_by_key(&label, |(edge, _)| *edge).ok()
                };
                let Some(edge_index) = edge_index else {
                    self.finished = true;
                    return None;
                };
                self.node_index = edges[edge_index].1;
            }
            if let Some(value) = self.trie.nodes[self.node_index].value {
                return Some((self.length, value));
            }
        }
    }
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Self {
            nodes: vec![Node::default()],
            edges: Vec::new(),
            root_table: Vec::new(),
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

#[derive(Clone, Default)]
struct Node {
    value: Option<u32>,
    first_edge: usize,
    edge_count: usize,
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
            size_of::<Trie<u8>>()
                + self.nodes.capacity() * size_of::<Node>()
                + self.edges.capacity() * size_of::<(u8, usize)>()
                + self.root_table.capacity() * size_of::<usize>()
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
