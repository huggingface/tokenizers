use std::hash::Hash;

#[derive(Default)]
pub struct TrieBuilder<Label> {
    trie: Trie<Label>,
}

impl<Label: Eq + Hash + Copy> TrieBuilder<Label> {
    pub fn push(&mut self, element: &[Label]) {
        self.trie.push(element);
    }

    pub fn build(self) -> Trie<Label> {
        self.trie
    }
}

#[derive(Clone)]
pub struct Trie<Label> {
    root: Node<Label>,
}

impl<Label: Eq + Hash + Copy> Trie<Label> {
    pub fn push(&mut self, element: &[Label]) {
        let mut node = &mut self.root;
        for label in element.iter() {
            // Children store: a flat Vec<(Label, Node)>. For Unigram
            // trie nodes the fan-out is tiny — usually 1–4 children, with
            // root being the only "wide" node (≤ alphabet size). At those
            // sizes a linear scan over a packed Vec beats a HashMap on
            // both memory (no per-node hash table overhead × millions of
            // nodes) and lookup latency (predictable, cache-resident).
            //
            // The (A)HashMap variant in upstream tokenizers ≤ 0.22.x
            // ships an empty hash table with every Default::default()
            // Node, so for a 500k-vocab tokenizer (e.g. multilingual
            // model) it's 500k+ near-empty hash tables — hundreds of MB
            // of resident hashbrown structure with millions of small
            // allocs. Switching to ahash (PR #1799) cut hasher cost
            // but left the per-node table overhead unchanged.
            let pos = node.children.iter().position(|(l, _)| *l == *label);
            node = match pos {
                Some(idx) => &mut node.children[idx].1,
                None => {
                    node.children.push((*label, Node::default()));
                    let last = node.children.len() - 1;
                    &mut node.children[last].1
                }
            };
        }
        node.is_leaf = true;
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<'_, Label, T>
    where
        T: Iterator<Item = Label>,
    {
        TrieIterator {
            node: &self.root,
            prefix: vec![],
            iterator,
        }
    }
}

pub struct TrieIterator<'a, Label, T> {
    node: &'a Node<Label>,
    prefix: Vec<Label>,
    iterator: T,
}

impl<Label, T> Iterator for TrieIterator<'_, Label, T>
where
    Label: Eq + Hash + Copy,
    T: Iterator<Item = Label>,
{
    type Item = Vec<Label>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let label = self.iterator.next()?;
            self.prefix.push(label);
            // Linear find — same fan-out argument as in `push`.
            let child = self
                .node
                .children
                .iter()
                .find(|(l, _)| *l == label)
                .map(|(_, n)| n)?;
            self.node = child;
            if self.node.is_leaf {
                return Some(self.prefix.clone());
            }
        }
    }
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Self {
            root: Node::default(),
        }
    }
}

#[derive(Clone)]
pub struct Node<Label> {
    is_leaf: bool,
    /// Packed list of children. Empty `Vec` (no allocation) when the
    /// node has no children, which is the dominant case in deep trie
    /// branches.
    children: Vec<(Label, Node<Label>)>,
}

impl<Label> Default for Node<Label> {
    fn default() -> Self {
        Self {
            is_leaf: false,
            children: Vec::new(),
        }
    }
}
