use std::collections::HashMap;
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

pub struct Trie<Label> {
    root: Node<Label>,
}

impl<Label: Eq + Hash + Copy> Trie<Label> {
    pub fn push(&mut self, element: &[Label]) {
        let mut node = &mut self.root;
        for label in element.iter() {
            node = node.children.entry(*label).or_insert_with(Node::default);
        }
        node.is_leaf = true;
    }

    pub fn common_prefix_search(&self, element: &[Label]) -> Vec<Vec<Label>> {
        let mut node = &self.root;
        let mut results = vec![];
        let mut prefix = vec![];
        for label in element.iter() {
            prefix.push(*label);
            let child_opt = node.children.get(label);
            if let Some(child) = child_opt {
                node = child;
                if node.is_leaf {
                    results.push(prefix.clone());
                }
            } else {
                return results;
            }
        }
        results
    }
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Trie {
            root: Node::default(),
        }
    }
}

pub struct Node<Label> {
    is_leaf: bool,
    children: HashMap<Label, Node<Label>>,
}

impl<Label> Default for Node<Label> {
    fn default() -> Self {
        Node {
            is_leaf: false,
            children: HashMap::new(),
        }
    }
}
