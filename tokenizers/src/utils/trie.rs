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

#[derive(Clone)]
pub struct Trie<Label> {
    root: Node<Label>,
}

impl<Label> std::fmt::Debug for Trie<Label> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Trie(...)",)
    }
}

impl<Label: Eq + Hash + Copy> Trie<Label> {
    pub fn push(&mut self, element: &[Label]) {
        let mut node = &mut self.root;
        for label in element.iter() {
            node = node.children.entry(*label).or_insert_with(Node::default);
        }
        node.is_leaf = true;
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<Label, T>
    where
        T: Iterator<Item = Label>,
    {
        TrieIterator {
            node: &self.root,
            prefix: vec![],
            iterator,
        }
    }

    pub fn matches<'a, 'b>(&'a self, content: &'b [Label]) -> MatchesIterator<'_, 'b, Label> {
        MatchesIterator {
            root: &self.root,
            states: vec![],
            index: 0,
            content,
        }
    }
}

pub struct MatchesIterator<'a, 'b, Label> {
    root: &'a Node<Label>,
    states: Vec<(usize, &'a Node<Label>)>,
    index: usize,
    content: &'b [Label],
}

impl<'a, 'b, Label> Iterator for MatchesIterator<'a, 'b, Label>
where
    Label: Eq + Hash + Copy,
{
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        for label in &self.content[self.index..] {
            let mut result = None;
            self.states = self
                .states
                .iter()
                .filter_map(|(start, node)| {
                    if !result.is_none() {
                        None
                    } else if node.is_leaf {
                        // Lookahead to match longest first
                        // Important in case of extra_id_1 vs extra_id_100
                        // Here we are also actively looking for other earlier partial
                        // matches
                        // "[CLS]", "L", we need to match CLS even if L is special
                        let mut start = start;
                        let mut stop = self.index;
                        let mut skip = self.index;
                        for (lookstart, looktrie_pointer) in &self.states {
                            let mut looktrie_pointer = *looktrie_pointer;
                            if lookstart > start {
                                // This partial match is later, we can stop looking
                                break;
                                // }else if lookstart < start{
                                //     // This partial match is earlier, the trie pointer
                                //     // was already updated, so index is + 1
                                //     lookahead_index = current + 1
                                //     end = current + 1
                            }
                            // Here lookstart == start and
                            //      looktrie_pointer == trie_pointer
                            // It wasn't updated yet so indices are current ones
                            let mut lookahead_index = self.index;

                            let mut next_char = self.content.get(lookahead_index);
                            while let Some(nchar) = next_char {
                                if let Some(sublooktrie_pointer) =
                                    looktrie_pointer.children.get(nchar)
                                {
                                    looktrie_pointer = sublooktrie_pointer;
                                    lookahead_index += 1;
                                    if looktrie_pointer.is_leaf {
                                        start = lookstart;
                                        stop = lookahead_index;
                                        skip = lookahead_index;
                                    }
                                    next_char = self.content.get(lookahead_index);
                                } else {
                                    break;
                                }
                            }
                        }
                        result = Some((*start, stop, skip));
                        None
                    } else if let Some(subnode) = node.children.get(&label) {
                        Some((*start, subnode))
                    } else {
                        None
                    }
                })
                .collect();
            if !result.is_none() {
                self.states.clear();
            }
            if let Some(subnode) = self.root.children.get(&label) {
                self.states.push((self.index, subnode));
            }
            self.index += 1;

            if let Some((start, stop, skip)) = result {
                if skip > self.index {
                    self.index = skip;
                }
                return Some((start, stop));
            }
        }
        let mut result = None;
        for (start, node) in &self.states {
            if node.is_leaf {
                result = Some((*start, self.index));
                break;
            }
        }
        self.states.clear();
        return result;
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
            let child = self.node.children.get(&label)?;
            self.node = child;
            if self.node.is_leaf {
                return Some(self.prefix.clone());
            }
        }
    }
}

impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Trie {
            root: Node::default(),
        }
    }
}

#[derive(Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() {
        let mut trie = Trie::default();
        trie.push(b"abc");
        trie.push(b"bcd");

        let matches: Vec<_> = trie.matches("This is a test".as_bytes()).collect();
        assert_eq!(matches, vec![]);

        let matches: Vec<_> = trie.matches("abcd".as_bytes()).collect();
        assert_eq!(matches, vec![(0, 3)]);
    }

    #[test]
    fn test_single() {
        let mut trie = Trie::default();
        trie.push(b"A");
        assert_eq!(
            trie.matches("ABC".as_bytes()).collect::<Vec<_>>(),
            vec![(0, 1)]
        );
        assert_eq!(
            trie.matches("BCA".as_bytes()).collect::<Vec<_>>(),
            vec![(2, 3)]
        );
    }

    #[test]
    fn test_longest() {
        let mut trie = Trie::default();
        trie.push(b"[CLS]");
        trie.push(b"extra_id_1");
        trie.push(b"extra_id_100");
        assert_eq!(
            trie.matches("[CLS] This is a extra_id_100".as_bytes())
                .collect::<Vec<_>>(),
            vec![(0, 5), (16, 28)]
        );
    }

    #[test]
    fn test_trie_final() {
        let mut trie = Trie::default();
        trie.push(b"TOKEN]");
        trie.push(b"[SPECIAL_TOKEN]");
        assert_eq!(
            trie.matches("This is something [SPECIAL_TOKEN]".as_bytes())
                .collect::<Vec<_>>(),
            vec![(18, 33)]
        );
    }

    #[test]
    fn test_trie_subtokens() {
        let mut trie = Trie::default();
        trie.push(b"A");
        trie.push(b"P");
        trie.push(b"[SPECIAL_TOKEN]");
        assert_eq!(
            trie.matches("This is something [SPECIAL_TOKEN]".as_bytes())
                .collect::<Vec<_>>(),
            vec![(18, 33)]
        );
    }

    #[test]
    fn test_trie_suffix_tokens() {
        let mut trie = Trie::default();
        trie.push(b"AB");
        trie.push(b"B");
        trie.push(b"C");
        assert_eq!(
            trie.matches("ABC".as_bytes()).collect::<Vec<_>>(),
            vec![(0, 2), (2, 3)]
        );
    }
}
