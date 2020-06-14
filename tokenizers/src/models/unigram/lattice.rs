use std::rc::Rc;
use unicode_segmentation::UnicodeSegmentation;

pub struct Lattice<'a> {
    sentence: &'a str,
    graphemes: Vec<&'a str>,
    begin_nodes: Vec<Vec<Node>>,
    end_nodes: Vec<Vec<Node>>,
    current_id: usize,
}

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    pos: usize,
    length: usize,
    prev: Option<Rc<Node>>,
    backtrace_score: f64,
    score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

static mut CURRENT_ID: usize = 0;

impl Node {
    pub fn new(id: usize, pos: usize, length: usize) -> Node {
        Node {
            id,
            pos,
            length,
            prev: None,
            score: 0.0,
            backtrace_score: 0.0,
        }
    }
}

fn piece<'a>(lattice: &'a Lattice, node: &Node) -> String {
    lattice.graphemes[node.pos..node.pos + node.length].concat()
}

impl<'a> Lattice<'a> {
    pub fn from(sentence: &'a str) -> Lattice<'a> {
        let graphemes: Vec<_> = UnicodeSegmentation::graphemes(&sentence[..], true).collect();
        let k_reserved_node_size = 16;
        // We are adding 2 tokens, bos and eos
        let len = graphemes.len();
        let mut begin_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];
        let mut end_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];

        let bos = Node::new(0, 0, 0);
        let eos = Node::new(1, len, 0);
        begin_nodes[len].push(eos);
        end_nodes[0].push(bos);

        let current_id = 2;
        Lattice {
            sentence,
            graphemes,
            begin_nodes,
            end_nodes,
            current_id,
        }
    }

    pub fn insert(&mut self, pos: usize, length: usize) -> Node {
        let node = Node::new(self.current_id, pos, length);
        self.current_id += 1;
        // TODO node.piece ? Which is self.grapheme[pos..pos + length]
        // XXX: Careful, in sentence piece, length is in bytes, here we assume
        // it's in graphemes already, let's see if we can get away with it.
        self.begin_nodes[pos].push(node.clone());
        self.end_nodes[pos + length].push(node.clone());
        node
    }

    pub fn viterbi(&mut self) -> Vec<Node> {
        //TODO Remove this mut it's probably unnecessary
        //  const int len = size();
        let len = self.graphemes.len();
        for pos in 0..len {
            // println!("Pos {:?}", pos);
            // println!("n {:?}", self.begin_nodes[pos]);
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            for rnode in self.begin_nodes[pos].iter_mut() {
                // ??
                // rnode->prev = nullptr;
                // println!("Node {:?}", rnode);
                rnode.prev = None;
                let mut best_score = 0.0;
                let mut best_node: Option<Node> = None;
                for lnode in &self.end_nodes[pos] {
                    let score = lnode.backtrace_score + rnode.score;
                    if best_node.is_none() || score > best_score {
                        // TODO can we remove this clone ?
                        best_node = Some(lnode.clone());
                        best_score = score
                    }
                }
                // println!("Best node {:?}", best_node);
                match best_node {
                    Some(bnode) => {
                        rnode.prev = Some(Rc::new(bnode));
                        rnode.backtrace_score = best_score;
                    }
                    None => return vec![],
                }
            }
        }
        // println!("Here");

        let mut results: Vec<Node> = vec![];
        println!("prev {:?}", self.begin_nodes[len]);
        let prev = (&self.begin_nodes[len][0].prev).as_ref();
        if prev.is_none() {
            return vec![];
        }
        let mut node: Rc<Node> = prev.unwrap().clone();
        while !node.prev.is_none() {
            results.push(Rc::make_mut(&mut node).clone());
            node = node.prev.as_ref().unwrap().clone();
        }
        results.reverse();
        results
    }

    pub fn len(&self) -> usize {
        self.graphemes.len()
    }
    pub fn utf8_len(&self) -> usize {
        self.sentence.len()
    }

    pub fn bos_node(&self) -> &Node {
        &self.end_nodes[0][0]
    }
    pub fn eos_node(&self) -> &Node {
        &self.begin_nodes[self.graphemes.len()][0]
    }

    pub fn surface(&self, n: usize) -> &str {
        let mut m: usize = 0;
        self.graphemes.iter().take(n).for_each(|x| m += x.len());
        &self.sentence[m..]
    }
    pub fn sentence(&self) -> &str {
        &self.sentence
    }

    pub fn populate_marginal(&self, freq: f64, expected: &mut Vec<f64>) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_sentence() {
        let lattice = Lattice::from("");

        assert_eq!(lattice.len(), 0);
        assert_eq!(lattice.utf8_len(), 0);
        // EXPECT_EQ(0, lattice.utf8_size());

        let lattice = Lattice::from("");
        assert_eq!(lattice.len(), 0);
        assert_eq!(lattice.utf8_len(), 0);
        assert_eq!(lattice.sentence(), "");
        assert_eq!(lattice.surface(0), "");

        let lattice = Lattice::from("test");
        assert_eq!(lattice.len(), 4);
        assert_eq!(lattice.utf8_len(), 4);
        assert_eq!(lattice.sentence(), "test");
        assert_eq!(lattice.surface(0), "test");
        assert_eq!(lattice.surface(1), "est");
        assert_eq!(lattice.surface(2), "st");
        assert_eq!(lattice.surface(3), "t");

        let bos = lattice.bos_node();
        let eos = lattice.eos_node();

        assert_eq!(bos.id, 0);
        assert_eq!(eos.id, 1);
        assert_eq!(lattice.end_nodes[0].first().unwrap().id, bos.id);
        assert_eq!(lattice.begin_nodes[4].first().unwrap().id, eos.id);

        let lattice = Lattice::from("テストab");
        assert_eq!(lattice.len(), 5);
        assert_eq!(lattice.utf8_len(), 11);
        assert_eq!(lattice.sentence(), "テストab");
        assert_eq!(lattice.surface(0), "テストab");
        assert_eq!(lattice.surface(1), "ストab");
        assert_eq!(lattice.surface(2), "トab");
        assert_eq!(lattice.surface(3), "ab");
        assert_eq!(lattice.surface(4), "b");
    }

    #[test]
    fn insert_test() {
        let mut lattice = Lattice::from("ABあい");
        let nodes: Vec<Node> = vec![];

        let node0 = lattice.insert(0, 1);
        let node1 = lattice.insert(1, 1);
        let node2 = lattice.insert(2, 1);
        let node3 = lattice.insert(3, 1);
        let node4 = lattice.insert(0, 2);
        let node5 = lattice.insert(1, 2);
        let node6 = lattice.insert(2, 2);

        assert_eq!(piece(&lattice, &node0), "A");
        assert_eq!(piece(&lattice, &node1), "B");
        assert_eq!(piece(&lattice, &node2), "あ");
        assert_eq!(piece(&lattice, &node3), "い");
        assert_eq!(piece(&lattice, &node4), "AB");
        assert_eq!(piece(&lattice, &node5), "Bあ");
        assert_eq!(piece(&lattice, &node6), "あい");

        assert_eq!(node0.pos, 0);
        assert_eq!(node1.pos, 1);
        assert_eq!(node2.pos, 2);
        assert_eq!(node3.pos, 3);
        assert_eq!(node4.pos, 0);
        assert_eq!(node5.pos, 1);
        assert_eq!(node6.pos, 2);

        assert_eq!(node0.length, 1);
        assert_eq!(node1.length, 1);
        assert_eq!(node2.length, 1);
        assert_eq!(node3.length, 1);
        assert_eq!(node4.length, 2);
        assert_eq!(node5.length, 2);
        assert_eq!(node6.length, 2);

        assert_eq!(lattice.bos_node().id, 0);
        assert_eq!(lattice.eos_node().id, 1);
        assert_eq!(node0.id, 2);
        assert_eq!(node1.id, 3);
        assert_eq!(node2.id, 4);
        assert_eq!(node3.id, 5);
        assert_eq!(node4.id, 6);
        assert_eq!(node5.id, 7);
        assert_eq!(node6.id, 8);

        assert_eq!(lattice.begin_nodes[0].len(), 2);
        assert_eq!(lattice.begin_nodes[1].len(), 2);
        assert_eq!(lattice.begin_nodes[2].len(), 2);
        assert_eq!(lattice.begin_nodes[3].len(), 1);
        assert_eq!(lattice.begin_nodes[4].len(), 1);

        assert_eq!(lattice.end_nodes[0].len(), 1);
        assert_eq!(lattice.end_nodes[1].len(), 1);
        assert_eq!(lattice.end_nodes[2].len(), 2);
        assert_eq!(lattice.end_nodes[3].len(), 2);
        assert_eq!(lattice.end_nodes[4].len(), 2);

        assert_eq!(lattice.begin_nodes[0][0].id, node0.id);
        assert_eq!(lattice.begin_nodes[0][1].id, node4.id);
        assert_eq!(lattice.begin_nodes[1][0].id, node1.id);
        assert_eq!(lattice.begin_nodes[1][1].id, node5.id);
        assert_eq!(lattice.begin_nodes[2][0].id, node2.id);
        assert_eq!(lattice.begin_nodes[2][1].id, node6.id);
        assert_eq!(lattice.begin_nodes[3][0].id, node3.id);
        assert_eq!(lattice.eos_node().id, lattice.begin_nodes[4][0].id);

        assert_eq!(lattice.bos_node().id, lattice.end_nodes[0][0].id);
        assert_eq!(node0.id, lattice.end_nodes[1][0].id);
        assert_eq!(node1.id, lattice.end_nodes[2][0].id);
        assert_eq!(node4.id, lattice.end_nodes[2][1].id);
        assert_eq!(node2.id, lattice.end_nodes[3][0].id);
        assert_eq!(node5.id, lattice.end_nodes[3][1].id);
        assert_eq!(node3.id, lattice.end_nodes[4][0].id);
        assert_eq!(node6.id, lattice.end_nodes[4][1].id);
    }

    #[test]
    fn test_viterbi() {
        let mut lattice = Lattice::from("ABC");
        assert_eq!(lattice.viterbi(), vec![]);
        // Still incomplete
        lattice.insert(0, 1);
        assert_eq!(lattice.viterbi(), vec![]);
        lattice.insert(1, 1);
        lattice.insert(2, 1);
        // XXX: In sentence piece this is not tested, still incomplete ?
        // assert_eq!(lattice.viterbi(), vec![]);
    }
}
