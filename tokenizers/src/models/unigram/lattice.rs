use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::collections::BinaryHeap;
use std::rc::Rc;

type NodeRef = Rc<RefCell<Node>>;
type HypothesisRef = Rc<RefCell<Hypothesis>>;
type Agenda = BinaryHeap<Hypothesis>;

struct Hypothesis {
    node_ref: NodeRef,
    next: Option<HypothesisRef>,
    fx: f64,
    gx: f64,
}
impl Hypothesis {
    pub fn new(node_ref: NodeRef, next: Option<HypothesisRef>, fx: f64, gx: f64) -> Hypothesis {
        Hypothesis {
            node_ref,
            next,
            fx,
            gx,
        }
    }
}
impl PartialEq for Hypothesis {
    fn eq(&self, other: &Hypothesis) -> bool {
        self.fx == other.fx
    }
}
impl Eq for Hypothesis {}
impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Hypothesis) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// TODO Maybe use Ordered Floats (https://docs.rs/ordered-float/1.0.2/ordered_float/)
impl Ord for Hypothesis {
    fn cmp(&self, other: &Hypothesis) -> Ordering {
        if self.fx < other.fx {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

/// Structure to implement Viterbi algorithm to find the best encoding, or sample
/// from all possible encodings of a given sentence.
#[derive(Debug)]
pub struct Lattice<'a> {
    pub(super) sentence: &'a str,
    len: usize,
    nodes: Vec<NodeRef>,
    pub(super) begin_nodes: Vec<Vec<NodeRef>>,
    pub(super) end_nodes: Vec<Vec<NodeRef>>,
    bos_id: usize,
    eos_id: usize,
}

impl std::fmt::Display for Lattice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_pieces = |nodes: &Vec<Vec<NodeRef>>| {
            nodes
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|n| self.piece(&n.borrow()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        f.debug_struct("Lattice")
            .field("sentence", &self.sentence)
            .field("begin_nodes", &display_pieces(&self.begin_nodes))
            .field("end_nodes", &display_pieces(&self.end_nodes))
            .finish()
    }
}

/// A node from the lattice, that helps reconstruct the underlying `String`
#[derive(Debug, Clone)]
pub struct Node {
    // Vocabulary id
    pub(super) id: usize,
    // Local lattice identifier
    pub(super) node_id: usize,
    pos: usize,
    length: usize,
    prev: Option<NodeRef>,
    backtrace_score: f64,
    score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl Node {
    pub fn new(id: usize, node_id: usize, pos: usize, length: usize, score: f64) -> Node {
        Node {
            id,
            node_id,
            pos,
            length,
            prev: None,
            score,
            backtrace_score: 0.0,
        }
    }
}

/// Returns log(exp(x) + exp(y)).
/// if init_mode is true, returns log(exp(y)) == y.
/// log(\sum_i exp(a[i])) can be computed as
/// for (int i = 0; i < a.size(); ++i)
///   x = LogSumExp(x, a[i], i == 0);
fn log_sum_exp(x: f64, y: f64, init_mode: bool) -> f64 {
    if init_mode {
        y
    } else {
        let (vmin, vmax) = if x > y { (y, x) } else { (x, y) };
        let k_minus_log_epsilon = 50.0;
        if vmax > vmin + k_minus_log_epsilon {
            vmax
        } else {
            vmax + ((vmin - vmax).exp() + 1.0).ln()
        }
    }
}

impl<'a> Lattice<'a> {
    pub fn from(sentence: &'a str, bos_id: usize, eos_id: usize) -> Lattice<'a> {
        let len = sentence.bytes().count();
        let k_reserved_node_size = 16;
        // We are adding 2 tokens, bos and eos
        let mut nodes: Vec<NodeRef> = Vec::with_capacity(k_reserved_node_size);
        let mut begin_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];
        let mut end_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];

        let bos = Rc::new(RefCell::new(Node::new(bos_id, 0, 0, 0, 0.0)));
        let eos = Rc::new(RefCell::new(Node::new(eos_id, 1, len, 0, 0.0)));

        begin_nodes[len].push(Rc::clone(&eos));
        end_nodes[0].push(Rc::clone(&bos));

        nodes.push(bos);
        nodes.push(eos);

        Lattice {
            sentence,
            len,
            nodes,
            begin_nodes,
            end_nodes,
            bos_id,
            eos_id,
        }
    }

    pub fn insert(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node_id = self.nodes.len();
        let node = Rc::new(RefCell::new(Node::new(id, node_id, pos, length, score)));

        self.begin_nodes[pos].push(Rc::clone(&node));
        self.end_nodes[pos + length].push(Rc::clone(&node));

        self.nodes.push(node);
    }

    pub fn viterbi(&mut self) -> Vec<NodeRef> {
        let len = self.len;
        let mut pos = 0;
        while pos <= len {
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            for rnode in &self.begin_nodes[pos] {
                rnode.borrow_mut().prev = None;
                let mut best_score = 0.0;
                let mut best_node: Option<NodeRef> = None;
                for lnode in &self.end_nodes[pos] {
                    let score = lnode.borrow().backtrace_score + rnode.borrow().score;
                    if best_node.is_none() || score > best_score {
                        // TODO can we remove this clone ?
                        best_node = Some(lnode.clone());
                        best_score = score
                    }
                }
                match best_node {
                    Some(bnode) => {
                        rnode.borrow_mut().prev = Some(Rc::clone(&bnode));
                        rnode.borrow_mut().backtrace_score = best_score;
                    }
                    None => return vec![],
                }
            }
            if let Some(c) = self.sentence[pos..].chars().next() {
                pos += c.len_utf8();
            } else {
                break;
            }
        }

        let mut results: Vec<NodeRef> = vec![];
        let root = self.begin_nodes[len][0].borrow();
        let prev = root.prev.as_ref();
        if prev.is_none() {
            return vec![];
        }
        let mut node: NodeRef = prev.unwrap().clone();
        while node.borrow().prev.is_some() {
            results.push(node.clone());
            let n = node.borrow().clone();
            node = n.prev.as_ref().unwrap().clone();
        }
        results.reverse();
        results
    }

    pub fn piece(&self, node: &Node) -> String {
        self.sentence[node.pos..node.pos + node.length].to_owned()
    }

    pub fn tokens(&mut self) -> Vec<String> {
        self.viterbi()
            .iter()
            .map(|node| self.piece(&node.borrow()))
            .collect()
    }

    pub fn nbest(&mut self, n: usize) -> Vec<Vec<NodeRef>> {
        match n {
            0 => vec![],
            1 => vec![self.viterbi()],
            _ => {
                // let k_reserved_hypothesis_size = 512;
                let mut agenda: Agenda = BinaryHeap::new();
                let mut hypotheses: Vec<Vec<NodeRef>> = vec![];
                let eos = self.eos_node();
                let score = eos.borrow().score;
                let hypo = Hypothesis::new(eos, None, score, score);
                agenda.push(hypo);

                // Fill backtrace scores
                self.viterbi();

                while !agenda.is_empty() {
                    let top = Rc::new(RefCell::new(agenda.pop().unwrap()));
                    let node = Rc::clone(&top.borrow().node_ref);
                    if node.borrow().id == self.bos_node().borrow().id {
                        let mut hypothesis = vec![];
                        let mut next: HypothesisRef =
                            Rc::clone(&top.borrow().next.as_ref().unwrap());
                        while next.borrow().next.is_some() {
                            hypothesis.push(next.borrow().node_ref.clone());
                            let c: HypothesisRef = next.clone();
                            // let c: Ref<Hypothesis> = next.clone().borrow();
                            next = Rc::clone(c.borrow().next.as_ref().unwrap());
                        }
                        hypotheses.push(hypothesis);
                        if hypotheses.len() == n {
                            return hypotheses;
                        }
                    } else {
                        for lnode in &self.end_nodes[node.borrow().pos] {
                            let top_gx = top.borrow().gx;
                            let fx = lnode.borrow().backtrace_score + top_gx;
                            let gx = lnode.borrow().score + top_gx;
                            let hyp =
                                Hypothesis::new(Rc::clone(lnode), Some(Rc::clone(&top)), fx, gx);
                            agenda.push(hyp);
                        }
                        // When the input is too long or contains duplicated phrases,
                        // `agenda` will get extremely big. Here we avoid this case by
                        // dynamically shrinking the agenda.
                        let k_max_agenda_size = 100_000;
                        let k_min_agenda_size = 512;
                        if agenda.len() > k_max_agenda_size {
                            let mut new_agenda = BinaryHeap::new();
                            let len = min(k_min_agenda_size, n * 10);
                            for _i in 0..len {
                                new_agenda.push(agenda.pop().unwrap());
                            }
                            agenda = new_agenda;
                        }
                    }
                }
                hypotheses
            }
        }
    }

    pub fn nbest_tokens(&mut self, n: usize) -> Vec<Vec<String>> {
        self.nbest(n)
            .iter()
            .map(|v| v.iter().map(|node| self.piece(&node.borrow())).collect())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bos_node(&self) -> NodeRef {
        Rc::clone(&self.end_nodes[0][0])
    }
    pub fn eos_node(&self) -> NodeRef {
        Rc::clone(&self.begin_nodes[self.len][0])
    }

    pub fn surface(&self, n: usize) -> &str {
        match self.sentence.char_indices().nth(n) {
            Some((pos, _)) => &self.sentence[pos..],
            None => "",
        }
    }
    pub fn sentence(&self) -> &str {
        &self.sentence
    }

    pub fn populate_marginal(&self, freq: f64, expected: &mut Vec<f64>) -> f64 {
        let len = self.len();
        let n_nodes = self.nodes.len();
        let mut alpha = vec![0.0; n_nodes];
        let mut beta = vec![0.0; n_nodes];
        for pos in 0..=len {
            for rnode in &self.begin_nodes[pos] {
                for lnode in &self.end_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        lnode.borrow().score + alpha[lid],
                        *lnode == self.end_nodes[pos][0],
                    );
                }
            }
        }
        for pos in (0..=len).rev() {
            // let rpos = len - pos;
            for lnode in &self.end_nodes[pos] {
                for rnode in &self.begin_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;
                    beta[lid] = log_sum_exp(
                        beta[lid],
                        rnode.borrow().score + beta[rid],
                        *rnode == self.begin_nodes[pos][0],
                    );
                }
            }
        }

        let eos_id = self.begin_nodes[len][0].borrow().node_id;
        let z = alpha[eos_id];
        for pos in 0..len {
            for node in &self.begin_nodes[pos] {
                let node_id = node.borrow().node_id;
                let id = node.borrow().id;
                let a = alpha[node_id];
                let b = beta[node_id];
                let total = a + node.borrow().score + b - z;
                let update = freq * total.exp();
                expected[id] += update;
            }
        }
        freq * z
    }

    pub fn sample(&self, theta: f64) -> Vec<NodeRef> {
        let len = self.len();
        if len == 0 {
            return vec![];
        }
        let mut alpha = vec![0.0; self.nodes.len()];
        for pos in 0..=len {
            for rnode in &self.begin_nodes[pos] {
                for lnode in &self.end_nodes[pos] {
                    let lid = lnode.borrow().node_id;
                    let rid = rnode.borrow().node_id;
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        theta * (lnode.borrow().score + alpha[lid]),
                        *lnode == self.end_nodes[pos][0],
                    );
                }
            }
        }

        let mut rng = thread_rng();
        let mut results: Vec<NodeRef> = vec![];
        let mut probs: Vec<f64> = vec![];
        let mut z = alpha[self.eos_node().borrow().node_id];
        let mut node = self.eos_node();
        loop {
            probs.clear();
            let pos = node.borrow().pos;
            for lnode in &self.end_nodes[pos] {
                let lid = lnode.borrow().node_id;
                probs.push((alpha[lid] + theta * lnode.borrow().score - z).exp())
            }
            let dist = WeightedIndex::new(&probs).unwrap();
            let index = dist.sample(&mut rng);
            node = Rc::clone(&self.end_nodes[pos][index]);
            if node == self.bos_node() {
                break;
            }
            z = alpha[node.borrow().node_id];
            results.push(Rc::clone(&node));
        }
        results.reverse();
        results
    }

    pub fn sample_token(&self, theta: f64) -> Vec<String> {
        self.sample(theta)
            .iter()
            .map(|node| self.piece(&node.borrow()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn set_sentence() {
        let lattice = Lattice::from("", 1, 2);

        assert_eq!(lattice.len(), 0);

        let lattice = Lattice::from("", 1, 2);
        assert_eq!(lattice.len(), 0);
        assert_eq!(lattice.sentence(), "");
        assert_eq!(lattice.surface(0), "");

        let lattice = Lattice::from("test", 1, 2);
        assert_eq!(lattice.len(), 4);
        assert_eq!(lattice.sentence(), "test");
        assert_eq!(lattice.surface(0), "test");
        assert_eq!(lattice.surface(1), "est");
        assert_eq!(lattice.surface(2), "st");
        assert_eq!(lattice.surface(3), "t");

        let bos = lattice.bos_node();
        let eos = lattice.eos_node();

        assert_eq!(bos.borrow().id, 1);
        assert_eq!(eos.borrow().id, 2);
        assert_eq!(
            lattice.end_nodes[0].first().unwrap().borrow().id,
            bos.borrow().id
        );
        assert_eq!(
            lattice.begin_nodes[4].first().unwrap().borrow().id,
            eos.borrow().id
        );

        let lattice = Lattice::from("テストab", 1, 2);
        assert_eq!(lattice.len(), 11);
        assert_eq!(lattice.sentence(), "テストab");
        assert_eq!(lattice.surface(0), "テストab");
        assert_eq!(lattice.surface(1), "ストab");
        assert_eq!(lattice.surface(2), "トab");
        assert_eq!(lattice.surface(3), "ab");
        assert_eq!(lattice.surface(4), "b");
    }

    #[test]
    fn insert_test() {
        let mut lattice = Lattice::from("ABあい", 1, 2);

        lattice.insert(0, 1, 0.0, 3);
        lattice.insert(1, 1, 0.0, 4);
        lattice.insert(2, 3, 0.0, 5);
        lattice.insert(5, 3, 0.0, 6);
        lattice.insert(0, 2, 0.0, 7);
        lattice.insert(1, 4, 0.0, 8);
        lattice.insert(2, 6, 0.0, 9);
        // 0 & 1 are bos and eos
        let node0 = lattice.nodes[2].borrow();
        let node1 = lattice.nodes[3].borrow();
        let node2 = lattice.nodes[4].borrow();
        let node3 = lattice.nodes[5].borrow();
        let node4 = lattice.nodes[6].borrow();
        let node5 = lattice.nodes[7].borrow();
        let node6 = lattice.nodes[8].borrow();

        assert_eq!(lattice.piece(&node0), "A");
        assert_eq!(lattice.piece(&node1), "B");
        assert_eq!(lattice.piece(&node2), "あ");
        assert_eq!(lattice.piece(&node3), "い");
        assert_eq!(lattice.piece(&node4), "AB");
        assert_eq!(lattice.piece(&node5), "Bあ");
        assert_eq!(lattice.piece(&node6), "あい");

        assert_eq!(node0.pos, 0);
        assert_eq!(node1.pos, 1);
        assert_eq!(node2.pos, 2);
        assert_eq!(node3.pos, 5);
        assert_eq!(node4.pos, 0);
        assert_eq!(node5.pos, 1);
        assert_eq!(node6.pos, 2);

        assert_eq!(node0.length, 1);
        assert_eq!(node1.length, 1);
        assert_eq!(node2.length, 3);
        assert_eq!(node3.length, 3);
        assert_eq!(node4.length, 2);
        assert_eq!(node5.length, 4);
        assert_eq!(node6.length, 6);

        assert_eq!(lattice.bos_node().borrow().id, 1);
        assert_eq!(lattice.eos_node().borrow().id, 2);
        assert_eq!(node0.id, 3);
        assert_eq!(node1.id, 4);
        assert_eq!(node2.id, 5);
        assert_eq!(node3.id, 6);
        assert_eq!(node4.id, 7);
        assert_eq!(node5.id, 8);
        assert_eq!(node6.id, 9);

        assert_eq!(lattice.begin_nodes[0].len(), 2);
        assert_eq!(lattice.begin_nodes[1].len(), 2);
        assert_eq!(lattice.begin_nodes[2].len(), 2);
        assert_eq!(lattice.begin_nodes[5].len(), 1);
        assert_eq!(lattice.begin_nodes[8].len(), 1);

        assert_eq!(lattice.end_nodes[0].len(), 1);
        assert_eq!(lattice.end_nodes[1].len(), 1);
        assert_eq!(lattice.end_nodes[2].len(), 2);
        assert_eq!(lattice.end_nodes[5].len(), 2);
        assert_eq!(lattice.end_nodes[8].len(), 2);

        assert_eq!(lattice.begin_nodes[0][0].borrow().id, node0.id);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, node4.id);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, node1.id);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, node5.id);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, node2.id);
        assert_eq!(lattice.begin_nodes[2][1].borrow().id, node6.id);
        assert_eq!(lattice.begin_nodes[5][0].borrow().id, node3.id);
        assert_eq!(
            lattice.eos_node().borrow().id,
            lattice.begin_nodes[8][0].borrow().id
        );

        assert_eq!(
            lattice.bos_node().borrow().id,
            lattice.end_nodes[0][0].borrow().id
        );
        assert_eq!(node0.id, lattice.end_nodes[1][0].borrow().id);
        assert_eq!(node1.id, lattice.end_nodes[2][0].borrow().id);
        assert_eq!(node4.id, lattice.end_nodes[2][1].borrow().id);
        assert_eq!(node2.id, lattice.end_nodes[5][0].borrow().id);
        assert_eq!(node5.id, lattice.end_nodes[5][1].borrow().id);
        assert_eq!(node3.id, lattice.end_nodes[8][0].borrow().id);
        assert_eq!(node6.id, lattice.end_nodes[8][1].borrow().id);
    }

    #[test]
    fn test_viterbi() {
        let mut lattice = Lattice::from("ABC", 1, 2);
        assert_eq!(lattice.viterbi(), vec![]);
        // Still incomplete
        lattice.insert(0, 1, 0.0, 3);
        assert_eq!(lattice.viterbi(), vec![]);
        lattice.insert(1, 1, 0.0, 4);
        lattice.insert(2, 1, 0.0, 5);
        // XXX: In sentence piece this is not tested, still incomplete ?
        assert_eq!(lattice.viterbi().len(), 3);
    }

    #[test]
    fn test_viterbi2() {
        let mut lattice = Lattice::from("ABC", 1, 2);

        lattice.insert(0, 1, 0.0, 3);
        lattice.insert(1, 1, 0.0, 4);
        lattice.insert(2, 1, 0.0, 5);

        assert_eq!(lattice.tokens(), ["A", "B", "C"]);

        lattice.insert(0, 2, 2.0, 6);
        assert_eq!(lattice.tokens(), ["AB", "C"]);

        lattice.insert(1, 2, 5.0, 7);
        assert_eq!(lattice.tokens(), ["A", "BC"]);

        lattice.insert(0, 3, 10.0, 8);
        assert_eq!(lattice.tokens(), ["ABC"]);
    }

    #[test]
    fn test_nbest() {
        let mut lattice = Lattice::from("ABC", 1, 2);
        lattice.insert(0, 1, 0.0, 3);
        lattice.insert(1, 1, 0.0, 4);
        lattice.insert(2, 1, 0.0, 5);
        lattice.insert(0, 2, 2.0, 6);
        lattice.insert(1, 2, 5.0, 7);
        lattice.insert(0, 3, 10.0, 8);

        let nbests = lattice.nbest_tokens(10);
        assert_eq!(
            nbests,
            vec![
                vec!["ABC"],
                vec!["A", "BC"],
                vec!["AB", "C"],
                vec!["A", "B", "C"]
            ]
        );

        assert!(lattice.nbest_tokens(0).is_empty());
        assert_eq!(lattice.nbest_tokens(1), vec![vec!["ABC"]]);
    }
    #[test]
    fn test_log_sum_exp() {
        let mut x = 0.0;

        let v: Vec<f64> = vec![1.0, 2.0, 3.0];
        for (i, y) in v.iter().enumerate() {
            x = log_sum_exp(x, *y, i == 0);
        }
        assert_approx_eq!(x, v.iter().map(|n| n.exp()).sum::<f64>().ln(), 0.001);
    }

    #[test]
    fn test_populate() {
        let mut lattice = Lattice::from("ABC", 1, 2);
        lattice.insert(0, 1, 1.0, 3); // A
        lattice.insert(1, 1, 1.2, 4); // B
        lattice.insert(2, 1, 2.5, 5); // C
        lattice.insert(0, 2, 3.0, 6); // AB
        lattice.insert(1, 2, 4.0, 7); // BC
        lattice.insert(0, 3, 2.0, 8); // ABC

        let mut probs = vec![0.0; 9];
        let p1 = (1.0_f64 + 1.2 + 2.5).exp();
        let p2 = (3.0_f64 + 2.5).exp();
        let p3 = (1.0_f64 + 4.0).exp();
        let p4 = 2.0_f64.exp();
        let z = p1 + p2 + p3 + p4;

        let log_z = lattice.populate_marginal(1.0, &mut probs);

        assert_approx_eq!(log_z, z.ln(), 0.001);
        assert_approx_eq!(probs[0], 0.0, 0.001);
        assert_approx_eq!(probs[1], 0.0, 0.001);
        assert_approx_eq!(probs[2], 0.0, 0.001);
        assert_approx_eq!(probs[3], (p1 + p3) / z, 0.001);
        assert_approx_eq!(probs[4], (p1) / z, 0.001);
        assert_approx_eq!(probs[5], (p1 + p2) / z, 0.001);
        assert_approx_eq!(probs[6], (p2) / z, 0.001);
        assert_approx_eq!(probs[7], (p3) / z, 0.001);
        assert_approx_eq!(probs[8], (p4) / z, 0.001);
    }
}
