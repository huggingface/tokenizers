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

pub struct Lattice<'a> {
    sentence: &'a str,
    pub chars: Vec<char>,
    nodes: Vec<NodeRef>,
    begin_nodes: Vec<Vec<NodeRef>>,
    end_nodes: Vec<Vec<NodeRef>>,
    current_id: usize,
}

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
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
    pub fn new(id: usize, pos: usize, length: usize, score: f64) -> Node {
        Node {
            id,
            pos,
            length,
            prev: None,
            score,
            backtrace_score: 0.0,
        }
    }
}

fn piece<'a>(lattice: &'a Lattice, node: &Node) -> String {
    lattice.chars[node.pos..node.pos + node.length]
        .into_iter()
        .collect()
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
    pub fn from(sentence: &'a str) -> Lattice<'a> {
        let chars: Vec<_> = sentence.chars().collect();
        let k_reserved_node_size = 16;
        // We are adding 2 tokens, bos and eos
        let len = chars.len();
        let mut nodes: Vec<NodeRef> = Vec::with_capacity(k_reserved_node_size);
        let mut begin_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];
        let mut end_nodes = vec![Vec::with_capacity(k_reserved_node_size); len + 1];

        let bos = Rc::new(RefCell::new(Node::new(0, 0, 0, 0.0)));
        let eos = Rc::new(RefCell::new(Node::new(1, len, 0, 0.0)));

        begin_nodes[len].push(Rc::clone(&eos));
        end_nodes[0].push(Rc::clone(&bos));

        nodes.push(bos);
        nodes.push(eos);

        let current_id = 2;
        Lattice {
            sentence,
            chars,
            nodes,
            begin_nodes,
            end_nodes,
            current_id,
        }
    }

    pub fn insert(&mut self, pos: usize, length: usize, score: f64) {
        self.insert_with_id(pos, length, score, self.current_id);
        self.current_id += 1;
    }

    pub fn insert_with_id(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node = Rc::new(RefCell::new(Node::new(id, pos, length, score)));

        // TODO node.piece ? Which is self.chars[pos..pos + length]
        // XXX: Careful, in sentence piece, length is in bytes, here we assume
        // it's in chars already, let's see if we can get away with it.
        self.begin_nodes[pos].push(Rc::clone(&node));
        self.end_nodes[pos + length].push(Rc::clone(&node));

        self.nodes.push(node);
    }

    pub fn viterbi(&mut self) -> Vec<NodeRef> {
        //TODO Remove this mut it's probably unnecessary
        //  const int len = size();
        let len = self.chars.len();
        for pos in 0..=len {
            // println!("Pos {:?}", pos);
            // println!("n {:?}", self.begin_nodes[pos]);
            if self.begin_nodes[pos].is_empty() {
                println!("Empty");
                return vec![];
            }
            for rnode in &self.begin_nodes[pos] {
                // ??
                // rnode->prev = nullptr;
                // println!("Node {:?}", rnode);
                rnode.borrow_mut().prev = None;
                let mut best_score = 0.0;
                let mut best_node: Option<NodeRef> = None;
                // println!("End nodes {:?}", self.end_nodes[pos]);
                for lnode in &self.end_nodes[pos] {
                    let score = lnode.borrow().backtrace_score + rnode.borrow().score;
                    if best_node.is_none() || score > best_score {
                        // TODO can we remove this clone ?
                        best_node = Some(lnode.clone());
                        best_score = score
                    }
                }
                // println!("Best node {:?}", best_node);
                match best_node {
                    Some(bnode) => {
                        rnode.borrow_mut().prev = Some(Rc::clone(&bnode));
                        rnode.borrow_mut().backtrace_score = best_score;
                    }
                    None => return vec![],
                }
                // println!("r node {:?}", rnode);
            }
        }
        // println!("Here");

        let mut results: Vec<NodeRef> = vec![];
        // println!("prev {:?}", self.begin_nodes[len]);
        let root = self.begin_nodes[len][0].borrow();
        let prev = root.prev.as_ref();
        if prev.is_none() {
            // println!("NOOONE");
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

    pub fn tokens(&mut self) -> Vec<String> {
        self.viterbi()
            .iter()
            .map(|node| piece(self, &node.borrow()))
            .collect()
    }

    pub fn nbest(&mut self, n: usize) -> Vec<Vec<NodeRef>> {
        match n {
            n if n < 1 => vec![],
            n if n == 1 => vec![self.viterbi()],
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
                    // println!("Node {:?}, bos_node {:?}", node, self.bos_node());
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
            .map(|v| v.iter().map(|node| piece(self, &node.borrow())).collect())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn utf8_len(&self) -> usize {
        self.sentence.len()
    }

    pub fn bos_node(&self) -> NodeRef {
        Rc::clone(&self.end_nodes[0][0])
    }
    pub fn eos_node(&self) -> NodeRef {
        Rc::clone(&self.begin_nodes[self.chars.len()][0])
    }

    pub fn surface(&self, n: usize) -> &str {
        let m = self.chars[..n].iter().map(|c| c.to_string().len()).sum();
        &self.sentence[m..]
    }
    pub fn sentence(&self) -> &str {
        &self.sentence
    }

    pub fn populate_marginal(&self, freq: f64, expected: &mut Vec<f64>) -> f64 {
        let len = self.len();
        let mut alpha = vec![0.0; self.nodes.len()];
        let mut beta = vec![0.0; self.nodes.len()];
        // let mut marginal = vec![0.0, self.nodes.len()];
        for pos in 0..=len {
            for rnode in &self.begin_nodes[pos] {
                for lnode in &self.end_nodes[pos] {
                    let lid = lnode.borrow().id;
                    let rid = rnode.borrow().id;
                    alpha[rid] = log_sum_exp(
                        alpha[rid],
                        lnode.borrow().score + alpha[lid],
                        *lnode == self.end_nodes[pos][0],
                    );
                }
            }
            let rpos = len - pos;
            for lnode in &self.end_nodes[rpos] {
                for rnode in &self.begin_nodes[rpos] {
                    let lid = lnode.borrow().id;
                    let rid = rnode.borrow().id;
                    beta[lid] = log_sum_exp(
                        beta[lid],
                        rnode.borrow().score + beta[rid],
                        *rnode == self.begin_nodes[rpos][0],
                    );
                }
            }
        }

        let z = alpha[self.begin_nodes[len][0].borrow().id];
        for pos in 0..len {
            for node in &self.begin_nodes[pos] {
                // TODO node.id is -1 for OOV, but we don't support this yet.
                if node.borrow().id > 1 {
                    let a = alpha[node.borrow().id];
                    let b = beta[node.borrow().id];
                    // Unigram uses -1 id for bos and eos, we don't so just substract 2 here.
                    // as bos, eos are 0 and 1.
                    expected[node.borrow().id - 2] +=
                        freq * (a + node.borrow().score + b - z).exp();
                }
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
                    let lid = lnode.borrow().id;
                    let rid = rnode.borrow().id;
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
        let mut z = alpha[self.eos_node().borrow().id];
        let mut node = self.eos_node();
        loop {
            probs.clear();
            let pos = node.borrow().pos;
            for lnode in &self.end_nodes[pos] {
                let lid = lnode.borrow().id;
                probs.push((alpha[lid] + theta * lnode.borrow().score - z).exp())
            }
            let dist = WeightedIndex::new(&probs).unwrap();
            let index = dist.sample(&mut rng);
            node = Rc::clone(&self.end_nodes[pos][index]);
            if node == self.bos_node() {
                break;
            }
            z = alpha[node.borrow().id];
            results.push(Rc::clone(&node));
        }
        results.reverse();
        results
    }

    pub fn sample_token(&self, theta: f64) -> Vec<String> {
        self.sample(theta)
            .iter()
            .map(|node| piece(self, &node.borrow()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use std::collections::HashMap;

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

        assert_eq!(bos.borrow().id, 0);
        assert_eq!(eos.borrow().id, 1);
        assert_eq!(
            lattice.end_nodes[0].first().unwrap().borrow().id,
            bos.borrow().id
        );
        assert_eq!(
            lattice.begin_nodes[4].first().unwrap().borrow().id,
            eos.borrow().id
        );

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

        lattice.insert(0, 1, 0.0);
        lattice.insert(1, 1, 0.0);
        lattice.insert(2, 1, 0.0);
        lattice.insert(3, 1, 0.0);
        lattice.insert(0, 2, 0.0);
        lattice.insert(1, 2, 0.0);
        lattice.insert(2, 2, 0.0);
        let node0 = lattice.nodes[2].borrow();
        let node1 = lattice.nodes[3].borrow();
        let node2 = lattice.nodes[4].borrow();
        let node3 = lattice.nodes[5].borrow();
        let node4 = lattice.nodes[6].borrow();
        let node5 = lattice.nodes[7].borrow();
        let node6 = lattice.nodes[8].borrow();

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

        assert_eq!(lattice.bos_node().borrow().id, 0);
        assert_eq!(lattice.eos_node().borrow().id, 1);
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

        assert_eq!(lattice.begin_nodes[0][0].borrow().id, node0.id);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, node4.id);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, node1.id);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, node5.id);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, node2.id);
        assert_eq!(lattice.begin_nodes[2][1].borrow().id, node6.id);
        assert_eq!(lattice.begin_nodes[3][0].borrow().id, node3.id);
        assert_eq!(
            lattice.eos_node().borrow().id,
            lattice.begin_nodes[4][0].borrow().id
        );

        assert_eq!(
            lattice.bos_node().borrow().id,
            lattice.end_nodes[0][0].borrow().id
        );
        assert_eq!(node0.id, lattice.end_nodes[1][0].borrow().id);
        assert_eq!(node1.id, lattice.end_nodes[2][0].borrow().id);
        assert_eq!(node4.id, lattice.end_nodes[2][1].borrow().id);
        assert_eq!(node2.id, lattice.end_nodes[3][0].borrow().id);
        assert_eq!(node5.id, lattice.end_nodes[3][1].borrow().id);
        assert_eq!(node3.id, lattice.end_nodes[4][0].borrow().id);
        assert_eq!(node6.id, lattice.end_nodes[4][1].borrow().id);
    }

    #[test]
    fn test_viterbi() {
        let mut lattice = Lattice::from("ABC");
        assert_eq!(lattice.viterbi(), vec![]);
        // Still incomplete
        lattice.insert(0, 1, 0.0);
        assert_eq!(lattice.viterbi(), vec![]);
        lattice.insert(1, 1, 0.0);
        lattice.insert(2, 1, 0.0);
        // XXX: In sentence piece this is not tested, still incomplete ?
        assert_eq!(lattice.viterbi().len(), 3);
    }

    #[test]
    fn test_viterbi2() {
        let mut lattice = Lattice::from("ABC");

        lattice.insert(0, 1, 0.0);
        lattice.insert(1, 1, 0.0);
        lattice.insert(2, 1, 0.0);

        assert_eq!(lattice.tokens(), ["A", "B", "C"]);

        lattice.insert(0, 2, 2.0);
        assert_eq!(lattice.tokens(), ["AB", "C"]);

        lattice.insert(1, 2, 5.0);
        assert_eq!(lattice.tokens(), ["A", "BC"]);

        lattice.insert(0, 3, 10.0);
        assert_eq!(lattice.tokens(), ["ABC"]);
    }

    #[test]
    fn test_nbest() {
        let mut lattice = Lattice::from("ABC");
        lattice.insert(0, 1, 0.0);
        lattice.insert(1, 1, 0.0);
        lattice.insert(2, 1, 0.0);
        lattice.insert(0, 2, 2.0);
        lattice.insert(1, 2, 5.0);
        lattice.insert(0, 3, 10.0);

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
        let mut lattice = Lattice::from("ABC");
        lattice.insert(0, 1, 1.0); // A
        lattice.insert(1, 1, 1.2); // B
        lattice.insert(2, 1, 2.5); // C
        lattice.insert(0, 2, 3.0); // AB
        lattice.insert(1, 2, 4.0); // BC
        lattice.insert(0, 3, 2.0); // ABC

        let mut probs = vec![0.0; 6];
        let p1 = (1.0_f64 + 1.2 + 2.5).exp();
        let p2 = (3.0_f64 + 2.5).exp();
        let p3 = (1.0_f64 + 4.0).exp();
        let p4 = 2.0_f64.exp();
        let z = p1 + p2 + p3 + p4;

        let log_z = lattice.populate_marginal(1.0, &mut probs);

        assert_approx_eq!(log_z, z.ln(), 0.001);
        assert_approx_eq!(probs[0], (p1 + p3) / z, 0.001);
        assert_approx_eq!(probs[1], (p1) / z, 0.001);
        assert_approx_eq!(probs[2], (p1 + p2) / z, 0.001);
        assert_approx_eq!(probs[3], (p2) / z, 0.001);
        assert_approx_eq!(probs[4], (p3) / z, 0.001);
        assert_approx_eq!(probs[5], (p4) / z, 0.001);
    }

    #[test]
    fn test_sample() {
        let mut lattice = Lattice::from("ABC");
        lattice.insert(0, 1, 1.0); // A
        lattice.insert(1, 1, 1.2); // B
        lattice.insert(2, 1, 1.5); // C
        lattice.insert(0, 2, 1.6); // AB
        lattice.insert(1, 2, 1.7); // BC
        lattice.insert(0, 3, 1.8); // ABC

        let thetas: Vec<f64> = vec![0.0, 0.01, 0.5, 0.7, 1.0];

        for theta in thetas {
            let mut probs: HashMap<String, f64> = HashMap::new();
            probs.insert("A B C".to_string(), (theta * (1.0 + 1.2 + 1.5)).exp());
            probs.insert("AB C".to_string(), (theta * (1.6 + 1.5)).exp());
            probs.insert("A BC".to_string(), (theta * (1.0 + 1.7)).exp());
            probs.insert("ABC".to_string(), (theta * (1.8)).exp());

            // Computes expected probabilities.
            let mut z = 0.0;

            for (_, p) in probs.iter() {
                z += p;
            }
            for (_, p) in probs.iter_mut() {
                *p /= z;
            }

            let n_trials = 100_000;
            let mut freq: HashMap<String, u32> = HashMap::new();
            for _ in 0..n_trials {
                let string = lattice.sample_token(theta).join(" ");
                *freq.entry(string).or_insert(0) += 1;
            }

            assert_eq!(freq.len(), probs.len());
            for (s, p) in probs.iter() {
                assert_approx_eq!(1.0 * (freq[s] as f64) / (n_trials as f64), p, 0.03)
            }
        }
    }
}
