use dary_heap::QuaternaryHeap;
use rand::distr::weighted::WeightedIndex;
use rand::{prelude::*, rng};
use std::cmp::{Ordering, min};
use std::rc::Rc;

#[inline]
fn utf8_char_width(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

type HypothesisRef = Rc<Hypothesis>;
type Agenda = QuaternaryHeap<Hypothesis>;

struct Hypothesis {
    node_id: usize,
    next: Option<HypothesisRef>,
    fx: f64,
    gx: f64,
}
impl Hypothesis {
    pub fn new(node_id: usize, next: Option<HypothesisRef>, fx: f64, gx: f64) -> Self {
        Self {
            node_id,
            next,
            fx,
            gx,
        }
    }
}
impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.fx == other.fx
    }
}
impl Eq for Hypothesis {}
impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// TODO Maybe use Ordered Floats (https://docs.rs/ordered-float/1.0.2/ordered_float/)
impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
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
    pub(super) nodes: Vec<Node>,
    pub(super) begin_nodes: Vec<Vec<usize>>,
    pub(super) end_nodes: Vec<Vec<usize>>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    _bos_id: usize,
    _eos_id: usize,
}

impl std::fmt::Display for Lattice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_pieces = |nodes: &Vec<Vec<usize>>| {
            nodes
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|&node_id| self.piece(&self.nodes[node_id]))
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
    pub id: usize,
    // Local lattice identifier
    pub(super) node_id: usize,
    pos: usize,
    length: usize,
    prev: usize,
    backtrace_score: f64,
    score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl Node {
    pub(crate) fn new(id: usize, node_id: usize, pos: usize, length: usize, score: f64) -> Self {
        Self {
            id,
            node_id,
            pos,
            length,
            prev: usize::MAX,
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
#[inline]
fn log_sum_exp(x: f64, y: f64, init_mode: bool) -> f64 {
    if init_mode {
        y
    } else {
        let (vmin, vmax) = if x > y { (y, x) } else { (x, y) };
        let k_minus_log_epsilon = 50.0;
        if vmax > vmin + k_minus_log_epsilon {
            vmax
        } else {
            vmax + (vmin - vmax).exp().ln_1p()
        }
    }
}

#[inline]
#[cfg(test)]
fn log_sum_exp_add(max: &mut f64, sum_exp: &mut f64, value: f64) {
    // Comparisons with NaN are all false, so without this a NaN would be skipped unless it happened
    // to be the first value. `log_sum_exp` propagates NaN; keep that explicit here too.
    if value.is_nan() || max.is_nan() {
        *max = f64::NAN;
        *sum_exp = f64::NAN;
        return;
    }
    if value > *max {
        let diff = *max - value;
        *sum_exp = if diff < -50.0 {
            1.0
        } else {
            *sum_exp * diff.exp() + 1.0
        };
        *max = value;
    } else if value - *max > -50.0 {
        *sum_exp += (value - *max).exp();
    }
}

impl<'a> Lattice<'a> {
    pub fn from(sentence: &'a str, bos_id: usize, eos_id: usize) -> Self {
        let k_reserved_node_size = 16;
        // We are adding 2 tokens, bos and eos
        let mut lattice = Self {
            sentence,
            len: 0,
            nodes: Vec::with_capacity(k_reserved_node_size),
            begin_nodes: Vec::new(),
            end_nodes: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
            _bos_id: bos_id,
            _eos_id: eos_id,
        };
        lattice.set_sentence(sentence);
        lattice
    }

    pub fn set_sentence(&mut self, sentence: &'a str) {
        // Only buckets that could still hold nodes from the previous sentence need
        // clearing; buckets beyond the previous length are guaranteed empty.
        let prev_len = self.len;
        if !self.begin_nodes.is_empty() {
            let clear_through = prev_len.min(self.begin_nodes.len() - 1);
            for bucket in &mut self.begin_nodes[..=clear_through] {
                bucket.clear();
            }
            for bucket in &mut self.end_nodes[..=clear_through] {
                bucket.clear();
            }
        }
        self.nodes.clear();

        self.sentence = sentence;
        let new_len = sentence.len();
        self.len = new_len;
        const RESERVED: usize = 16;
        if self.begin_nodes.len() < new_len + 1 {
            self.begin_nodes
                .resize_with(new_len + 1, || Vec::with_capacity(RESERVED));
        }
        if self.end_nodes.len() < new_len + 1 {
            self.end_nodes
                .resize_with(new_len + 1, || Vec::with_capacity(RESERVED));
        }

        self.nodes.push(Node::new(self._bos_id, 0, 0, 0, 0.0));
        self.nodes
            .push(Node::new(self._eos_id, 1, self.len, 0, 0.0));

        self.begin_nodes[self.len].push(1);
        self.end_nodes[0].push(0);
    }

    #[inline]
    pub(crate) fn insert(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node_id = self.nodes.len();
        self.nodes.push(Node::new(id, node_id, pos, length, score));
        self.begin_nodes[pos].push(node_id);
        self.end_nodes[pos + length].push(node_id);
    }

    pub fn viterbi(&mut self) -> Vec<Node> {
        let mut node_indices = Vec::new();
        self.viterbi_indices_into(&mut node_indices);
        node_indices
            .into_iter()
            .map(|node_id| self.nodes[node_id].clone())
            .collect()
    }

    /// Fill `token_ids` with the vocabulary IDs in the best segmentation.
    ///
    /// The output buffer is cleared and reused, avoiding per-node `Node` clones.
    pub fn viterbi_token_ids(&mut self, token_ids: &mut Vec<usize>) {
        self.viterbi_indices_into(token_ids);
        for node_id in token_ids {
            *node_id = self.nodes[*node_id].id;
        }
    }

    fn viterbi_indices_into(&mut self, results: &mut Vec<usize>) {
        results.clear();
        if !self.populate_viterbi_backtrace() {
            return;
        }
        let root_id = self.begin_nodes[self.len][0];
        let mut node_id = self.nodes[root_id].prev;
        while node_id != usize::MAX {
            let prev = self.nodes[node_id].prev;
            if prev == usize::MAX {
                break;
            }
            results.push(node_id);
            node_id = prev;
        }
        results.reverse();
    }

    fn populate_viterbi_backtrace(&mut self) -> bool {
        let len = self.len;
        let mut pos = 0;
        while pos <= len {
            if self.begin_nodes[pos].is_empty() {
                return false;
            }
            for &rnode_id in &self.begin_nodes[pos] {
                self.nodes[rnode_id].prev = usize::MAX;
                let mut best_score = 0.0;
                let mut best_node = None;
                for &lnode_id in &self.end_nodes[pos] {
                    let score = self.nodes[lnode_id].backtrace_score + self.nodes[rnode_id].score;
                    if best_node.is_none() || score > best_score {
                        best_node = Some(lnode_id);
                        best_score = score
                    }
                }
                match best_node {
                    Some(bnode) => {
                        self.nodes[rnode_id].prev = bnode;
                        self.nodes[rnode_id].backtrace_score = best_score;
                    }
                    None => return false,
                }
            }
            if pos < self.len {
                pos += utf8_char_width(self.sentence.as_bytes()[pos]);
            } else {
                break;
            }
        }

        let root_id = self.begin_nodes[len][0];
        self.nodes[root_id].prev != usize::MAX
    }

    pub(crate) fn piece(&self, node: &Node) -> String {
        self.sentence[node.pos..node.pos + node.length].to_owned()
    }

    pub fn nbest(&mut self, n: usize) -> Vec<Vec<Node>> {
        match n {
            0 => vec![],
            1 => vec![self.viterbi()],
            _ => {
                // let k_reserved_hypothesis_size = 512;
                let mut agenda: Agenda = QuaternaryHeap::new();
                let mut hypotheses: Vec<Vec<Node>> = vec![];
                let eos_id = self.eos_node().node_id;
                let score = self.nodes[eos_id].score;
                let hypo = Hypothesis::new(eos_id, None, score, score);
                agenda.push(hypo);

                // Fill backtrace scores
                self.populate_viterbi_backtrace();

                while !agenda.is_empty() {
                    let top = Rc::new(agenda.pop().unwrap());
                    let node_id = top.node_id;
                    if self.nodes[node_id].id == self.bos_node().id {
                        let mut hypothesis = vec![];
                        let mut next = top.next.as_ref().unwrap();
                        while let Some(next_next) = &next.next {
                            hypothesis.push(self.nodes[next.node_id].clone());
                            next = next_next;
                        }
                        hypotheses.push(hypothesis);
                        if hypotheses.len() == n {
                            return hypotheses;
                        }
                    } else {
                        for &lnode_id in &self.end_nodes[self.nodes[node_id].pos] {
                            let fx = self.nodes[lnode_id].backtrace_score + top.gx;
                            let gx = self.nodes[lnode_id].score + top.gx;
                            let hyp = Hypothesis::new(lnode_id, Some(Rc::clone(&top)), fx, gx);
                            agenda.push(hyp);
                        }
                        // When the input is too long or contains duplicated phrases,
                        // `agenda` will get extremely big. Here we avoid this case by
                        // dynamically shrinking the agenda.
                        let k_max_agenda_size = 100_000;
                        let k_min_agenda_size = 512;
                        if agenda.len() > k_max_agenda_size {
                            let mut new_agenda = QuaternaryHeap::new();
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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn bos_node(&self) -> &Node {
        &self.nodes[self.end_nodes[0][0]]
    }
    pub(crate) fn eos_node(&self) -> &Node {
        &self.nodes[self.begin_nodes[self.len][0]]
    }

    pub fn populate_marginal(&mut self, freq: f64, expected: &mut [f64]) -> f64 {
        let len = self.len();
        let n_nodes = self.nodes.len();
        self.alpha.resize(n_nodes, 0.0);
        self.beta.resize(n_nodes, 0.0);
        self.alpha.fill(0.0);
        self.beta.fill(0.0);

        // Keep the base per-node accumulation order; per-position sums change trained scores.
        for pos in 0..=len {
            for &rnode_id in &self.begin_nodes[pos] {
                for &lnode_id in &self.end_nodes[pos] {
                    self.alpha[rnode_id] = log_sum_exp(
                        self.alpha[rnode_id],
                        self.nodes[lnode_id].score + self.alpha[lnode_id],
                        lnode_id == self.end_nodes[pos][0],
                    );
                }
            }
        }
        for pos in (0..=len).rev() {
            for &lnode_id in &self.end_nodes[pos] {
                for &rnode_id in &self.begin_nodes[pos] {
                    self.beta[lnode_id] = log_sum_exp(
                        self.beta[lnode_id],
                        self.nodes[rnode_id].score + self.beta[rnode_id],
                        rnode_id == self.begin_nodes[pos][0],
                    );
                }
            }
        }

        let eos_id = self.begin_nodes[len][0];
        let z = self.alpha[eos_id];
        for begin_nodes in self.begin_nodes.iter().take(len) {
            for &node_id in begin_nodes {
                let node = &self.nodes[node_id];
                let a = self.alpha[node_id];
                let b = self.beta[node_id];
                let total = a + node.score + b - z;
                let update = freq * total.exp();
                expected[node.id] += update;
            }
        }
        freq * z
    }

    pub(crate) fn sample(&self, theta: f64) -> Vec<Node> {
        let len = self.len();
        if len == 0 {
            return vec![];
        }
        let mut alpha = vec![0.0; self.nodes.len()];
        for pos in 0..=len {
            for &rnode_id in &self.begin_nodes[pos] {
                for &lnode_id in &self.end_nodes[pos] {
                    alpha[rnode_id] = log_sum_exp(
                        alpha[rnode_id],
                        theta * (self.nodes[lnode_id].score + alpha[lnode_id]),
                        self.nodes[lnode_id] == self.nodes[self.end_nodes[pos][0]],
                    );
                }
            }
        }

        let mut rng = rng();
        let mut results = Vec::new();
        let mut probs: Vec<f64> = vec![];
        let mut z = alpha[self.eos_node().node_id];
        let mut node_id = self.eos_node().node_id;
        loop {
            probs.clear();
            let pos = self.nodes[node_id].pos;
            for &lnode_id in &self.end_nodes[pos] {
                probs.push((alpha[lnode_id] + theta * self.nodes[lnode_id].score - z).exp())
            }
            let dist = WeightedIndex::new(&probs).unwrap();
            let index = dist.sample(&mut rng);
            node_id = self.end_nodes[pos][index];
            if self.nodes[node_id] == self.nodes[self.end_nodes[0][0]] {
                break;
            }
            z = alpha[node_id];
            results.push(self.nodes[node_id].clone());
        }
        results.reverse();
        results
    }

    pub(crate) fn sample_nbest(&mut self, n: usize, theta: f64) -> Vec<Node> {
        let nbest_paths = self.nbest(n);
        if nbest_paths.is_empty() {
            return self.viterbi();
        }

        let mut probs: Vec<f64> = Vec::with_capacity(nbest_paths.len());
        for p in &nbest_paths {
            let path_score: f64 = p.iter().map(|node| node.score).sum();
            probs.push((theta * path_score).exp());
        }

        if let Ok(dist) = WeightedIndex::new(&probs) {
            let mut rng = rng();
            let index = dist.sample(&mut rng);
            nbest_paths[index].clone()
        } else {
            nbest_paths[0].clone()
        }
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
        assert_eq!(lattice.sentence, "");

        let lattice = Lattice::from("test", 1, 2);
        assert_eq!(lattice.len(), 4);
        assert_eq!(lattice.sentence, "test");

        let bos = lattice.bos_node();
        let eos = lattice.eos_node();

        assert_eq!(bos.id, 1);
        assert_eq!(eos.id, 2);
        assert_eq!(
            lattice.nodes[*lattice.end_nodes[0].first().unwrap()].id,
            bos.id
        );
        assert_eq!(
            lattice.nodes[*lattice.begin_nodes[4].first().unwrap()].id,
            eos.id
        );

        let lattice = Lattice::from("テストab", 1, 2);
        assert_eq!(lattice.len(), 11);
        assert_eq!(lattice.sentence, "テストab");
    }

    #[test]
    fn set_sentence_clears_previous_nodes() {
        let mut lattice = Lattice::from("abc", 1, 2);
        lattice.insert(0, 1, 1.0, 3);

        lattice.set_sentence("abc");

        assert_eq!(lattice.nodes.len(), 2);
        assert_eq!(lattice.begin_nodes.iter().map(Vec::len).sum::<usize>(), 1);
        assert_eq!(lattice.end_nodes.iter().map(Vec::len).sum::<usize>(), 1);

        lattice.insert(0, 1, 1.0, 3);
        lattice.set_sentence("a");

        assert_eq!(lattice.nodes.len(), 2);
        assert_eq!(lattice.begin_nodes.len(), 4);
        assert_eq!(lattice.end_nodes.len(), 4);
        assert_eq!(lattice.begin_nodes.iter().map(Vec::len).sum::<usize>(), 1);
        assert_eq!(lattice.end_nodes.iter().map(Vec::len).sum::<usize>(), 1);
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
        let node0 = &lattice.nodes[2];
        let node1 = &lattice.nodes[3];
        let node2 = &lattice.nodes[4];
        let node3 = &lattice.nodes[5];
        let node4 = &lattice.nodes[6];
        let node5 = &lattice.nodes[7];
        let node6 = &lattice.nodes[8];

        assert_eq!(lattice.piece(node0), "A");
        assert_eq!(lattice.piece(node1), "B");
        assert_eq!(lattice.piece(node2), "あ");
        assert_eq!(lattice.piece(node3), "い");
        assert_eq!(lattice.piece(node4), "AB");
        assert_eq!(lattice.piece(node5), "Bあ");
        assert_eq!(lattice.piece(node6), "あい");

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

        assert_eq!(lattice.bos_node().id, 1);
        assert_eq!(lattice.eos_node().id, 2);
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

        assert_eq!(lattice.nodes[lattice.begin_nodes[0][0]].id, node0.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[0][1]].id, node4.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[1][0]].id, node1.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[1][1]].id, node5.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[2][0]].id, node2.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[2][1]].id, node6.id);
        assert_eq!(lattice.nodes[lattice.begin_nodes[5][0]].id, node3.id);
        assert_eq!(
            lattice.eos_node().id,
            lattice.nodes[lattice.begin_nodes[8][0]].id
        );

        assert_eq!(
            lattice.bos_node().id,
            lattice.nodes[lattice.end_nodes[0][0]].id
        );
        assert_eq!(node0.id, lattice.nodes[lattice.end_nodes[1][0]].id);
        assert_eq!(node1.id, lattice.nodes[lattice.end_nodes[2][0]].id);
        assert_eq!(node4.id, lattice.nodes[lattice.end_nodes[2][1]].id);
        assert_eq!(node2.id, lattice.nodes[lattice.end_nodes[5][0]].id);
        assert_eq!(node5.id, lattice.nodes[lattice.end_nodes[5][1]].id);
        assert_eq!(node3.id, lattice.nodes[lattice.end_nodes[8][0]].id);
        assert_eq!(node6.id, lattice.nodes[lattice.end_nodes[8][1]].id);
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

        assert_eq!(tokens(&mut lattice), ["A", "B", "C"]);

        lattice.insert(0, 2, 2.0, 6);
        assert_eq!(tokens(&mut lattice), ["AB", "C"]);

        lattice.insert(1, 2, 5.0, 7);
        assert_eq!(tokens(&mut lattice), ["A", "BC"]);

        lattice.insert(0, 3, 10.0, 8);
        assert_eq!(tokens(&mut lattice), ["ABC"]);
    }

    #[test]
    fn viterbi_token_ids_reuse_output_buffer() {
        let mut lattice = Lattice::from("ABC", 1, 2);
        lattice.insert(0, 1, 0.0, 3);
        lattice.insert(1, 1, 0.0, 4);
        lattice.insert(2, 1, 0.0, 5);
        lattice.insert(0, 3, 10.0, 8);

        let mut token_ids = vec![99];
        lattice.viterbi_token_ids(&mut token_ids);
        assert_eq!(token_ids, [8]);

        lattice.set_sentence("A");
        lattice.viterbi_token_ids(&mut token_ids);
        assert!(token_ids.is_empty());
    }

    #[test]
    fn drops_long_path_without_recursing() {
        // Pins the default test-thread stack, so a large RUST_MIN_STACK cannot hide a regression.
        std::thread::Builder::new()
            .name("drops_long_path_without_recursing".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let sentence = "0".repeat(100_000);
                let mut lattice = Lattice::from(&sentence, 1, 2);
                for pos in 0..sentence.len() {
                    lattice.insert(pos, 1, 0.0, 3);
                }
                assert_eq!(lattice.viterbi().len(), sentence.len());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    fn tokens(lattice: &mut Lattice) -> Vec<String> {
        let path = lattice.viterbi();
        let mut pieces = Vec::with_capacity(path.len());
        for node in &path {
            pieces.push(lattice.piece(node));
        }
        pieces
    }

    fn nbest_tokens(lattice: &mut Lattice, n: usize) -> Vec<Vec<String>> {
        let paths = lattice.nbest(n);
        let mut tokens = Vec::with_capacity(paths.len());
        for path in &paths {
            let pieces: Vec<String> = path.iter().map(|node| lattice.piece(node)).collect();
            tokens.push(pieces);
        }
        tokens
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

        let nbests = nbest_tokens(&mut lattice, 10);
        assert_eq!(
            nbests,
            vec![
                vec!["ABC"],
                vec!["A", "BC"],
                vec!["AB", "C"],
                vec!["A", "B", "C"]
            ]
        );

        assert!(nbest_tokens(&mut lattice, 0).is_empty());
        assert_eq!(nbest_tokens(&mut lattice, 1), vec![vec!["ABC"]]);
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
    fn test_log_sum_exp_accumulator() {
        let values = [0.0, -10.0, -60.0, 2.0, -100.0];
        let mut expected = 0.0;
        for (index, value) in values.iter().enumerate() {
            expected = log_sum_exp(expected, *value, index == 0);
        }

        let mut max = values[0];
        let mut sum_exp = 1.0;
        for &value in &values[1..] {
            log_sum_exp_add(&mut max, &mut sum_exp, value);
        }
        assert_approx_eq!(max + sum_exp.ln(), expected, 1e-12);
    }

    #[test]
    fn log_sum_exp_add_propagates_nan_in_any_position() {
        for values in [
            [f64::NAN, 0.0, 1.0],
            [0.0, f64::NAN, 1.0],
            [0.0, 1.0, f64::NAN],
        ] {
            let mut max = values[0];
            let mut sum_exp = 1.0;
            for &value in &values[1..] {
                log_sum_exp_add(&mut max, &mut sum_exp, value);
            }
            assert!((max + sum_exp.ln()).is_nan(), "{values:?}");
        }
    }

    #[test]
    fn populate_marginal_propagates_nan_from_any_end_node() {
        // `AB` and `B` both end at position 2, so the NaN must propagate in either order.
        for nan_first in [true, false] {
            let mut lattice = Lattice::from("AB", 1, 2);
            lattice.insert(0, 1, 0.0, 1); // A
            if nan_first {
                lattice.insert(1, 1, f64::NAN, 2); // B
                lattice.insert(0, 2, 0.0, 3); // AB
            } else {
                lattice.insert(0, 2, 0.0, 3); // AB
                lattice.insert(1, 1, f64::NAN, 2); // B
            }

            let mut probs = vec![0.0; 4];
            assert!(
                lattice.populate_marginal(1.0, &mut probs).is_nan(),
                "nan_first={nan_first}"
            );
        }
    }

    #[test]
    fn populate_marginal_matches_node_ordered_precision() {
        let mut lattice = Lattice::from("ABC", 0, 7);
        lattice.insert(0, 1, -0.2, 1); // A
        lattice.insert(1, 1, -0.4, 2); // B
        lattice.insert(2, 1, -0.7, 3); // C
        lattice.insert(0, 2, -0.3, 4); // AB
        lattice.insert(1, 2, -0.1, 5); // BC
        lattice.insert(0, 3, -0.8, 6); // ABC

        let mut alpha = vec![0.0; lattice.nodes.len()];
        let mut beta = vec![0.0; lattice.nodes.len()];
        for pos in 0..=lattice.len {
            for &rnode_id in &lattice.begin_nodes[pos] {
                for &lnode_id in &lattice.end_nodes[pos] {
                    alpha[rnode_id] = log_sum_exp(
                        alpha[rnode_id],
                        lattice.nodes[lnode_id].score + alpha[lnode_id],
                        lnode_id == lattice.end_nodes[pos][0],
                    );
                }
            }
        }
        for pos in (0..=lattice.len).rev() {
            for &lnode_id in &lattice.end_nodes[pos] {
                for &rnode_id in &lattice.begin_nodes[pos] {
                    beta[lnode_id] = log_sum_exp(
                        beta[lnode_id],
                        lattice.nodes[rnode_id].score + beta[rnode_id],
                        rnode_id == lattice.begin_nodes[pos][0],
                    );
                }
            }
        }

        let freq = 3.25;
        let z = alpha[lattice.begin_nodes[lattice.len][0]];
        let mut expected = vec![0.0; lattice.nodes.len()];
        for begin_nodes in lattice.begin_nodes.iter().take(lattice.len) {
            for &node_id in begin_nodes {
                let node = &lattice.nodes[node_id];
                let total = alpha[node_id] + node.score + beta[node_id] - z;
                expected[node.id] += freq * total.exp();
            }
        }
        let expected_z = freq * z;

        let mut actual = vec![0.0; lattice.nodes.len()];
        let actual_z = lattice.populate_marginal(freq, &mut actual);
        assert_eq!(actual_z.to_bits(), expected_z.to_bits());
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
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
