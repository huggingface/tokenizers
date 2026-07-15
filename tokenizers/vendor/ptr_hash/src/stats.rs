use crate::Pilot;

#[derive(Default, Clone, serde::Serialize, Debug)]
struct Row {
    buckets: usize,
    elements: usize,
    elements_max: usize,
    pilot_sum: Pilot,
    pilot_max: Pilot,
    evictions: usize,
    evictions_max: usize,
}

impl Row {
    fn add(&mut self, bucket_len: usize, pilot: Pilot, evictions: usize) {
        self.buckets += 1;
        self.elements += bucket_len;
        self.elements_max = self.elements_max.max(bucket_len);
        self.pilot_sum += pilot;
        self.pilot_max = self.pilot_max.max(pilot);
        self.evictions += evictions;
        self.evictions_max = self.evictions_max.max(evictions);
    }
}

#[derive(Default, serde::Serialize, Debug)]
pub struct BucketStats {
    by_pct: Vec<Row>,
    by_bucket_len: Vec<Row>,
}

impl BucketStats {
    pub fn new() -> Self {
        Self {
            by_pct: vec![Row::default(); 100],
            by_bucket_len: vec![Row::default(); 100],
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.by_pct.resize(100, Row::default());
        self.by_bucket_len.resize(
            self.by_bucket_len.len().max(other.by_bucket_len.len()),
            Row::default(),
        );
        for (a, b) in self.by_pct.iter_mut().zip(other.by_pct.iter()) {
            a.buckets += b.buckets;
            a.elements += b.elements;
            a.elements_max = a.elements_max.max(b.elements_max);
            a.pilot_sum += b.pilot_sum;
            a.pilot_max = a.pilot_max.max(b.pilot_max);
            a.evictions += b.evictions;
            a.evictions_max = a.evictions_max.max(b.evictions_max);
        }
        for (a, b) in self
            .by_bucket_len
            .iter_mut()
            .zip(other.by_bucket_len.iter())
        {
            a.buckets += b.buckets;
            a.elements += b.elements;
            a.elements_max = a.elements_max.max(b.elements_max);
            a.pilot_sum += b.pilot_sum;
            a.pilot_max = a.pilot_max.max(b.pilot_max);
            a.evictions += b.evictions;
            a.evictions_max = a.evictions_max.max(b.evictions_max);
        }
    }

    pub fn add(
        &mut self,
        bucket_id: usize,
        buckets_total: usize,
        bucket_len: usize,
        pilot: Pilot,
        evictions: usize,
    ) {
        let pct = bucket_id * 100 / buckets_total;
        self.by_pct[pct].add(bucket_len, pilot, evictions);
        if self.by_bucket_len.len() <= bucket_len {
            self.by_bucket_len.resize(bucket_len + 1, Row::default());
        }
        self.by_bucket_len[bucket_len].add(bucket_len, pilot, evictions);
    }

    pub fn print(&self) {
        eprintln!();
        Self::print_rows(&self.by_pct, false);
        // eprintln!();
        // Self::print_rows(&self.by_bucket_len, true);
        eprintln!();
    }

    fn print_rows(rows: &[Row], reverse: bool) {
        let b_total = rows.iter().map(|r| r.buckets).sum::<usize>();
        let n = rows.iter().map(|r| r.elements).sum::<usize>();

        eprintln!(
            "{:>4}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
            "sz",
            "cnt",
            "bucket%",
            "cuml%",
            "elem%",
            "cuml%",
            "avg p",
            "max p",
            "avg evict",
            "max evict"
        );
        let mut bucket_cuml = 0;
        let mut elem_cuml = 0;
        let process_row = |row: &Row| {
            if row.buckets == 0 {
                return;
            }
            bucket_cuml += row.buckets;
            elem_cuml += row.elements;
            eprintln!(
                "{:>4}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10.5} {:>10}",
                row.elements_max,
                row.buckets,
                row.buckets as f32 / b_total as f32 * 100.,
                bucket_cuml as f32 / b_total as f32 * 100.,
                row.elements as f32 / n as f32 * 100.,
                elem_cuml as f32 / n as f32 * 100.,
                row.pilot_sum as f32 / row.buckets as f32,
                row.pilot_max,
                row.evictions as f32 / row.buckets as f32,
                row.evictions_max
            );
        };
        if reverse {
            rows.iter().rev().for_each(process_row);
        } else {
            rows.iter().for_each(process_row);
        }
        let sum_pilots = rows.iter().map(|r| r.pilot_sum).sum::<Pilot>();
        let max_pilot = rows.iter().map(|r| r.pilot_max).max().unwrap();
        let sum_evictions = rows.iter().map(|r| r.evictions).sum::<usize>();
        let max_evictions = rows.iter().map(|r| r.evictions_max).max().unwrap();
        eprintln!(
            "{:>4}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10.5} {:>10}",
            "",
            b_total,
            100.,
            100.,
            100.,
            100.,
            sum_pilots as f32 / b_total as f32,
            max_pilot,
            sum_evictions as f32 / b_total as f32,
            max_evictions
        );
    }
}
