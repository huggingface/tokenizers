/// Immutable full-width interval boundaries. Repeated boundaries form empty intervals.
pub(in super::super) struct IntervalIndex<T> {
    starts: Vec<u64>,
    values: Vec<T>,
}
impl<T> IntervalIndex<T> {
    pub(in super::super) fn new(starts: Vec<u64>, values: Vec<T>) -> Self {
        assert_eq!(starts.len(), values.len());
        assert!(starts.is_sorted());
        Self { starts, values }
    }
    #[inline]
    pub(in super::super) fn interval_containing(&self, position: u64) -> Option<usize> {
        self.starts
            .partition_point(|&start| start <= position)
            .checked_sub(1)
    }
    #[cfg(test)]
    pub(in super::super) fn get(&self, position: u64) -> Option<&T> {
        self.interval_containing(position).map(|i| &self.values[i])
    }
    pub(in super::super) fn cursor(&self) -> IntervalCursor<'_, T> {
        IntervalCursor {
            index: self,
            current: self
                .starts
                .first()
                .map(|&start| (start, self.starts.get(1).copied(), &self.values[0])),
        }
    }
    /// Group sorted coordinates by interval. The count includes duplicates;
    /// coordinates before the first interval produce a `None` value.
    /// The coordinate function must preserve the input's nondecreasing order.
    pub(in super::super) fn runs_for_sorted<U>(
        &self,
        mut items: &[U],
        coordinate: impl Fn(&U) -> u64,
    ) -> impl Iterator<Item = (usize, Option<&T>)> {
        std::iter::from_fn(move || {
            let first = items.first()?;
            let interval = self.interval_containing(coordinate(first));
            let boundary = self.starts.get(interval.map_or(0, |i| i + 1)).copied();
            let before_end = |item: &U| boundary.is_none_or(|end| coordinate(item) < end);
            // PERF: One galloping search replaces a lookup and accumulation for
            // each coordinate in a long same-interval run. Short runs stop early.
            let count = if before_end(items.last().unwrap()) {
                items.len()
            } else {
                let mut low = 1;
                let mut high = 2.min(items.len());
                while high < items.len() && before_end(&items[high - 1]) {
                    low = high;
                    high = high.saturating_mul(2).min(items.len());
                }
                low + items[low..high].partition_point(before_end)
            };
            items = &items[count..];
            Some((count, interval.map(|i| &self.values[i])))
        })
    }
    pub(in super::super) fn values(&self) -> &[T] {
        &self.values
    }
}
/// Cache the last matching interval. Cache misses binary-search the boundaries;
/// full-width queries may move in either direction.
pub(in super::super) struct IntervalCursor<'index, T> {
    index: &'index IntervalIndex<T>,
    current: Option<(u64, Option<u64>, &'index T)>,
}
impl<'index, T> IntervalCursor<'index, T> {
    pub(in super::super) fn get(&mut self, position: u64) -> Option<&'index T> {
        if let Some((start, end, value)) = self.current
            && position >= start
            && end.is_none_or(|end| position < end)
        {
            return Some(value);
        }
        self.current = self.index.interval_containing(position).map(|i| {
            (
                self.index.starts[i],
                self.index.starts.get(i + 1).copied(),
                &self.index.values[i],
            )
        });
        self.current.map(|(_, _, value)| value)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sorted_runs_match_scalar_queries_across_full_width_boundaries() {
        let intervals = IntervalIndex::new(
            vec![1, 1 << 32, 1 << 32, 1 << 63, u64::MAX],
            vec![3, 5, 7, 11, 13],
        );
        let positions: Vec<_> = (0..1024)
            .chain([1 << 32, 1 << 32, 1 << 63, u64::MAX])
            .collect();
        let scalar: Vec<_> = positions
            .iter()
            .map(|&p| intervals.get(p).copied())
            .collect();
        let runs: Vec<_> = intervals
            .runs_for_sorted(&positions, |&p| p)
            .flat_map(|(count, value)| std::iter::repeat_n(value.copied(), count))
            .collect();
        assert_eq!(runs, scalar);
        assert_eq!(intervals.runs_for_sorted(&[] as &[u64], |&p| p).count(), 0);
    }
    #[test]
    fn boundaries_duplicates_and_backwards_queries() {
        let index = IntervalIndex::new(
            vec![1, 1 << 32, 1 << 32, 1 << 63, u64::MAX],
            vec![3, 5, 7, 11, 13],
        );
        let mut cursor = index.cursor();
        for (position, value) in [
            (0, None),
            (1, Some(3)),
            ((1 << 32) - 1, Some(3)),
            (1 << 32, Some(7)),
            (1 << 63, Some(11)),
            (u64::MAX, Some(13)),
            (2, Some(3)),
        ] {
            assert_eq!(cursor.get(position).copied(), value);
            assert_eq!(index.get(position).copied(), value);
        }
    }
}
