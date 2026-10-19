//! Compact change references and stable routing to pair-count owners.
use super::super::{
    pair_index::ShardRouter,
    storage::{PositionChain, PositionChains},
};

/// One neighbor's removal and birth are committed together. With reusable IDs,
/// the two keys can coincide.
pub(in super::super) struct PairChanges {
    pub(in super::super) removed_key: u64,
    pub(in super::super) born_key: u64,
    pub(in super::super) removed_weight: u64,
    pub(in super::super) born_weight: u64,
    pub(in super::super) positions: PositionChain,

    pub(in super::super) bucket: u32,
}
#[derive(Clone, Copy)]
pub(in super::super) enum ChangeAction {
    Remove,
    Birth,
    Both,
}
/// Two low tag bits share one word with a record index. A resident Vec of
/// 48-byte records bounds its indices well below the available upper bits.
pub(in super::super) struct RoutedChangeRef {
    pub(in super::super) chunk: usize,
    index_and_action: usize,
}
impl RoutedChangeRef {
    fn new(chunk: usize, index: usize, action: ChangeAction) -> Self {
        Self {
            chunk,
            index_and_action: (index << 2) | action as usize,
        }
    }
    pub(in super::super) fn index(&self) -> usize {
        self.index_and_action >> 2
    }
    pub(in super::super) fn action(&self) -> ChangeAction {
        match self.index_and_action & 3 {
            0 => ChangeAction::Remove,
            1 => ChangeAction::Birth,
            2 => ChangeAction::Both,
            _ => unreachable!("record tags are constructed from ChangeAction"),
        }
    }
}
/// One owner's original-order count actions and separately grouped birth indices.
/// Birth grouping may reorder only `births`; `changes` preserves checked update
/// order, including removal-before-birth when both keys route to the same owner.
#[derive(Default)]
pub(in super::super) struct OwnerRoute {
    pub(in super::super) changes: Vec<RoutedChangeRef>,
    pub(in super::super) births: Vec<usize>,
    // Scratch retains capacity; grouping overwrites its active entries before
    // reading them. Offsets need resetting only when there are multiple births.
    grouped_births: Vec<usize>,
    bucket_offsets: Vec<usize>,
}
pub(in super::super) struct EventChunk {
    pub(in super::super) chains: PositionChains,
    pub(in super::super) changes: Vec<PairChanges>,
}
/// Owns buffered position chains until the joined commit finishes borrowing them.
/// Already encoded complete births retain only removal events here.
pub(in super::super) struct MergeEvents {
    pub(in super::super) buckets: usize,
    pub(in super::super) chunks: Vec<EventChunk>,
}
impl MergeEvents {
    /// One directory per owner for the whole batch. Only actual actions and
    /// births occupy entries; producers do not allocate an owner/bucket matrix.
    #[cfg(test)]
    pub(in super::super) fn route(&self, workers: usize) -> Vec<OwnerRoute> {
        {
            let mut routes = (0..workers)
                .map(|_| OwnerRoute::default())
                .collect::<Vec<_>>();
            self.dispatch_into(&mut routes, ShardRouter::new(workers));
            for route in &mut routes {
                route.group_births(self);
            }
            routes
        }
    }

    /// Route only metadata. Birth grouping can run inside the owner task.
    pub(in super::super) fn dispatch_into(
        &self,
        routes: &mut Vec<OwnerRoute>,
        router: ShardRouter,
    ) {
        routes.resize_with(router.shards(), OwnerRoute::default);
        for route in routes.iter_mut() {
            route.clear();
        }
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            for (index, change) in chunk.changes.iter().enumerate() {
                debug_assert!((change.bucket as usize) < self.buckets);
                let removed =
                    (change.removed_weight != 0).then(|| router.owner(change.removed_key));
                // Zero-weight identity-reuse births still own positions. Only an empty
                // chain has no birth action; weight alone cannot decide this.
                let born = (!change.positions.is_empty()).then(|| router.owner(change.born_key));
                match (removed, born) {
                    (Some(removed), Some(born)) if removed == born => {
                        let route = &mut routes[removed];
                        route.births.push(route.changes.len());
                        route.changes.push(RoutedChangeRef::new(
                            chunk_index,
                            index,
                            ChangeAction::Both,
                        ));
                    }
                    (removed, born) => {
                        if let Some(owner) = removed {
                            routes[owner].changes.push(RoutedChangeRef::new(
                                chunk_index,
                                index,
                                ChangeAction::Remove,
                            ));
                        }
                        if let Some(owner) = born {
                            let route = &mut routes[owner];
                            route.births.push(route.changes.len());
                            route.changes.push(RoutedChangeRef::new(
                                chunk_index,
                                index,
                                ChangeAction::Birth,
                            ));
                        }
                    }
                }
            }
        }
    }
}
impl OwnerRoute {
    pub(in super::super) fn clear(&mut self) {
        self.changes.clear();
        self.births.clear();
        self.grouped_births.clear();
    }

    #[cfg(test)]
    pub(in super::super) fn assert_cleared(&self) {
        assert!(self.changes.is_empty(), "routed count actions remain");
        assert!(self.births.is_empty(), "routed birth indices remain");
        assert!(
            self.grouped_births.is_empty(),
            "grouping scratch entries remain"
        );
    }

    /// Stably group birth references by rule/direction without changing actions.
    /// Equal-bucket fragments keep producer order for the fresh spatial encoder;
    /// reuse uses an encoder that also handles genuinely interleaved chains.
    pub(in super::super) fn group_births(&mut self, events: &MergeEvents) {
        if self.births.len() < 2 {
            return;
        }
        let bucket_of = |index: usize| {
            let reference = &self.changes[index];
            events.chunks[reference.chunk].changes[reference.index()].bucket as usize
        };
        self.bucket_offsets.resize(events.buckets, 0);
        self.bucket_offsets[..events.buckets].fill(0);
        for &index in &self.births {
            self.bucket_offsets[bucket_of(index)] += 1;
        }
        let mut total = 0;
        for offset in &mut self.bucket_offsets[..events.buckets] {
            let count = *offset;
            *offset = total;
            total += count;
        }
        self.grouped_births.resize(self.births.len(), 0);
        for &index in &self.births {
            let offset = &mut self.bucket_offsets[bucket_of(index)];
            self.grouped_births[*offset] = index;
            *offset += 1;
        }
        std::mem::swap(&mut self.births, &mut self.grouped_births);
    }
}
