//! Private BPE storage: coordinates, encoding, allocation and bounded directories.
mod arena;
mod id_accumulator;
mod interval_index;
mod position_chains;
mod position_storage;
pub(super) mod radix;
mod sorted_positions;

pub(super) use arena::{AllocationArena, AllocationLease};
pub(super) use id_accumulator::{IdAccumulator, IdDirectory};
pub(super) use interval_index::{IntervalCursor, IntervalIndex};
pub(super) type PositionBuffer = position_storage::PositionStorage<()>;
pub(super) use position_chains::{PositionChain, PositionChains};
pub(super) use sorted_positions::{PositionEncodingScratch, SortedPositions};

/// Invalid sorted input or a position allocation that cannot be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StorageError(pub &'static str);
impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for StorageError {}
type Result<T> = std::result::Result<T, StorageError>;
