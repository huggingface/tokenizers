mod model;
mod trainer;
mod word;

pub type Pair = (u32, u32);

// Re-export
pub use model::*;
pub use trainer::*;
pub use word::*;
