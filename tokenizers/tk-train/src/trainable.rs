#[cfg(feature = "bpe")]
use tk_encode::models::bpe::PipelineBPE;
#[cfg(feature = "unigram")]
use tk_encode::models::unigram::Unigram;
#[cfg(feature = "wordlevel")]
use tk_encode::models::wordlevel::WordLevel;
#[cfg(feature = "wordpiece")]
use tk_encode::models::wordpiece::WordPiece;

use crate::Trainer;
#[cfg(feature = "bpe")]
use crate::trainers::BpeTrainer;
use crate::trainers::TrainerWrapper;
#[cfg(feature = "unigram")]
use crate::trainers::UnigramTrainer;
#[cfg(feature = "wordlevel")]
use crate::trainers::WordLevelTrainer;
#[cfg(feature = "wordpiece")]
use crate::trainers::WordPieceTrainer;

/// One of the trainable models compiled in, so that [`TrainerWrapper`] has a single `Model` to name.
///
/// This is the dispatch half of what `tk-convert`'s `ModelWrapper` used to be. The other half was a
/// hand-written `Deserialize` that read a legacy `model` object -- config-layer work that went with
/// that crate's strip, and that a trainer never needed.
/// No `Debug`/`Clone`: `PipelineBPE` derives neither, and no caller needs them.
#[allow(clippy::large_enum_variant)]
pub enum ModelWrapper {
    #[cfg(feature = "bpe")]
    BPE(PipelineBPE),
    #[cfg(feature = "wordpiece")]
    WordPiece(WordPiece),
    #[cfg(feature = "wordlevel")]
    WordLevel(WordLevel),
    #[cfg(feature = "unigram")]
    Unigram(Unigram),
}

#[cfg(feature = "bpe")]
impl From<PipelineBPE> for ModelWrapper {
    fn from(m: PipelineBPE) -> Self {
        Self::BPE(m)
    }
}
#[cfg(feature = "wordpiece")]
impl From<WordPiece> for ModelWrapper {
    fn from(m: WordPiece) -> Self {
        Self::WordPiece(m)
    }
}
#[cfg(feature = "wordlevel")]
impl From<WordLevel> for ModelWrapper {
    fn from(m: WordLevel) -> Self {
        Self::WordLevel(m)
    }
}
#[cfg(feature = "unigram")]
impl From<Unigram> for ModelWrapper {
    fn from(m: Unigram) -> Self {
        Self::Unigram(m)
    }
}

/// A `Model` that knows how to build a `Trainer` capable of training it.
///
/// In v1 this was the `type Trainer` / `get_trainer` part of the `Model` trait.
/// It now lives in `tk-train` so that `tk-encode` (inference) carries no
/// training-related coupling.
pub trait Trainable {
    type Trainer: Trainer<Model = Self> + Sync;
    /// Get an instance of a Trainer capable of training this Model.
    fn get_trainer(&self) -> Self::Trainer;
}

#[cfg(feature = "bpe")]
impl Trainable for PipelineBPE {
    type Trainer = BpeTrainer;
    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

#[cfg(feature = "unigram")]
impl Trainable for Unigram {
    type Trainer = UnigramTrainer;
    fn get_trainer(&self) -> UnigramTrainer {
        UnigramTrainer::default()
    }
}

#[cfg(feature = "wordlevel")]
impl Trainable for WordLevel {
    type Trainer = WordLevelTrainer;
    fn get_trainer(&self) -> WordLevelTrainer {
        WordLevelTrainer::default()
    }
}

#[cfg(feature = "wordpiece")]
impl Trainable for WordPiece {
    type Trainer = WordPieceTrainer;
    fn get_trainer(&self) -> WordPieceTrainer {
        WordPieceTrainer::builder().build()
    }
}

impl Trainable for ModelWrapper {
    type Trainer = TrainerWrapper;
    fn get_trainer(&self) -> TrainerWrapper {
        match self {
            #[cfg(feature = "wordlevel")]
            Self::WordLevel(t) => t.get_trainer().into(),
            #[cfg(feature = "wordpiece")]
            Self::WordPiece(t) => t.get_trainer().into(),
            #[cfg(feature = "bpe")]
            Self::BPE(t) => t.get_trainer().into(),
            #[cfg(feature = "unigram")]
            Self::Unigram(t) => t.get_trainer().into(),
        }
    }
}
