"""Recipes for learning a vocabulary from text."""

from .._native import trainers as _trainers

Trainer = _trainers.Trainer
BpeTrainer = _trainers.BpeTrainer
UnigramTrainer = _trainers.UnigramTrainer
WordLevelTrainer = _trainers.WordLevelTrainer
WordPieceTrainer = _trainers.WordPieceTrainer

__all__ = [
    "Trainer",
    "BpeTrainer",
    "UnigramTrainer",
    "WordLevelTrainer",
    "WordPieceTrainer",
]
