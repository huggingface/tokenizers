"""Train a parity-aware BPE tokenizer over several languages at once.

`ParityBpeTrainer` takes one iterator per language and balances merges across them, so a
low-resource language is not left far worse compressed than a high-resource one. It needs
a per-language signal to do that, supplied either as a parallel dev set (`dev_iterators=`)
or as target compression rates (`ratio=`). Both are shown below.

Per-language corpora are not something you are likely to have on disk, so the data comes
from the Hub: training text from `wikimedia/wikipedia`, and a parallel dev set from
`openlanguagedata/flores_plus` (997 sentences per language, aligned across languages).

FLORES+ is gated: accept the terms at
https://huggingface.co/datasets/openlanguagedata/flores_plus and run `huggingface-cli login`
before running the dev-set demo. The ratio demo needs neither.
"""

import datasets

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import ParityBpeTrainer


WIKIPEDIA = "wikimedia/wikipedia"
FLORES_PLUS = "openlanguagedata/flores_plus"

# Wikipedia uses ISO 639-1 codes, FLORES+ uses ISO 639-3 plus a script tag, so each
# language needs both: (display name, wikipedia config, flores+ config, sample sentence).
LANGUAGES = [
    ("English", "20231101.en", "eng_Latn", "The quick brown fox jumps over the lazy dog."),
    ("German", "20231101.de", "deu_Latn", "Der schnelle braune Fuchs springt über den faulen Hund."),
    ("French", "20231101.fr", "fra_Latn", "Le rapide renard brun saute par-dessus le chien paresseux."),
    ("Hindi", "20231101.hi", "hin_Deva", "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"),
]

# Target compression rates, one per language, in the order of LANGUAGES. Only the values
# relative to each other matter: the trainer picks whichever language has the lowest
# compression_rate / ratio, so a higher ratio earns more merges.
#
# ByteLevel counts compression in bytes, so equal ratios would not give equal tokenization.
# Mean bytes per FLORES+ sentence are English 125.7, German 149.4, French 155.0, Hindi
# 323.2, because Devanagari is 3 bytes per character in UTF-8 while its character count is
# close to English. These ratios are those byte counts normalized by English, which brings
# the per-language token counts closer together (in one run, Hindi went from 94.2 to 82.0
# tokens per sentence and the spread between best and worst from 2.56x to 1.75x).
RATIOS = [1.00, 1.19, 1.23, 2.57]

# Wikipedia articles average around 20 KB, and `train_from_iterator` materializes each
# language's corpus in memory before training, so peak memory scales with these caps.
# The defaults keep the demo near 20 MB of text. Raise both for a tokenizer you intend to use.
ARTICLES_PER_LANGUAGE = 200
NUM_MERGES = 8000


def wikipedia_iterator(config, limit=ARTICLES_PER_LANGUAGE):
    """Stream up to `limit` articles for one language, so no full dump is downloaded."""
    dataset = datasets.load_dataset(WIKIPEDIA, config, split="train", streaming=True)
    for count, article in enumerate(dataset):
        if count >= limit:
            return
        yield article["text"]


def flores_iterator(config):
    """The FLORES+ dev split: 997 sentences, the same ones in every language."""
    dataset = datasets.load_dataset(FLORES_PLUS, config, split="dev")
    yield from dataset["text"]


def new_tokenizer():
    """Byte-level BPE, so every language is representable and compression is counted in bytes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def new_trainer():
    return ParityBpeTrainer(
        num_merges=NUM_MERGES,
        variant="window",
        window_size=100,
        alpha=2.0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )


def report(tokenizer, label):
    """Token count per language on one sentence, so the balance across languages is visible."""
    print(f"\n{label}: vocabulary {tokenizer.get_vocab_size()}")
    for name, _, _, sentence in LANGUAGES:
        print(f"  {name:8s} {len(tokenizer.encode(sentence).tokens):3d} tokens")


# Mode 1: balance against a parallel dev set.
dev_set_tokenizer = new_tokenizer()
new_trainer().train_from_iterator(
    dev_set_tokenizer,
    train_iterators=[wikipedia_iterator(wiki) for _, wiki, _, _ in LANGUAGES],
    dev_iterators=[flores_iterator(flores) for _, _, flores, _ in LANGUAGES],
)
dev_set_tokenizer.save("parity_bpe_devset.json")
report(dev_set_tokenizer, "dev-set balancing")

# Mode 2: balance against target compression rates. No parallel data is involved, and the
# training iterators are rebuilt because generators are single-use.
ratio_tokenizer = new_tokenizer()
new_trainer().train_from_iterator(
    ratio_tokenizer,
    train_iterators=[wikipedia_iterator(wiki) for _, wiki, _, _ in LANGUAGES],
    ratio=RATIOS,
)
ratio_tokenizer.save("parity_bpe_ratio.json")
report(ratio_tokenizer, "ratio balancing")

# Neither mode reaches equal token counts, which is expected. Merges go into one shared
# vocabulary, so how well a language compresses depends on which other languages are in
# the mixture: English, German and French share the Latin script and much substring
# structure, so a merge selected for German often helps the other two, while Hindi shares
# nothing with them and has to fund its whole subword inventory from merges selected for
# Hindi. The byte budget hides a second imbalance: the four corpora are within 20% of each
# other in bytes, but Devanagari averages 2.6 bytes per character, so Hindi contributes
# about 2.2M characters of running text against English's 6.2M. `ratio` also targets a rate
# measured on the training data while parity is judged on held-out parallel text, and the
# window variant lets no language hold more than alpha / n_languages of the window (half,
# here), so a language cannot be given an unbroken run of merges to catch up. Raising
# NUM_MERGES and ARTICLES_PER_LANGUAGE closes the remaining gap more reliably than ratio
# tuning.
