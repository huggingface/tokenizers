const globRequire = require;

describe("pipelineExample", () => {
    // This is a hack to let us require using path similar to what the user has to use
    function require(mod: string) {
        let path = mod.slice("tokenizers/".length);
        return globRequire("../../lib/" + path);
    }

    it("shows pipeline parts", async () => {
        // START reload_tokenizer
        let { Tokenizer } = require("tokenizers/bindings/tokenizer");

        let tokenizer = Tokenizer.fromFile("data/tokenizer-wiki.json");
        // END reload_tokenizer
        // START setup_normalizer
        let { sequenceNormalizer, nfdNormalizer, stripAccentsNormalizer } = require("tokenizers/bindings/normalizers");

        let normalizer = sequenceNormalizer([nfdNormalizer(), stripAccentsNormalizer()]);
        // END setup_normalizer
        // START test_normalizer
        let normalized = normalizer.normalizeStr("Héllò hôw are ü?")
        // "Hello how are u?"
        // END test_normalizer
        expect(normalized).toEqual("Hello how are u?");
        // START replace_normalizer
        tokenizer.setNormalizer(normalizer)
        // END replace_normalizer
        // START setup_pre_tokenizer
        let { whitespacePreTokenizer } = require("tokenizers/bindings/pre_tokenizers");

        var preTokenizer = whitespacePreTokenizer();
        var preTokenized = preTokenizer.preTokenizeStr("Hello! How are you? I'm fine, thank you.");
        // END setup_pre_tokenizer
        expect(preTokenized).toEqual([
            ["Hello", [0, 5]],
            ["!", [5, 6]],
            ["How", [7, 10]],
            ["are", [11, 14]],
            ["you", [15, 18]],
            ["?", [18, 19]],
            ["I", [20, 21]],
            ["'", [21, 22]],
            ['m', [22, 23]],
            ["fine", [24, 28]],
            [",", [28, 29]],
            ["thank", [30, 35]],
            ["you", [36, 39]],
            [".", [39, 40]]
        ]);
        // START combine_pre_tokenizer
        let { sequencePreTokenizer, digitsPreTokenizer } = require("tokenizers/bindings/pre_tokenizers");

        var preTokenizer = sequencePreTokenizer([whitespacePreTokenizer(), digitsPreTokenizer(true)]);
        var preTokenized = preTokenizer.preTokenizeStr("Call 911!");
        // END combine_pre_tokenizer
        // START replace_pre_tokenizer
        tokenizer.setPreTokenizer(preTokenizer)
        // END replace_pre_tokenizer
        // START setup_processor
        let { templateProcessing } = require("tokenizers/bindings/processors");

        tokenizer.setPostProcessor(templateProcessing(
            "[CLS] $A [SEP]",
            "[CLS] $A [SEP] $B:1 [SEP]:1",
            [["[CLS]", 1], ["[SEP]", 2]]
        ));
        // END setup_processor
    });

    it("shows a full bert example", async () => {
        // START bert_setup_tokenizer
        let { Tokenizer } = require("tokenizers/bindings/tokenizer");
        let { WordPiece } = require("tokenizers/bindings/models");

        let bert_tokenizer = Tokenizer(WordPiece.empty());
        // END bert_setup_tokenizer
        // START bert_setup_normalizer
        let { sequenceNormalizer, lowercaseNormalizer, nfdNormalizer, stripAccentsNormalizer }
            = require("tokenizers/bindings/normalizers");

        bert_tokenizer.setNormalizer(sequenceNormalizer([
            nfdNormalizer(), lowercaseNormalizer(), stripAccentsNormalizer()
        ]))
        // END bert_setup_normalizer
        // START bert_setup_pre_tokenizer
        let { whitespacePreTokenizer } = require("tokenizers/bindings/pre_tokenizers");

        bert_tokenizer.setPreTokenizer = whitespacePreTokenizer();
        // END bert_setup_pre_tokenizer
        // START bert_setup_processor
        let { templateProcessing } = require("tokenizers/bindings/processors");

        bert_tokenizer.setPostProcessor(templateProcessing(
            "[CLS] $A [SEP]",
            "[CLS] $A [SEP] $B:1 [SEP]:1",
            [["[CLS]", 1], ["[SEP]", 2]]
        ));
        // END bert_setup_processor
        // START bert_train_tokenizer
        let { wordPieceTrainer } = require("tokenizers/bindings/trainers");
        let { promisify } = require("utils");

        let trainer = wordPieceTrainer({
            vocabSize: 30522,
            specialTokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        });
        let files = ["test", "train", "valid"].map(split => `data/wikitext-103-raw/wiki.${split}.raw`);
        bert_tokenizer.train(trainer, files);

        let model_files = bert_tokenizer.getModel.save("data", "bert-wiki");
        let fromFile = promisify(WordPiece.fromFile);
        bert_tokenizer.setModel(await fromFile(model_files[0], {
            unkToken: "[UNK]"
        }));

        bert_tokenizer.save("data/bert-wiki.json")
        // END bert_train_tokenizer
    });
});
