var globRequire = require;

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
        // START test_decoding
        let output = tokenizer.encode("Hello, y'all! How are you 😁 ?");
        console.log(output.getIds());
        // [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]

        tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]);
        // "Hello , y ' all ! How are you ?"
        // END test_decoding
    });

    var { Tokenizer } = require("tokenizers/bindings/tokenizer");
    const slow_bert_training = async (bertTokenizer: typeof Tokenizer) => {
        let { WordPiece } = require("tokenizers/bindings/models");

        // START bert_train_tokenizer
        let { wordPieceTrainer } = require("tokenizers/bindings/trainers");
        let { promisify } = require("util");

        let trainer = wordPieceTrainer({
            vocabSize: 30522,
            specialTokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        });
        let files = ["test", "train", "valid"].map(split => `data/wikitext-103-raw/wiki.${split}.raw`);
        bertTokenizer.train(trainer, files);

        let modelFiles = bertTokenizer.getModel.save("data", "bert-wiki");
        let fromFile = promisify(WordPiece.fromFile);
        bertTokenizer.setModel(await fromFile(modelFiles[0], {
            unkToken: "[UNK]"
        }));

        bertTokenizer.save("data/bert-wiki.json")
        // END bert_train_tokenizer
    };
    console.log(slow_bert_training); // disable unused warning

    it("shows a full bert example", async () => {
        // START bert_setup_tokenizer
        let { Tokenizer } = require("tokenizers/bindings/tokenizer");
        let { WordPiece } = require("tokenizers/bindings/models");

        let bertTokenizer = new Tokenizer(WordPiece.empty());
        // END bert_setup_tokenizer
        // START bert_setup_normalizer
        let { sequenceNormalizer, lowercaseNormalizer, nfdNormalizer, stripAccentsNormalizer }
            = require("tokenizers/bindings/normalizers");

        bertTokenizer.setNormalizer(sequenceNormalizer([
            nfdNormalizer(), lowercaseNormalizer(), stripAccentsNormalizer()
        ]))
        // END bert_setup_normalizer
        // START bert_setup_pre_tokenizer
        let { whitespacePreTokenizer } = require("tokenizers/bindings/pre-tokenizers");

        bertTokenizer.setPreTokenizer = whitespacePreTokenizer();
        // END bert_setup_pre_tokenizer
        // START bert_setup_processor
        let { templateProcessing } = require("tokenizers/bindings/post-processors");

        bertTokenizer.setPostProcessor(templateProcessing(
            "[CLS] $A [SEP]",
            "[CLS] $A [SEP] $B:1 [SEP]:1",
            [["[CLS]", 1], ["[SEP]", 2]]
        ));
        // END bert_setup_processor
        // START bert_test_decoding
        let output = bertTokenizer.encode("Welcome to the 🤗 Tokenizers library.");
        console.log(output.getTokens());
        // ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]

        bertTokenizer.decode(output.getIds());
        // "welcome to the tok ##eni ##zer ##s library ."
        // END bert_test_decoding
        // START bert_proper_decoding
        let { wordPieceDecoder } = require("tokenizers/bindings/decoders");
        bertTokenizer.setDecoder(wordPieceDecoder());
        bertTokenizer.decode(output.ids);
        // "welcome to the tokenizers library."
        // END bert_proper_decoding
    });
});
