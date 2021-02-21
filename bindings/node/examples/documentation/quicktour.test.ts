/* eslint-disable */
var globRequire = require;

describe("quicktourExample", () => {
    function require(mod: string) {
        if (mod.startsWith("tokenizers/")) {
            let path = mod.slice("tokenizers/".length);
            return globRequire("../../lib/" + path);
        } else {
            return globRequire(mod);
        }
    }

    it.skip("trains the tokenizer", async () => {
        // START init_tokenizer
        let { Tokenizer } = require("tokenizers/bindings/tokenizer");
        let { BPE } = require("tokenizers/bindings/models");

        let tokenizer = new Tokenizer(BPE.init({}, [], { unkToken: "[UNK]" }));
        // END init_tokenizer
        // START init_trainer
        let { bpeTrainer } = require("tokenizers/bindings/trainers");

        let trainer = bpeTrainer({
            specialTokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        });
        // END init_trainer
        // START init_pretok
        let { whitespacePreTokenizer } = require("tokenizers/bindings/pre-tokenizers");

        tokenizer.setPreTokenizer(whitespacePreTokenizer());
        // END init_pretok
        // START train
        let files = ["test", "train", "valid"].map(split => `data/wikitext-103-raw/wiki.${split}.raw`);
        tokenizer.train(files, trainer);
        // END train
        // START save
        tokenizer.save("data/tokenizer-wiki.json");
        // END save
    });

    it("shows a quicktour example", async () => {
        let { Tokenizer } = require("tokenizers/bindings/tokenizer");
        let console = {
            log: (..._args: any[]) => {}
        };

        // START reload_tokenizer
        let tokenizer = Tokenizer.fromFile("data/tokenizer-wiki.json");
        // END reload_tokenizer
        // START encode
        let { promisify } = require('util');
        let encode = promisify(tokenizer.encode.bind(tokenizer));

        var output = await encode("Hello, y'all! How are you 😁 ?");
        // END encode
        // START print_tokens
        console.log(output.getTokens());
        // ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
        // END print_tokens
        expect(output.getTokens()).toEqual([
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
        ]);
        // START print_ids
        console.log(output.getIds());
        // [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
        // END print_ids
        expect(output.getIds()).toEqual([27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]);
        // START print_offsets
        let offsets = output.getOffsets();
        console.log(offsets[9]);
        // (26, 27)
        // END print_offsets
        expect(offsets[9]).toEqual([26, 27]);
        // START use_offsets
        let { slice } = require("tokenizers/bindings/utils");

        let sentence = "Hello, y'all! How are you 😁 ?"
        let [start, end] = offsets[9];
        console.log(slice(sentence, start, end));
        // "😁"
        // END use_offsets
        expect(slice(sentence, start, end)).toEqual("😁");
        // START check_sep
        console.log(tokenizer.tokenToId("[SEP]"));
        // 2
        // END check_sep
        expect(tokenizer.tokenToId("[SEP]")).toEqual(2);
        // START init_template_processing
        let { templateProcessing } = require("tokenizers/bindings/post-processors");

        tokenizer.setPostProcessor(templateProcessing(
            "[CLS] $A [SEP]",
            "[CLS] $A [SEP] $B:1 [SEP]:1",
            [
                ["[CLS]", tokenizer.tokenToId("[CLS]")],
                ["[SEP]", tokenizer.tokenToId("[SEP]")],
            ],
        ));
        // END init_template_processing
        // START print_special_tokens
        var output = await encode("Hello, y'all! How are you 😁 ?");
        console.log(output.getTokens());
        // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
        // END print_special_tokens
        expect(output.getTokens()).toEqual([
            "[CLS]",
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
            "[SEP]",
        ]);
        // START print_special_tokens_pair
        var output = await encode("Hello, y'all!", "How are you 😁 ?");
        console.log(output.getTokens());
        // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
        // END print_special_tokens_pair
        expect(output.getTokens()).toEqual([
            "[CLS]",
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "[SEP]",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
            "[SEP]",
        ]);
        // START print_type_ids
        console.log(output.getTypeIds());
        // [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        // END print_type_ids
        expect(output.getTypeIds()).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
        // START encode_batch
        let encodeBatch = promisify(tokenizer.encodeBatch.bind(tokenizer));

        var output = await encodeBatch(["Hello, y'all!", "How are you 😁 ?"]);
        // END encode_batch
        // START encode_batch_pair
        var output = await encodeBatch(
            [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
        );
        // END encode_batch_pair
        // START enable_padding
        tokenizer.setPadding({ padId: 3, padToken: "[PAD]" });
        // END enable_padding
        // START print_batch_tokens
        var output = await encodeBatch(["Hello, y'all!", "How are you 😁 ?"]);
        console.log(output[1].getTokens());
        // ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
        // END print_batch_tokens
        expect(output[1].getTokens()).toEqual(["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]);
        // START print_attention_mask
        console.log(output[1].getAttentionMask());
        // [1, 1, 1, 1, 1, 1, 1, 0]
        // END print_attention_mask
        expect(output[1].getAttentionMask()).toEqual([1, 1, 1, 1, 1, 1, 1, 0]);
    });
});
