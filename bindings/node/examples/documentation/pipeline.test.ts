const globRequire = require;

describe("pipelineExample", () => {
    // This is a hack to let us require using path similar to what the user has to use
    function require(mod: string) {
        let path = mod.slice("tokenizers/".length);
        return globRequire("../../lib/" + path);
    }

    it("", async () => {
        // START reload_tokenizer
        const { Tokenizer } = require("tokenizers/bindings/tokenizer");

        const tokenizer = Tokenizer.fromFile("data/tokenizer-wiki.json");
        // END reload_tokenizer
        // START setup_normalizer
        const { sequenceNormalizer, nfdNormalizer, stripAccentsNormalizer } = require("tokenizers/bindings/normalizers");

        const normalizer = sequenceNormalizer([nfdNormalizer(), stripAccentsNormalizer()]);
        // END setup_normalizer
        // START test_normalizer
        let normalized = normalizer.normalizeStr("Héllò hôw are ü?")
        // "Hello how are u?"
        // END test_normalizer
        expect(normalized).toEqual("Hello how are u?");
        // START replace_normalizer
        tokenizer.setNormalizer(normalizer)
        // END replace_normalizer
    });
});
