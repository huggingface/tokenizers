import { BertWordPieceOptions, BertWordPieceTokenizer } from "./bert-wordpiece.tokenizer";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("BertWordPieceTokenizer", () => {
  describe("fromOptions", () => {
    it("does not throw any error if no vocabFile is provided", async () => {
      const tokenizer = await BertWordPieceTokenizer.fromOptions();
      expect(tokenizer).toBeDefined();
    });

    describe("when a vocabFile is provided and `addSpecialTokens === true`", () => {
      it("throws a `sepToken error` if no `sepToken` is provided", async () => {
        const options: BertWordPieceOptions = {
          vocabFile: MOCKS_DIR + "/bert-vocab-empty.txt",
        };

        await expect(BertWordPieceTokenizer.fromOptions(options)).rejects.toThrow(
          "sepToken not found in the vocabulary"
        );
      });

      it("throws a `clsToken error` if no `clsToken` is provided", async () => {
        const options: BertWordPieceOptions = {
          vocabFile: MOCKS_DIR + "/bert-vocab-without-cls.txt",
        };

        await expect(BertWordPieceTokenizer.fromOptions(options)).rejects.toThrow(
          "clsToken not found in the vocabulary"
        );
      });
    });
  });
});
