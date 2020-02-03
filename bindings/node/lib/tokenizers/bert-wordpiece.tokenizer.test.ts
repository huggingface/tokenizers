import { mocked } from "ts-jest/utils";

import { Tokenizer } from "../bindings/tokenizer";
import { BertWordPieceOptions, BertWordPieceTokenizer } from "./bert-wordpiece.tokenizer";

jest.mock("../bindings/models");
jest.mock("../bindings/tokenizer");

describe("BertWordPieceTokenizer", () => {
  describe("fromOptions", () => {
    it("does not throw any error if no vocabFile is provided", async () => {
      const tokenizer = await BertWordPieceTokenizer.fromOptions();
      expect(tokenizer).toBeDefined();
    });

    describe("when a vocabFile is provided and `addSpecialTokens === true`", () => {
      it("throws a `sepToken error` if no `sepToken` is provided", () => {
        const options: BertWordPieceOptions = {
          vocabFile: "./fake.txt",
          sepToken: undefined
        };

        expect.assertions(1);
        return BertWordPieceTokenizer.fromOptions(options).catch(e =>
          expect(e).toBeDefined()
        );
      });

      it("throws a `clsToken error` if no `clsToken` is provided", () => {
        const options: BertWordPieceOptions = {
          vocabFile: "./fake.txt",
          clsToken: undefined
        };

        mocked(Tokenizer.prototype.tokenToId).mockImplementationOnce(() => 10);

        expect.assertions(1);
        return BertWordPieceTokenizer.fromOptions(options).catch(e =>
          expect(e).toBeDefined()
        );
      });
    });
  });
});
