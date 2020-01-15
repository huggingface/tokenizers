import { BertWordPieceOptions, BertWordPieceTokenizer } from "./bert-wordpiece.tokenizer";
import { mocked } from "ts-jest/utils";
import { Tokenizer } from "../bindings/tokenizer";

jest.mock("../bindings/models");
jest.mock("../bindings/tokenizer");

describe("BertWordPieceTokenizer", () => {
  describe("fromOptions", () => {

    it("should not throw any error if no vocabFile is provided", async () => {
      await BertWordPieceTokenizer.fromOptions();
    });

    describe("when a vocabFile is provided and `addSpecialTokens === true`", () => {
      it("should throw a `sepToken error` if no `sepToken` is provided", () => {
        const options: BertWordPieceOptions = {
          vocabFile: "./fake.txt",
          sepToken: undefined
        };
        
        expect.assertions(1);
        BertWordPieceTokenizer.fromOptions(options)
          .catch(e => expect(e).toBeDefined());
      });

      it("should throw a `clsToken error` if no `clsToken` is provided", () => {
        const options: BertWordPieceOptions = {
          vocabFile: "./fake.txt",
          clsToken: undefined
        };

        mocked(Tokenizer.prototype.tokenToId).mockImplementationOnce(() => 10);
        
        expect.assertions(1);
        BertWordPieceTokenizer.fromOptions(options)
          .catch(e => expect(e).toBeDefined());
      });
    });

  });
});
