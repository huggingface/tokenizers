import { PaddingDirection, TruncationStrategy } from "../../bindings/enums";
import { BPE } from "../../bindings/models";
import {
  PaddingConfiguration,
  Tokenizer,
  TruncationConfiguration,
} from "../../bindings/tokenizer";
import { BaseTokenizer } from "./base.tokenizer";

describe("BaseTokenizer", () => {
  let tokenizer: BaseTokenizer<Record<string, unknown>>;

  beforeEach(() => {
    // Clear all instances and calls to constructor and all methods:
    // TokenizerMock.mockClear();

    const model = BPE.empty();
    const t = new Tokenizer(model);
    tokenizer = new BaseTokenizer(t, {});
  });

  describe("truncation", () => {
    it("returns `null` if no truncation setted", () => {
      expect(tokenizer.truncation).toBeNull();
    });

    it("returns configuration when `setTruncation` has been called", () => {
      tokenizer.setTruncation(2);
      const expectedConfig: TruncationConfiguration = {
        maxLength: 2,
        strategy: TruncationStrategy.LongestFirst,
        stride: 0,
      };
      expect(tokenizer.truncation).toEqual(expectedConfig);
    });

    it("returns null when `disableTruncation` has been called", () => {
      tokenizer.setTruncation(2);
      tokenizer.disableTruncation();
      expect(tokenizer.truncation).toBeNull();
    });
  });

  describe("padding", () => {
    it("returns `null` if no padding setted", () => {
      expect(tokenizer.padding).toBeNull();
    });

    it("returns configuration when `setPadding` has been called", () => {
      tokenizer.setPadding();
      const expectedConfig: PaddingConfiguration = {
        direction: PaddingDirection.Right,
        padId: 0,
        padToken: "[PAD]",
        padTypeId: 0,
      };
      expect(tokenizer.padding).toEqual(expectedConfig);
    });

    it("returns null when `disablePadding` has been called", () => {
      tokenizer.setPadding();
      tokenizer.disablePadding();
      expect(tokenizer.padding).toBeNull();
    });
  });
});
