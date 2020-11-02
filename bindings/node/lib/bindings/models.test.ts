/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { BPE, Unigram, WordPiece } from "./models";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("WordPiece", () => {
  describe("fromFile", () => {
    it("throws if called with only one argument", () => {
      expect(() => (WordPiece as any).fromFile("test")).toThrow("not enough arguments");
    });

    it("throws if called with 2 arguments without a callback as third argument", () => {
      expect(() => (WordPiece as any).fromFile("test", {})).toThrow(
        "not enough arguments"
      );
    });

    describe("when called with 2 correct arguments", () => {
      it("returns `undefined` ", () => {
        expect(WordPiece.fromFile(`${MOCKS_DIR}/vocab.txt`, () => {})).toBeUndefined();
      });

      it("has its callback called with the loaded model", () => {
        return new Promise((done) => {
          WordPiece.fromFile(`${MOCKS_DIR}/vocab.txt`, (err, model) => {
            expect(model).toBeDefined();
            done();
          });
        });
      });
    });

    describe("when called with 3 correct arguments", () => {
      it("returns `undefined`", () => {
        expect(
          WordPiece.fromFile(`${MOCKS_DIR}/vocab.txt`, {}, () => {})
        ).toBeUndefined();
      });

      it("has its callback called with the loaded model", () => {
        return new Promise((done) => {
          WordPiece.fromFile(`${MOCKS_DIR}/vocab.txt`, {}, (err, model) => {
            expect(model).toBeDefined();
            done();
          });
        });
      });
    });
  });
});

describe("BPE", () => {
  describe("fromFile", () => {
    it("throws if called with only two arguments", () => {
      expect(() => (BPE as any).fromFile("test", "bis")).toThrow("not enough arguments");
    });

    it("throws if called with 3 arguments without a callback as last argument", () => {
      expect(() => (BPE as any).fromFile("test", "bis", {})).toThrow(
        "not enough arguments"
      );
    });
  });

  describe("when called with 3 correct arguments", () => {
    it("returns `undefined`", () => {
      expect(
        BPE.fromFile(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`, () => {})
      ).toBeUndefined();
    });

    it("has its callback called with the loaded model", () => {
      return new Promise((done) => {
        BPE.fromFile(
          `${MOCKS_DIR}/vocab.json`,
          `${MOCKS_DIR}/merges.txt`,
          (err, model) => {
            expect(model).toBeDefined();
            done();
          }
        );
      });
    });
  });

  describe("when called with 4 correct arguments", () => {
    it("returns `undefined`", () => {
      expect(
        BPE.fromFile(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`, {}, () => {})
      ).toBeUndefined();
    });

    it("has its callback called with the loaded model", () => {
      return new Promise((done) => {
        BPE.fromFile(
          `${MOCKS_DIR}/vocab.json`,
          `${MOCKS_DIR}/merges.txt`,
          {},
          (err, model) => {
            expect(model).toBeDefined();
            done();
          }
        );
      });
    });
  });
  describe("When initialized from memory", () => {
    it("returns the loaded Model", () => {
      const bpe = BPE.init({ a: 0, b: 1, ab: 2 }, [["a", "b"]]);
      expect(bpe.constructor.name).toEqual("Model");
    });
  });
});

describe("Unigram", () => {
  it("can be initialized from memory", () => {
    const unigram = Unigram.init(
      [
        ["<unk>", 0],
        ["Hello", -1],
        ["there", -2],
      ],
      {
        unkId: 0,
      }
    );
    expect(unigram.constructor.name).toEqual("Model");
  });
});
