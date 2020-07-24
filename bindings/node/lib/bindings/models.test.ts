/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { BPE, WordPiece } from "./models";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("WordPiece", () => {
  describe("fromFiles", () => {
    it("throws if called with only one argument", () => {
      expect(() => (WordPiece as any).fromFiles("test")).toThrow("not enough arguments");
    });

    it("throws if called with 2 arguments without a callback as third argument", () => {
      expect(() => (WordPiece as any).fromFiles("test", {})).toThrow(
        "not enough arguments"
      );
    });

    describe("when called with 2 correct arguments", () => {
      it("returns `undefined` ", () => {
        expect(WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, () => {})).toBeUndefined();
      });

      it("has its callback called with the loaded model", () => {
        return new Promise((done) => {
          WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, (err, model) => {
            expect(model).toBeDefined();
            done();
          });
        });
      });
    });

    describe("when called with 3 correct arguments", () => {
      it("returns `undefined`", () => {
        expect(
          WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, {}, () => {})
        ).toBeUndefined();
      });

      it("has its callback called with the loaded model", () => {
        return new Promise((done) => {
          WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, {}, (err, model) => {
            expect(model).toBeDefined();
            done();
          });
        });
      });
    });
  });
});

describe("BPE", () => {
  describe("fromFiles", () => {
    it("throws if called with only two arguments", () => {
      expect(() => (BPE as any).fromFiles("test", "bis")).toThrow("not enough arguments");
    });

    it("throws if called with 3 arguments without a callback as last argument", () => {
      expect(() => (BPE as any).fromFiles("test", "bis", {})).toThrow(
        "not enough arguments"
      );
    });
  });

  describe("when called with 3 correct arguments", () => {
    it("returns `undefined`", () => {
      expect(
        BPE.fromFiles(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`, () => {})
      ).toBeUndefined();
    });

    it("has its callback called with the loaded model", () => {
      return new Promise((done) => {
        BPE.fromFiles(
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
        BPE.fromFiles(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`, {}, () => {})
      ).toBeUndefined();
    });

    it("has its callback called with the loaded model", () => {
      return new Promise((done) => {
        BPE.fromFiles(
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
});
