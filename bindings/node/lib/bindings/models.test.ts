/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { BPE, WordPiece } from "./models";

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
    it("returns `undefined`", () => {
      expect(
        (BPE as any).init({ a: 0, b: 1, ab: 2 }, [["a", "b"]], () => {})
      ).toBeUndefined();
    });
    it("has its callback called with the loaded model", () => {
      return new Promise((done) => {
        (BPE as any).init({ a: 0, b: 1, ab: 2 }, [["a", "b"]], (err: any, model: any) => {
          expect(model).toBeDefined();
          done();
        });
      });
    });
  });
});
