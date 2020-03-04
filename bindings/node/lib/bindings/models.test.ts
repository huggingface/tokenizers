/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { WordPiece } from "./models";

const MOCKS_DIR = __dirname + "/__mocks__";

describe("WordPiece", () => {
  describe("fromFiles", () => {
    it("throws if called with only one argument", () => {
      expect(() => (WordPiece as any).fromFiles("test")).toThrow("not enough arguments");
    });

    it("throws if called with 2 arguments without a callback as second argument", () => {
      expect(() => (WordPiece as any).fromFiles("test", {})).toThrow(
        "failed downcast to function"
      );
    });

    describe("when called with 2 correct arguments", () => {
      it("returns `undefined` ", () => {
        expect(WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, () => {})).toBeUndefined();
      });

      it("has its callback called with the loaded model", () => {
        return new Promise(done => {
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
        return new Promise(done => {
          WordPiece.fromFiles(`${MOCKS_DIR}/vocab.txt`, {}, (err, model) => {
            expect(model).toBeDefined();
            done();
          });
        });
      });
    });
  });
});
