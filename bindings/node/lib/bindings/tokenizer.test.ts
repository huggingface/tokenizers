import { Tokenizer } from "./tokenizer";
import { BPE } from './models';
import { promisify } from 'util';

// jest.mock('../bindings/tokenizer');
// jest.mock('../bindings/models', () => ({
//   __esModule: true,
//   Model: jest.fn()
// }));

// Or:
// jest.mock('../bindings/models', () => {
//   return require('../bindings/__mocks__/models');
// });

// const TokenizerMock = mocked(Tokenizer);

let tokenizer: Tokenizer;
beforeEach(() => {
  // Clear all instances and calls to constructor and all methods:
  // TokenizerMock.mockClear();

  const model = BPE.empty();
  tokenizer = new Tokenizer(model);
  tokenizer.addTokens(['my', 'name', 'is', 'john', 'pair']);
});

describe("Tokenizer", () => {

  describe("encode", () => {
    it("is a function w/ parameters", async () => {
      expect(typeof(tokenizer.encode)).toBe('function');
      const encode = promisify(tokenizer.encode.bind(tokenizer));
      await encode('my name is john', null);
      await encode('my name is john', 'pair');
    });

    it("returns an Encoding", async () => {
      const encode = promisify(tokenizer.encode.bind(tokenizer));
      const encoding = await encode('my name is john', 'pair');
      
      expect(encoding.getAttentionMask()).toEqual([1, 1, 1, 1, 1]);

      const ids = encoding.getIds();
      expect(Array.isArray(ids)).toBe(true);
      expect(ids.length).toBe(5);
      for (const id of ids) {
        expect(typeof(id)).toBe('number');
      }
      
      expect(encoding.getOffsets()).toEqual([ [ 0, 2 ], [ 2, 6 ], [ 6, 8 ], [ 8, 12 ], [ 12, 16 ] ]);
      expect(encoding.getOverflowing()).toBeUndefined();
      expect(encoding.getSpecialTokensMask()).toEqual([0, 0, 0, 0, 0]);
      expect(encoding.getTokens()).toEqual([ 'my', 'name', 'is', 'john', 'pair' ]);
      expect(encoding.getTypeIds()).toEqual([ 0, 0, 0, 0, 1 ]);
    });
  });

});
