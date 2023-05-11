/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { BPE, Unigram, WordPiece } from '../../'

const MOCKS_DIR = __dirname + '/__mocks__'

describe('WordPiece', () => {
  describe('fromFile', () => {
    it('throws if called with only one argument', () => {
      expect(() => (WordPiece as any).fromFile()).toThrow(
        'Failed to convert JavaScript value `Undefined` into rust type `String`',
      )
    })

    it('throws if called with 2 arguments without a callback as third argument', () => {
      expect(() => (WordPiece as any).fromFile({})).toThrow(
        'Failed to convert JavaScript value `Object {}` into rust type `String`',
      )
    })

    it('has its callback called with the loaded model', async () => {
      const model = await WordPiece.fromFile(`${MOCKS_DIR}/vocab.txt`)
      expect(model).toBeDefined()
    })
  })
})

describe('BPE', () => {
  describe('fromFile', () => {
    it('has its callback called with the loaded model', async () => {
      const model = await BPE.fromFile(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`)
      expect(model).toBeDefined()
    })

    it('has its callback called with the loaded model', async () => {
      const model = await BPE.fromFile(`${MOCKS_DIR}/vocab.json`, `${MOCKS_DIR}/merges.txt`, {})
      expect(model).toBeDefined()
    })
  })
  describe('When initialized from memory', () => {
    it('returns the loaded Model', () => {
      const bpe = BPE.init({ a: 0, b: 1, ab: 2 }, [['a', 'b']])
      // expect(bpe.constructor.name).toEqual("Model");
      expect(bpe.constructor.name).toEqual('BPE')
    })
  })
})

describe('Unigram', () => {
  it('can be initialized from memory', () => {
    const unigram = Unigram.init(
      [
        ['<unk>', 0],
        ['Hello', -1],
        ['there', -2],
      ],
      {
        unkId: 0,
      },
    )
    expect(unigram.constructor.name).toEqual('Unigram')
  })
})
