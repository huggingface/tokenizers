import { describe, it } from 'node:test'

import { WordPiece } from '../../index.js'
import { expect } from '../expect.ts'
/* eslint-disable @typescript-eslint/no-empty-function */
/* eslint-disable @typescript-eslint/no-explicit-any */

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
  })
})
