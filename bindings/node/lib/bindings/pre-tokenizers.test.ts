import { describe, it } from 'node:test'

import { metaspacePreTokenizer } from '../../index.js'
import { expect } from '../expect.ts'

describe('metaspacePreTokenizer', () => {
  it('can pre-tokenize strings', () => {
    const pretok = metaspacePreTokenizer()
    expect(pretok.preTokenizeString('Hello there friend')).toEqual([
      ['▁Hello', [0, 5]],
      ['▁there', [5, 11]],
      ['▁friend', [11, 18]],
    ])
  })
})
