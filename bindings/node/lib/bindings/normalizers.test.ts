import { describe, it } from 'node:test'

import { prependNormalizer, stripNormalizer } from '../../index.js'
import { expect } from '../expect.ts'

describe('stripNormalizer', () => {
  it('prepend instantiates with one parameter', () => {
    const normalizer = prependNormalizer('_')
    expect(normalizer.constructor.name).toEqual('Normalizer')
    expect(normalizer.normalizeString('Hello')).toEqual('_Hello')
  })

  it('can normalize strings', () => {
    const normalizer = stripNormalizer()
    expect(normalizer.normalizeString('     Hello there   ')).toEqual('Hello there')
  })
})
