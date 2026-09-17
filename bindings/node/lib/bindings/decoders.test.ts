import { describe, it } from 'node:test'

import {
  byteFallbackDecoder,
  ctcDecoder,
  fuseDecoder,
  metaspaceDecoder,
  replaceDecoder,
  sequenceDecoder,
  stripDecoder,
  wordPieceDecoder,
} from '../../index.js'
import { expect } from '../expect.ts'

describe('wordPieceDecoder', () => {
  it('can decode arrays of strings', () => {
    expect(wordPieceDecoder().decode(['Hel', '##lo', 'there', 'my', 'fr', '##iend'])).toEqual('Hello there my friend')
  })
})

describe('byteFallbackDecoder', () => {
  it('can decode arrays of strings', () => {
    expect(byteFallbackDecoder().decode(['Hel', 'lo'])).toEqual('Hello')
    expect(byteFallbackDecoder().decode(['<0x61>'])).toEqual('a')
    expect(byteFallbackDecoder().decode(['<0x61>'])).toEqual('a')
    expect(byteFallbackDecoder().decode(['My', ' na', 'me'])).toEqual('My name')
    expect(byteFallbackDecoder().decode(['<0x61>'])).toEqual('a')
    expect(byteFallbackDecoder().decode(['<0xE5>'])).toEqual('�')
    expect(byteFallbackDecoder().decode(['<0xE5>', '<0x8f>'])).toEqual('��')
    expect(byteFallbackDecoder().decode(['<0xE5>', '<0x8f>', '<0xab>'])).toEqual('叫')
    expect(byteFallbackDecoder().decode(['<0xE5>', '<0x8f>', 'a'])).toEqual('��a')
    expect(byteFallbackDecoder().decode(['<0xE5>', '<0x8f>', '<0xab>', 'a'])).toEqual('叫a')
  })
})

describe('replaceDecoder', () => {
  it('can decode arrays of strings', () => {
    expect(replaceDecoder('_', ' ').decode(['Hello', '_Hello'])).toEqual('Hello Hello')
  })
})

describe('fuseDecoder', () => {
  it('can decode arrays of strings', () => {
    expect(fuseDecoder().decode(['Hel', 'lo'])).toEqual('Hello')
  })
})

describe('stripDecoder', () => {
  it('can decode arrays of strings', () => {
    expect(stripDecoder('_', 1, 0).decode(['_Hel', 'lo', '__there'])).toEqual('Hello_there')
  })
})

describe('metaspaceDecoder', () => {
  it('works', () => {
    expect(metaspaceDecoder().decode(['▁Hello'])).toEqual('Hello')
  })
})

describe('ctcDecoder', () => {
  it('encodes correctly', () => {
    expect(ctcDecoder().decode(['<pad>', 'h', 'h', 'e', 'e', 'l', 'l', '<pad>', 'l', 'l', 'o'])).toEqual('hello')
  })
})

describe('sequenceDecoder', () => {
  it('encodes correctly', () => {
    expect(
      sequenceDecoder([ctcDecoder(), metaspaceDecoder()]).decode(['▁', '▁', 'H', 'H', 'i', 'i', '▁', 'y', 'o', 'u']),
    ).toEqual('Hi you')
  })
})
