import { describe, it } from 'node:test'

import { bertProcessing, templateProcessing } from '../../index.js'
import { expect } from '../expect.ts'
/* eslint-disable @typescript-eslint/no-explicit-any */

describe('bertProcessing', () => {
  it('throws if only one argument is provided', () => {
    expect(() => (bertProcessing as any)(['sep', 1])).toThrow('Failed to get Array length')
  })

  it('throws if arguments are malformed', () => {
    expect(() => (bertProcessing as any)(['sep', '1'], ['cls', '2'])).toThrow(
      'Failed to convert napi value String into rust type `u32`',
    )
    expect(() => (bertProcessing as any)(['sep'], ['cls'])).toThrow('Array length < 2')
  })
})

describe('templateProcessing', () => {
  it('throws if special tokens are missing', () => {
    expect(() => templateProcessing('[CLS] $A [SEP]')).toThrow('Missing SpecialToken(s) with id(s)')
  })
})
