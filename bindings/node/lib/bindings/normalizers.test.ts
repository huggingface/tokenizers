import { prependNormalizer, stripAccentsNormalizer, stripNormalizer, unicodeFilter } from '../../'

describe('stripNormalizer', () => {
  it('instantiates with no parameters', () => {
    const normalizer = stripNormalizer()
    expect(normalizer.constructor.name).toEqual('Normalizer')
  })

  it('accepts `undefined` as first parameter', () => {
    expect(stripNormalizer(undefined)).toBeDefined()
  })

  it('accepts `undefined` as second parameter', () => {
    expect(stripNormalizer(false, undefined)).toBeDefined()
  })

  it('instantiates with one parameter', () => {
    const normalizer = stripNormalizer(false)
    expect(normalizer.constructor.name).toEqual('Normalizer')
  })

  it('instantiates with two parameters', () => {
    const normalizer = stripNormalizer(false, true)
    expect(normalizer.constructor.name).toEqual('Normalizer')
  })

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

describe('stripAccentsNormalizer', () => {
  it('initialize', () => {
    const normalizer = stripAccentsNormalizer()
    expect(normalizer.constructor.name).toEqual('Normalizer')
  })
})

describe('unicodeFilter', () => {
  it('instantiates with defaults', () => {
    const normalizer = unicodeFilter()
    expect(normalizer.constructor.name).toEqual('Normalizer')
  })

  it('handles default filtering', () => {
    const normalizer = unicodeFilter() // Default filters out Unassigned, PrivateUse
    const input = 'Hello' + String.fromCharCode(0xE000) + String.fromCodePoint(0xF0000) + String.fromCodePoint(0x10FFFF)
    expect(normalizer.normalizeString(input)).toEqual('Hello')
  })

  it('accepts custom filter options', () => {
    // Only filter private use areas
    const normalizer = unicodeFilter(false, true)
    const input = 'Hello' + String.fromCharCode(0xE000) + String.fromCodePoint(0xF0000) + String.fromCodePoint(0x10FFFF)
    const expected = 'Hello' + String.fromCodePoint(0x10FFFF)
    expect(normalizer.normalizeString(input)).toEqual(expected)
  })

  it('accepts undefined options', () => {
    const normalizer = unicodeFilter(undefined, undefined)
    const input = 'Hello' + String.fromCharCode(0xE000) + String.fromCodePoint(0xF0000) + String.fromCodePoint(0x10FFFF)
    expect(normalizer.normalizeString(input)).toEqual('Hello')
  })

  it('can disable all filtering', () => {
    const normalizer = unicodeFilter(false, false)
    const input = 'Hello' + String.fromCharCode(0xE000) + String.fromCodePoint(0xF0000) + String.fromCodePoint(0x10FFFF)
    expect(normalizer.normalizeString(input)).toEqual(input)
  })
})
