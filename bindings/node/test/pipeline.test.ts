import { test } from 'node:test'
import assert from 'node:assert'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PipelineTokenizer, type EncodeOptions } from '../index.js'

// `make test` fetches this one; see the Makefile's TESTS_RESOURCES.
const MODEL = new URL('../data/tokenizer-wiki.json', import.meta.url).pathname

// Every id in this file comes from released tokenizers 0.23.1 on the same fixture.
const LONG = 'Hello there, how are you today?'
const LONG_IDS = [27253, 5503, 16, 6447, 5112, 6218, 8773, 35]

// The fixture configures neither padding nor truncation, so tests that need one write a copy with
// that block added.
const PADDING = {
  strategy: { Fixed: 16 },
  direction: 'Right',
  pad_to_multiple_of: null,
  pad_id: 7,
  pad_type_id: 0,
  pad_token: '[PAD]',
}
const TRUNCATION = { direction: 'Left', max_length: 4, strategy: 'LongestFirst', stride: 0 }

function modelWith(blocks: Record<string, unknown>): string {
  const config = { ...JSON.parse(readFileSync(MODEL, 'utf8')), ...blocks }
  const path = join(mkdtempSync(join(tmpdir(), 'tokenizers-node-')), 'tokenizer.json')
  writeFileSync(path, JSON.stringify(config))
  return path
}

test('fromFile reads a legacy tokenizer.json', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  assert.ok(tok instanceof PipelineTokenizer)
})

test('encode returns a Uint32Array of ids', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const ids = tok.encode('Hello there, how are you?')
  assert.ok(ids instanceof Uint32Array, 'ids must marshal as a typed array, not a JS Array')
  assert.ok(ids.length > 0)
})

// The two entry points run the same encode; the only difference is who owns the buffer. If they
// ever disagree, one of them is marshalling wrong.
test('encodeBytesInto writes the same ids as encode', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const text = 'Hello there, how are you?'
  const want = tok.encode(text)
  const out = new Uint32Array(want.length)
  const n = tok.encodeBytesInto(Buffer.from(text, 'utf8'), out)
  assert.strictEqual(n, want.length)
  assert.deepStrictEqual(out.subarray(0, n), want)
})

test('encodeBytesInto rejects a buffer that is too small', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const text = 'Hello there, how are you?'
  const tooSmall = new Uint32Array(1)
  assert.throws(() => tok.encodeBytesInto(Buffer.from(text, 'utf8'), tooSmall), /buffer holds 1/)
})

test('addSpecialTokens is honoured', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const withSpecials = tok.encode('Hello', { addSpecialTokens: true })
  const without = tok.encode('Hello', { addSpecialTokens: false })
  assert.ok(withSpecials.length >= without.length)
})

test('padding.length pads to a fixed length', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const plain = tok.encode('Hello')
  const padded = tok.encode('Hello', { padding: { length: 16 } })
  assert.ok(plain.length < 16)
  assert.strictEqual(padded.length, 16)
  assert.deepStrictEqual(padded.subarray(0, plain.length), plain)
  assert.ok(padded.subarray(plain.length).every((id) => id === 0))
})

test('padding: false turns the configured padding off', () => {
  const tok = PipelineTokenizer.fromFile(modelWith({ padding: PADDING }))
  assert.strictEqual(tok.encode('Hello').length, 16)
  assert.ok(tok.encode('Hello', { padding: false }).length < 16)
})

test('a padding override replaces the configured padding', () => {
  const tok = PipelineTokenizer.fromFile(modelWith({ padding: PADDING }))
  assert.deepStrictEqual(Array.from(tok.encode('Hello')), [LONG_IDS[0], ...Array(15).fill(7)])
  assert.deepStrictEqual(Array.from(tok.encode('Hello', { padding: { length: 4 } })), [LONG_IDS[0], 0, 0, 0])
})

test('encodeBytesInto takes the same options', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const out = new Uint32Array(16)
  const n = tok.encodeBytesInto(Buffer.from('Hello', 'utf8'), out, { padding: { length: 16 } })
  assert.strictEqual(n, 16)
})

test('a padding direction other than left or right is refused', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  // @ts-expect-error the union is the whole point of the check
  assert.throws(() => tok.encode('Hello', { padding: { direction: 'up' } }), /'left' or 'right'/)
})

test('truncation.maxLength keeps the first ids', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  assert.deepStrictEqual(Array.from(tok.encode(LONG)), LONG_IDS)
  assert.deepStrictEqual(Array.from(tok.encode(LONG, { truncation: { maxLength: 4 } })), LONG_IDS.slice(0, 4))
})

test('a text shorter than maxLength is left alone', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  assert.deepStrictEqual(tok.encode('Hello there', { truncation: { maxLength: 4 } }), tok.encode('Hello there'))
})

test('truncation.direction left keeps the last ids', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const left = tok.encode(LONG, { truncation: { maxLength: 4, direction: 'left' } })
  assert.deepStrictEqual(Array.from(left), LONG_IDS.slice(-4))
})

test('fromFile reads a truncation block', () => {
  const tok = PipelineTokenizer.fromFile(modelWith({ truncation: TRUNCATION }))
  assert.deepStrictEqual(Array.from(tok.encode(LONG)), LONG_IDS.slice(-4))
})

test('truncation: false turns the configured truncation off', () => {
  const tok = PipelineTokenizer.fromFile(modelWith({ truncation: TRUNCATION }))
  assert.deepStrictEqual(Array.from(tok.encode(LONG, { truncation: false })), LONG_IDS)
})

test('a truncation override replaces the configured truncation', () => {
  const tok = PipelineTokenizer.fromFile(modelWith({ truncation: TRUNCATION }))
  assert.deepStrictEqual(Array.from(tok.encode(LONG, { truncation: { maxLength: 2 } })), LONG_IDS.slice(0, 2))
})

test('truncation.maxLength is required', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  // @ts-expect-error the required field is the whole point of the check
  const noLength: EncodeOptions = { truncation: {} }
  assert.throws(() => tok.encode(LONG, noLength), /none of these types .*TruncationOptions/)
})

// Padding to more than maxLength tells the two orders apart: padding first would leave the long
// text at 4 ids, cutting first pads it back out to 6.
test('truncation cuts before padding fills', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const options = { truncation: { maxLength: 4 }, padding: { length: 6 } }
  assert.deepStrictEqual(Array.from(tok.encode('Hello', options)), [LONG_IDS[0], 0, 0, 0, 0, 0])
  assert.deepStrictEqual(Array.from(tok.encode(LONG, options)), [...LONG_IDS.slice(0, 4), 0, 0])
})

// `encode` takes one sequence, so `only_second` never has the sequence it cuts.
test('only_second has nothing to cut', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  const onlySecond = { truncation: { maxLength: 4, strategy: 'only_second' as const } }
  assert.throws(() => tok.encode(LONG, onlySecond), /Second sequence not provided/)
})

test('a truncation strategy other than the three is refused', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  // @ts-expect-error the union is the whole point of the check
  const sideways: EncodeOptions = { truncation: { maxLength: 4, strategy: 'sideways' } }
  assert.throws(() => tok.encode(LONG, sideways), /'longest_first', 'only_first' or 'only_second'/)
})

test('a truncation direction other than left or right is refused', () => {
  const tok = PipelineTokenizer.fromFile(MODEL)
  // @ts-expect-error the union is the whole point of the check
  const up: EncodeOptions = { truncation: { maxLength: 4, direction: 'up' } }
  assert.throws(() => tok.encode(LONG, up), /truncation direction must be 'left' or 'right'/)
})
