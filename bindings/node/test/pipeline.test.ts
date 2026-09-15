import { test } from 'node:test'
import assert from 'node:assert'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PipelineTokenizer } from '../index.js'

// `make test` fetches this one; see the Makefile's TESTS_RESOURCES.
const MODEL = new URL('../data/tokenizer-wiki.json', import.meta.url).pathname

// The fixture configures no padding, so tests that need some write a copy with a `padding` block.
function paddedModel(): string {
  const config = JSON.parse(readFileSync(MODEL, 'utf8'))
  config.padding = {
    strategy: { Fixed: 16 },
    direction: 'Right',
    pad_to_multiple_of: null,
    pad_id: 0,
    pad_type_id: 0,
    pad_token: '[PAD]',
  }
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
  const tok = PipelineTokenizer.fromFile(paddedModel())
  assert.strictEqual(tok.encode('Hello').length, 16)
  assert.ok(tok.encode('Hello', { padding: false }).length < 16)
})

test('a padding override keeps the configured fields it leaves out', () => {
  const tok = PipelineTokenizer.fromFile(paddedModel())
  const plain = tok.encode('Hello', { padding: false })
  const left = tok.encode('Hello', { padding: { direction: 'left', padId: 7 } })
  assert.strictEqual(left.length, 16)
  assert.ok(left.subarray(0, 16 - plain.length).every((id) => id === 7))
  assert.deepStrictEqual(left.subarray(16 - plain.length), plain)
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
