import { test } from 'node:test'
import assert from 'node:assert'
import { PipelineTokenizer } from '../index.js'

// `make test` fetches this one; see the Makefile's TESTS_RESOURCES.
const MODEL = new URL('../data/tokenizer-wiki.json', import.meta.url).pathname

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
  const withSpecials = tok.encode('Hello', true)
  const without = tok.encode('Hello', false)
  assert.ok(withSpecials.length >= without.length)
})
