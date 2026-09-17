/* eslint-disable @typescript-eslint/no-explicit-any */
// Minimal `expect` over node:assert, so the suite runs on the built-in
// `node --test` runner instead of pulling jest (~207 packages) for the eight
// matchers we actually use.
//
// `equals` deliberately mirrors jest's `toEqual` rather than
// assert.deepStrictEqual: it ignores `undefined`-valued properties and does not
// compare prototypes. The napi bindings hand back class instances (Encoding,
// AddedToken, ...) that tests compare against plain object literals, which
// deepStrictEqual would reject on the prototype check alone.
import assert from 'node:assert'

function equals(a: any, b: any): boolean {
  if (Object.is(a, b)) return true
  if (a instanceof Date && b instanceof Date) return a.getTime() === b.getTime()
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false
    return a.every((v, i) => equals(v, b[i]))
  }
  // jest's toEqual treats a missing key and an explicit `undefined` as equal
  const keys = (o: any) => Object.keys(o).filter((k) => o[k] !== undefined)
  const ka = keys(a)
  const kb = keys(b)
  if (ka.length !== kb.length) return false
  return ka.every((k) => kb.includes(k) && equals(a[k], b[k]))
}

const show = (v: any) => {
  try {
    return JSON.stringify(v) ?? String(v)
  } catch {
    return String(v)
  }
}

function matchError(err: unknown, expected?: string | RegExp): boolean {
  if (expected === undefined) return true
  const msg = err instanceof Error ? err.message : String(err)
  return typeof expected === 'string' ? msg.includes(expected) : expected.test(msg)
}

class Expectation {
  // NB: an explicit field, not a constructor parameter property -- node's
  // strip-only TypeScript mode rejects the latter.
  private readonly actual: any

  constructor(actual: any) {
    this.actual = actual
  }

  toEqual(expected: any): void {
    assert.ok(
      equals(this.actual, expected),
      `toEqual failed\n  actual:   ${show(this.actual)}\n  expected: ${show(expected)}`,
    )
  }

  toBe(expected: any): void {
    assert.ok(
      Object.is(this.actual, expected),
      `toBe failed\n  actual:   ${show(this.actual)}\n  expected: ${show(expected)}`,
    )
  }

  toBeDefined(): void {
    assert.ok(this.actual !== undefined, `expected value to be defined, got undefined`)
  }

  toBeUndefined(): void {
    assert.ok(this.actual === undefined, `expected undefined, got ${show(this.actual)}`)
  }

  toBeNull(): void {
    assert.ok(this.actual === null, `expected null, got ${show(this.actual)}`)
  }

  toHaveLength(n: number): void {
    assert.ok(this.actual?.length === n, `expected length ${n}, got ${show(this.actual?.length)}`)
  }

  toThrow(expected?: string | RegExp): void {
    let threw = false
    try {
      this.actual()
    } catch (err) {
      threw = true
      assert.ok(matchError(err, expected), `threw, but message did not match ${show(expected)}: ${err}`)
    }
    assert.ok(threw, 'expected function to throw, but it did not')
  }

  get rejects(): { toThrow: (expected?: string | RegExp) => Promise<void> } {
    const actual = this.actual
    return {
      toThrow: async (expected?: string | RegExp): Promise<void> => {
        try {
          await (typeof actual === 'function' ? actual() : actual)
        } catch (err) {
          assert.ok(matchError(err, expected), `rejected, but message did not match ${show(expected)}: ${err}`)
          return
        }
        assert.fail('expected promise to reject, but it resolved')
      },
    }
  }
}

export function expect(actual: any): Expectation {
  return new Expectation(actual)
}
