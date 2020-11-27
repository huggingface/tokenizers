/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of a
 * PreTokenizer will return an instance of this class when instantiated.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface PreTokenizer {
  preTokenizeString(s: string): [string, [number, number]][];
}

/**
 * Instantiate a new ByteLevel PreTokenizer
 *
 * @param [addPrefixSpace=true] Whether to add a space to the first word if there isn't already one.
 * This lets us treat `hello` exactly like `say hello`.
 * @returns ByteLevel PreTokenizer.
 * This pre-tokenizer takes care of replacing all bytes of the given string
 * with a corresponding representation, as well as splitting into words.
 */
export function byteLevelPreTokenizer(addPrefixSpace?: boolean): PreTokenizer;

/**
 * Returns the alphabet used by the ByteLevel PreTokenizer.
 * Since the ByteLevel works as its name suggests, at the byte level, it
 * encodes any byte to one visible character. This means that there is a
 * total of 256 different characters composing this alphabet.
 */
export function byteLevelAlphabet(): string[];

/**
 * Returns a Whitespace PreTokenizer
 * This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
 */
export function whitespacePreTokenizer(): PreTokenizer;

/**
 * Returns a WhitespaceSplit PreTokenizer
 * This pre-tokenizer simply splits on whitespaces only. Works almost like the `.split(' ')`
 * function, except that it accounts for multiple consecutive spaces
 */
export function whitespaceSplitPreTokenizer(): PreTokenizer;

/**
 * Returns a Split PreTokenizer
 * This versatile pre-tokenizer splits using the provided pattern and
 * according to the provided behavior. The pattern can be inverted by
 * making use of the invert flag.
 *
 * @param [pattern] A pattern used to split the string. Usually a string or a Regex.
 * @param [behavior] The behavior to use when splitting.
 * Choices: "removed", "isolated", "mergedWithPrevious", "mergedWithNext",
 * "contiguous".
 * @param [invert=false] Whether to invert the pattern.
 */
export function splitPreTokenizer(
  pattern?: string,
  behavior?: string,
  invert?: boolean
): PreTokenizer;

/**
 * Returns a new Bert PreTokenizer.
 * This pre-tokenizer splits tokens on spaces, and also on punctuation.
 * Each occurrence of a punctuation character will be treated separately.
 */
export function bertPreTokenizer(): PreTokenizer;

/**
 * Returns a new Metaspace PreTokenizer.
 * This pre-tokenizer replaces any whitespace by the provided replacement character.
 * It then tries to split on these spaces.
 *
 * @param [replacement="▁"] The replacement character. Must be exactly one character.
 * By default we use the `▁` (U+2581) meta symbol (Same as in SentencePiece).
 * @param [addPrefixSpace] Whether to add a space to the first word if there isn't already one.
 * This lets us treat `hello` exactly like `say hello`.
 */
export function metaspacePreTokenizer(
  replacement?: string,
  addPrefixSpace?: boolean
): PreTokenizer;

/**
 * Returns a CharDelimiterSplit PreTokenizer
 * This pre-tokenizer simply splits on the provided delimiter. Works almost like the `.split(delimiter)`
 * function, except that it accounts for multiple consecutive spaces
 *
 * @param delimiter The delimiter character on which the sequence will be split.
 */
export function charDelimiterSplitPreTokenizer(delimiter: string): PreTokenizer;

/**
 * Returns a new Punctuation PreTokenizer.
 * This pre-tokenizer splits tokens on punctuation.
 * Each occurrence of a punctuation character will be treated separately.
 */
export function punctuationPreTokenizer(): PreTokenizer;

/**
 * Returns a new Sequence PreTokenizer.
 * This pre-tokenizer combines other pretokenizers and applies them.
 * sequentially.
 */
export function sequencePreTokenizer(pretokenizers: PreTokenizer[]): PreTokenizer;

/**
 * Returns a new Digits PreTokenizer.
 * This pre-tokenizer splits on numbers. Optionnaly it can split on individual digits.
 *
 * @param [individualDigits=false] Whether to split on individual digits.
 */
export function digitsPreTokenizer(individualDigits?: boolean): PreTokenizer;
