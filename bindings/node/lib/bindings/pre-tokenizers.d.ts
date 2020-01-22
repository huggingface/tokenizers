/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of a
 * PreTokenizer will return an instance of this class when instantiated.
 */
declare class PreTokenizer {}

/**
 * Instantiate a new ByteLevel PreTokenizer
 *
 * @param {boolean} [addPrefixSpace=true] Whether to add a space to the first word if there isn't already one.
 * This lets us treat `hello` exactly like `say hello`.
 * @returns {PreTokenizer} ByteLevel PreTokenizer.
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
 * Returns a new Bert PreTokenizer.
 * This pre-tokenizer splits tokens on spaces, and also on punctuation.
 * Each occurence of a punctuation character will be treated separately.
 */
export function bertPreTokenizer(): PreTokenizer;

/**
 * Returns a new Metaspace Tokenizer.
 * This pre-tokenizer replaces any whitespace by the provided replacement character.
 * It then tries to split on these spaces.
 *
 * @param {string} [replacement="▁"] The replacement character. Must be exactly one character.
 * By default we use the `▁` (U+2581) meta symbol (Same as in SentencePiece).
 * @param {boolean} [addPrefixSpace] Whether to add a space to the first word if there isn't already one.
 * This lets us treat `hello` exactly like `say hello`.
 */
export function metaspacePreTokenizer(
  replacement?: string,
  addPrefixSpace?: boolean
): PreTokenizer;
