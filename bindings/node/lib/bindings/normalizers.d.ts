/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of a
 * Normalizer will return an instance of this class when instantiated.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface Normalizer {
  normalizeString(s: string): string;
}

export interface BertNormalizerOptions {
  /**
   * Whether to clean the text, by removing any control characters
   * and replacing all whitespaces by the classic one.
   * @default true
   */
  cleanText?: boolean;
  /**
   * Whether to handle chinese chars by putting spaces around them.
   * @default true
   */
  handleChineseChars?: boolean;
  /**
   * Whether to lowercase.
   * @default true
   */
  lowercase?: boolean;
  /**
   * Whether to strip all accents.
   * @default undefined
   */
  stripAccents?: boolean;
}

/**
 * Instantiate a Bert Normalizer with the given options
 *
 * @param [options] Normalizer options
 * @returns Bert Normalizer. Takes care of normalizing raw text before giving it to a Bert model.
 * This includes cleaning the text, handling accents, chinese chars and lowercasing
 */
export function bertNormalizer(options?: BertNormalizerOptions): Normalizer;

/**
 * Returns a new NFC Unicode Normalizer
 */
export function nfcNormalizer(): Normalizer;

/**
 * Returns a new NFD Unicode Normalizer
 */
export function nfdNormalizer(): Normalizer;

/**
 * Returns a new NFKC Unicode Normalizer
 */
export function nfkcNormalizer(): Normalizer;

/**
 * Returns a new NFKD Unicode Normalizer
 */
export function nfkdNormalizer(): Normalizer;

/**
 * Instantiate a new Normalization Sequence using the given normalizers
 * @param normalizers A list of Normalizer to be run as a sequence
 */
export function sequenceNormalizer(normalizers: Normalizer[]): Normalizer;

/**
 * Returns a new Lowercase Normalizer
 */
export function lowercaseNormalizer(): Normalizer;

/**
 *  Returns a new Strip Normalizer
 * @param [left=true] Whether or not to strip on the left (defaults to `true`)
 * @param [right=true] Whether or not to strip on the right (defaults to `true`)
 */
export function stripNormalizer(left?: boolean, right?: boolean): Normalizer;

/**
 *  Returns a new StripAccents Normalizer
 */
export function stripAccentsNormalizer(): Normalizer;

/**
 * Returns a new Nmt Normalizer
 */
export function nmtNormalizer(): Normalizer;

/**
 * Returns a new Precompiled Normalizer
 */
export function precompiledNormalizer(): Normalizer;

/**
 * Returns a new Replace Normalizer
 */
export function replaceNormalizer(): Normalizer;
