/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of
 * a Decoder will return an instance of this class when instantiated.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface Decoder {
  decode(tokens: string[]): string;
}

/**
 * Instantiate a new ByteLevel Decoder
 */
export function byteLevelDecoder(): Decoder;

/**
 * Instantiate a new Replace Decoder
 * @param [pattern] The pattern to replace
 * @param [content] The replacement.
 */
export function replaceDecoder(pattern: string, content: string): Decoder;

/**
 * Instantiate a new WordPiece Decoder
 * @param [prefix='##'] The prefix to use for subwords that are not a beginning-of-word
 * @param [cleanup=true] Whether to cleanup some tokenization artifacts.
 * Mainly spaces before punctuation, and some abbreviated english forms.
 */
export function wordPieceDecoder(prefix?: string, cleanup?: boolean): Decoder;

/**
 * Instantiate a new ByteFallback Decoder
 * ByteFallback is a simple trick which converts tokens looking like `<0x61>`
 * to pure bytes, and attempts to make them into a string. If the tokens
 * cannot be decoded you will get � instead for each inconvertable byte token
 */
export function byteFallbackDecoder(): Decoder;

/**
 * Instantiate a new Fuse Decoder which fuses all tokens into one string
 */
export function fuseDecoder(): Decoder;

/**
 * Instantiate a new Strip Decoder
 * @param [content] The character to strip
 * @param [left] The number of chars to remove from the left of each token
 * @param [right] The number of chars to remove from the right of each token
 */
export function stripDecoder(content: string, left: number, right: number): Decoder;

/**
 * Instantiate a new Metaspace
 *
 * @param [replacement='▁'] The replacement character.
 * Must be exactly one character. By default we use the `▁` (U+2581) meta symbol (same as in SentencePiece).
 * @param [addPrefixSpace=true] Whether to add a space to the first word if there isn't already one.
 * This lets us treat `hello` exactly like `say hello`.
 */
export function metaspaceDecoder(replacement?: string, addPrefixSpace?: boolean): Decoder;

/**
 * Instantiate a new BPE Decoder
 * @param [suffix='</w>'] The suffix that was used to characterize an end-of-word.
 * This suffix will be replaced by whitespaces during the decoding
 */
export function bpeDecoder(suffix?: string): Decoder;

/**
 * Instantiate a new CTC Decoder
 * @param [pad_token='pad'] The pad token used by CTC to delimit a new token.
 * @param [word_delimiter_token='|'] The word delimiter token. It will be replaced by a space
 * @param [cleanup=true] Whether to cleanup some tokenization artifacts.
 * Mainly spaces before punctuation, and some abbreviated english forms.
 */
export function ctcDecoder(
  pad_token?: string,
  word_delimiter_token?: string,
  cleanup?: boolean
): Decoder;

/**
 * Instantiate a new Sequence Decoder
 * @param [decoders] The decoders to chain
 */
export function sequenceDecoder(decoders: Decoder[]): Decoder;
