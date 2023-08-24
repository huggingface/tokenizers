/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

export function bpeDecoder(suffix?: string | undefined | null): Decoder
export function byteFallbackDecoder(): Decoder
export function ctcDecoder(
  padToken?: string = '<pad>',
  wordDelimiterToken?: string | undefined | null,
  cleanup?: boolean | undefined | null,
): Decoder
export function fuseDecoder(): Decoder
export function metaspaceDecoder(replacement?: string = '▁', addPrefixSpace?: bool = true): Decoder
export function replaceDecoder(pattern: string, content: string): Decoder
export function sequenceDecoder(decoders: Array<Decoder>): Decoder
export function stripDecoder(content: string, left: number, right: number): Decoder
export function wordPieceDecoder(prefix?: string = '##', cleanup?: bool = true): Decoder
export const enum TruncationDirection {
  Left = 'Left',
  Right = 'Right',
}
export const enum TruncationStrategy {
  LongestFirst = 'LongestFirst',
  OnlyFirst = 'OnlyFirst',
  OnlySecond = 'OnlySecond',
}
export interface BpeOptions {
  cacheCapacity?: number
  dropout?: number
  unkToken?: string
  continuingSubwordPrefix?: string
  endOfWordSuffix?: string
  fuseUnk?: boolean
  byteFallback?: boolean
}
export interface WordPieceOptions {
  unkToken?: string
  continuingSubwordPrefix?: string
  maxInputCharsPerWord?: number
}
export interface WordLevelOptions {
  unkToken?: string
}
export interface UnigramOptions {
  unkId?: number
  byteFallback?: boolean
}
export function prependNormalizer(prepend: string): Normalizer
export function stripAccentsNormalizer(): Normalizer
export interface BertNormalizerOptions {
  cleanText?: boolean
  handleChineseChars?: boolean
  stripAccents?: boolean
  lowercase?: boolean
}
/**
 * bert_normalizer(options?: {
 *   cleanText?: bool = true,
 *   handleChineseChars?: bool = true,
 *   stripAccents?: bool = true,
 *   lowercase?: bool = true
 * })
 */
export function bertNormalizer(options?: BertNormalizerOptions | undefined | null): Normalizer
export function nfdNormalizer(): Normalizer
export function nfkdNormalizer(): Normalizer
export function nfcNormalizer(): Normalizer
export function nfkcNormalizer(): Normalizer
export function stripNormalizer(left?: boolean | undefined | null, right?: boolean | undefined | null): Normalizer
export function sequenceNormalizer(normalizers: Array<Normalizer>): Normalizer
export function lowercase(): Normalizer
export function replace(pattern: string, content: string): Normalizer
export function nmt(): Normalizer
export function precompiled(bytes: Array<number>): Normalizer
export const enum JsSplitDelimiterBehavior {
  Removed = 'Removed',
  Isolated = 'Isolated',
  MergedWithPrevious = 'MergedWithPrevious',
  MergedWithNext = 'MergedWithNext',
  Contiguous = 'Contiguous',
}
/** byte_level(addPrefixSpace: bool = true, useRegex: bool = true) */
export function byteLevelPreTokenizer(
  addPrefixSpace?: boolean | undefined | null,
  useRegex?: boolean | undefined | null,
): PreTokenizer
export function byteLevelAlphabet(): Array<string>
export function whitespacePreTokenizer(): PreTokenizer
export function whitespaceSplitPreTokenizer(): PreTokenizer
export function bertPreTokenizer(): PreTokenizer
export function metaspacePreTokenizer(replacement?: string = '▁', addPrefixSpace?: bool = true): PreTokenizer
export function splitPreTokenizer(pattern: string, behavior: string, invert?: boolean | undefined | null): PreTokenizer
export function punctuationPreTokenizer(behavior?: string | undefined | null): PreTokenizer
export function sequencePreTokenizer(preTokenizers: Array<PreTokenizer>): PreTokenizer
export function charDelimiterSplit(delimiter: string): PreTokenizer
export function digitsPreTokenizer(individualDigits?: boolean | undefined | null): PreTokenizer
export function bertProcessing(sep: [string, number], cls: [string, number]): Processor
export function robertaProcessing(
  sep: [string, number],
  cls: [string, number],
  trimOffsets?: boolean | undefined | null,
  addPrefixSpace?: boolean | undefined | null,
): Processor
export function byteLevelProcessing(trimOffsets?: boolean | undefined | null): Processor
export function templateProcessing(
  single: string,
  pair?: string | undefined | null,
  specialTokens?: Array<[string, number]> | undefined | null,
): Processor
export function sequenceProcessing(processors: Array<Processor>): Processor
export const enum PaddingDirection {
  Left = 0,
  Right = 1,
}
export interface PaddingOptions {
  maxLength?: number
  direction?: string | PaddingDirection
  padToMultipleOf?: number
  padId?: number
  padTypeId?: number
  padToken?: string
}
export interface EncodeOptions {
  isPretokenized?: boolean
  addSpecialTokens?: boolean
}
export interface TruncationOptions {
  maxLength?: number
  strategy?: TruncationStrategy
  direction?: string | TruncationDirection
  stride?: number
}
export interface AddedTokenOptions {
  singleWord?: boolean
  leftStrip?: boolean
  rightStrip?: boolean
  normalized?: boolean
}
export interface JsFromPretrainedParameters {
  revision?: string
  authToken?: string
}
export function slice(s: string, beginIndex?: number | undefined | null, endIndex?: number | undefined | null): string
export function mergeEncodings(encodings: Array<Encoding>, growingOffsets?: boolean | undefined | null): Encoding
/** Decoder */
export class Decoder {
  decode(tokens: Array<string>): string
}
export type JsEncoding = Encoding
export class Encoding {
  constructor()
  getLength(): number
  getNSequences(): number
  getIds(): Array<number>
  getTypeIds(): Array<number>
  getAttentionMask(): Array<number>
  getSpecialTokensMask(): Array<number>
  getTokens(): Array<string>
  getOffsets(): Array<Array<number>>
  getWordIds(): Array<number | undefined | null>
  charToToken(pos: number, seqId?: number | undefined | null): number | null
  charToWord(pos: number, seqId?: number | undefined | null): number | null
  pad(length: number, options?: PaddingOptions | undefined | null): void
  truncate(
    length: number,
    stride?: number | undefined | null,
    direction?: string | TruncationDirection | undefined | null,
  ): void
  wordToTokens(word: number, seqId?: number | undefined | null): [number, number] | null | undefined
  wordToChars(word: number, seqId?: number | undefined | null): [number, number] | null | undefined
  tokenToChars(token: number): [number, [number, number]] | null | undefined
  tokenToWord(token: number): number | null
  getOverflowing(): Array<Encoding>
  getSequenceIds(): Array<number | undefined | null>
  tokenToSequence(token: number): number | null
}
export class Model {}
export type Bpe = BPE
export class BPE {
  static empty(): Model
  static init(vocab: Vocab, merges: Merges, options?: BpeOptions | undefined | null): Model
  static fromFile(vocab: string, merges: string, options?: BpeOptions | undefined | null): Promise<Model>
}
export class WordPiece {
  static init(vocab: Vocab, options?: WordPieceOptions | undefined | null): Model
  static empty(): WordPiece
  static fromFile(vocab: string, options?: WordPieceOptions | undefined | null): Promise<Model>
}
export class WordLevel {
  static init(vocab: Vocab, options?: WordLevelOptions | undefined | null): Model
  static empty(): WordLevel
  static fromFile(vocab: string, options?: WordLevelOptions | undefined | null): Promise<Model>
}
export class Unigram {
  static init(vocab: Array<[string, number]>, options?: UnigramOptions | undefined | null): Model
  static empty(): Model
}
/** Normalizer */
export class Normalizer {
  normalizeString(sequence: string): string
}
/** PreTokenizers */
export class PreTokenizer {
  preTokenizeString(sequence: string): [string, [number, number]][]
}
export class Processor {}
export class AddedToken {
  constructor(token: string, isSpecial: boolean, options?: AddedTokenOptions | undefined | null)
  getContent(): string
}
export class Tokenizer {
  constructor(model: Model)
  setPreTokenizer(preTokenizer: PreTokenizer): void
  setDecoder(decoder: Decoder): void
  setModel(model: Model): void
  setPostProcessor(postProcessor: Processor): void
  setNormalizer(normalizer: Normalizer): void
  save(path: string, pretty?: boolean | undefined | null): void
  addAddedTokens(tokens: Array<AddedToken>): number
  addTokens(tokens: Array<string>): number
  encode(
    sentence: InputSequence,
    pair?: InputSequence | null,
    encodeOptions?: EncodeOptions | undefined | null,
  ): Promise<JsEncoding>
  encodeBatch(sentences: EncodeInput[], encodeOptions?: EncodeOptions | undefined | null): Promise<JsEncoding[]>
  decode(ids: Array<number>, skipSpecialTokens: boolean): Promise<string>
  decodeBatch(ids: Array<Array<number>>, skipSpecialTokens: boolean): Promise<string[]>
  static fromString(s: string): Tokenizer
  static fromFile(file: string): Tokenizer
  static fromPretrained(file: string, parameters?: JsFromPretrainedParameters | undefined | null): Tokenizer
  addSpecialTokens(tokens: Array<string>): void
  setTruncation(maxLength: number, options?: TruncationOptions | undefined | null): void
  disableTruncation(): void
  setPadding(options?: PaddingOptions | undefined | null): void
  disablePadding(): void
  getDecoder(): Decoder | null
  getNormalizer(): Normalizer | null
  getPreTokenizer(): PreTokenizer | null
  getPostProcessor(): Processor | null
  getVocab(withAddedTokens?: boolean | undefined | null): Record<string, number>
  getVocabSize(withAddedTokens?: boolean | undefined | null): number
  idToToken(id: number): string | null
  tokenToId(token: string): number | null
  train(files: Array<string>): void
  runningTasks(): number
  postProcess(
    encoding: Encoding,
    pair?: Encoding | undefined | null,
    addSpecialTokens?: boolean | undefined | null,
  ): Encoding
}
export class Trainer {}
