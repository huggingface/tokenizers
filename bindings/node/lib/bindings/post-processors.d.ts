/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of
 * a PostProcessor will return an instance of this class when instantiated.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface PostProcessor {}

/**
 * Instantiate a new BertProcessing with the given tokens
 *
 * @param sep A tuple with the string representation of the SEP token, and its id
 * @param cls A tuple with the string representation of the CLS token, and its id
 */
export function bertProcessing(
  sep: [string, number],
  cls: [string, number]
): PostProcessor;

/**
 * Instantiate a new ByteLevelProcessing.
 *
 * @param [trimOffsets=true] Whether to trim the whitespaces from the produced offsets.
 * Takes care of trimming the produced offsets to avoid whitespaces.
 * By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you
 * don't want the offsets to include these whitespaces, then this processing step must be used.
 * @since 0.6.0
 */
export function byteLevelProcessing(trimOffsets?: boolean): PostProcessor;

/**
 * Instantiate a new RobertaProcessing with the given tokens
 *
 * @param sep A tuple with the string representation of the SEP token, and its id
 * @param cls A tuple with the string representation of the CLS token, and its id
 * @param [trimOffsets=true] Whether to trim the whitespaces in the produced offsets
 * @param [addPrefixSpace=true] Whether addPrefixSpace was ON during the pre-tokenization
 */
export function robertaProcessing(
  sep: [string, number],
  cls: [string, number],
  trimOffsets?: boolean,
  addPrefixSpace?: boolean
): PostProcessor;

/**
 * Instantiate a new TemplateProcessing.
 *
 * @param single A string describing the template for a single sequence
 * @param pair A string describing the template for a pair of sequences
 * @param specialTokens An array with all the special tokens
 */
export function templateProcessing(
  single: string,
  pair?: string,
  specialTokens?: [string, number][]
): PostProcessor;
