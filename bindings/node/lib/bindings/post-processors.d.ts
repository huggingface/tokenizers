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
 * Instantiate a new RobertaProcessing with the given tokens
 *
 * @param sep A tuple with the string representation of the SEP token, and its id
 * @param cls A tuple with the string representation of the CLS token, and its id
 */
export function robertaProcessing(
    sep: [string, number],
    cls: [string, number]
): PostProcessor;
