/**
 * This class is not supposed to be instantiated directly. Instead, any implementation of
 * a PostProcessor will return an instance of this class when instantiated.
 */
declare class PostProcessor {}

export namespace postProcessors {
  /**
   * Instantiate a new BertProcessing with the given tokens
   *
   * @param {[string, number]} sep A tuple with the string representation of the SEP token, and its id
   * @param {[string, number]} cls A tuple with the string representation of the CLS token, and its id
   */
  export function BertProcessing(sep: [string, number], cls: [string, number]): PostProcessor;
}
