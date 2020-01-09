import { promisify } from "util";
import { BaseTokenizer } from "./base.tokenizer";
import { Tokenizer } from "../bindings/tokenizer";
import { Model, models } from "../bindings/models";

interface BertWordpieceOptions {
  /**
   * @default true
   */
  addSpecialTokens?:   boolean;
  /**
   * @default true
   */
  cleanText?:          boolean;
  /**
   * @default "[CLS]"
   */
  clsToken?:           string;
  /**
   * @default true
   */
  handleChineseChars?: boolean;
  /**
   * @default true
   */
  lowercase?:          boolean;
  /**
   * @default "[SEP]"
   */
  sepToken?:           string;
  /**
   * @default true
   */
  stripAccents?:       boolean;
  /**
   * @default "[UNK]"
   */
  unkToken?:           string;
  vocabFile?:          string;
  /**
   * @default "##"
   */
  wordpiecesPrefix?:   string;
}

const defaultBertOptions: Required<Omit<BertWordpieceOptions, 'vocabFile'>> & { vocabFile?: string } = {
  addSpecialTokens:   true,
  cleanText:          true,
  clsToken:           '[CLS]',
  handleChineseChars: true,
  lowercase:          true,
  sepToken:           '[SEP]',
  stripAccents:       true,
  unkToken:           '[UNK]',
  wordpiecesPrefix:   '##'
};

/**
 * Instantiate and returns a new Bert WordPiece tokenizer
 * @param options 
 */
export async function getBertWordpieceTokenizer(options?: BertWordpieceOptions): Promise<BertWordpieceTokenizer> {
  const mergedOptions = { ...defaultBertOptions, ...options };

  let model: Model;
  if (mergedOptions.vocabFile) {
    const fromFiles = promisify(models.WordPiece.fromFiles);
    model = await fromFiles(mergedOptions.vocabFile, mergedOptions.unkToken, null);
  } else {
    model = models.WordPiece.empty();
  }

  const tokenizer = new Tokenizer(model);
  return new BertWordpieceTokenizer(tokenizer);
}

/**
 * Bert WordPiece Tokenizer
 */
class BertWordpieceTokenizer extends BaseTokenizer {
  constructor(tokenizer: Tokenizer) {
    super(tokenizer);
  }
}
