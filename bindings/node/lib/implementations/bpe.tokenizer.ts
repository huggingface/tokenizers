import { promisify } from "util";
import { BaseTokenizer } from "./base.tokenizer";
import { Model, models } from "../bindings/models";
import { Tokenizer } from "../bindings/tokenizer";

interface BPEOptions {
  dropout?:    number;
  mergesFile?: string;
  suffix?:     string;
  unkToken?:   string;
  vocabFile?:  string;
}

const defaultBPEOptions: BPEOptions & Required<Pick<BPEOptions, 'unkToken' | 'suffix'>> = {
  suffix:   '</w>',
  unkToken: '<unk>'
};

/**
 * Instantiate and returns a new BPE tokenizer
 * @param options 
 */
export async function getBPETokenizer(options?: BPEOptions): Promise<BPETokenizer> {
  const mergedOptions = { ...defaultBPEOptions, ...options };

  let model: Model;
  if (mergedOptions.vocabFile && mergedOptions.mergesFile) {
    const fromFiles = promisify(models.BPE.fromFiles);
    const modelOptions: models.BPE.BPEOptions = {
      dropout:         mergedOptions.dropout,
      endOfWordSuffix: mergedOptions.suffix,
      unkToken:        mergedOptions.unkToken
    };

    model = await fromFiles(mergedOptions.vocabFile, mergedOptions.mergesFile, modelOptions);
  } else {
    model = models.BPE.empty();
  }

  const tokenizer = new Tokenizer(model);
  return new BPETokenizer(tokenizer);
}

/**
 * Original BPE Tokenizer.
 * Represents the BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909)
 */
class BPETokenizer extends BaseTokenizer {
  constructor(tokenizer: Tokenizer) {
    super(tokenizer);
  }
}
