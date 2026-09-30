import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const [packageName, tokenizerPath] = process.argv.slice(2);
const { Tokenizer, initSync } = await import(packageName);

// Node's fetch can't read file: URLs, so the default async init fails there.
const wasm = readFileSync(fileURLToPath(import.meta.resolve(`${packageName}/tokenizers_wasm_bg.wasm`)));
initSync({ module: wasm });

function roundTrip(name, tokenizer) {
  const text = "Hello world, from the packed tarball!";
  const ids = tokenizer.encode(text);
  assert.ok(ids.length > 0);
  assert.equal(tokenizer.decode(ids), text);
  tokenizer.free();
  console.log(`${packageName} ${name}: ${ids.length} ids, round trip ok`);
}

roundTrip("from_json", Tokenizer.from_json(readFileSync(tokenizerPath, "utf8")));
roundTrip(
  "from_pretrained",
  await Tokenizer.from_pretrained("openai-community/gpt2", { token: process.env.HF_TOKEN || undefined }),
);
