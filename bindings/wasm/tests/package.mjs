import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const [packageName, tokenizerPath] = process.argv.slice(2);
const { Tokenizer } = await import(packageName);

function roundTrip(name, tokenizer) {
  const text = "Hello world, from the packed tarball!";
  const { ids } = tokenizer.encode(text);
  assert.ok(ids.length > 0);
  assert.equal(tokenizer.decode(ids), text);
  console.log(`${packageName} ${name}: ${ids.length} ids, round trip ok`);
}

roundTrip("fromJson", Tokenizer.fromJson(readFileSync(tokenizerPath, "utf8")));
roundTrip(
  "fromPretrained",
  await Tokenizer.fromPretrained("openai-community/gpt2", { token: process.env.HF_TOKEN || undefined }),
);
