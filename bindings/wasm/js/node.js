import { readFileSync } from "node:fs";
import { initSync } from "./tokenizers_web.js";

export * from "./tokenizers_web.js";

// Node's fetch can't read file: URLs, which the default init relies on.
initSync({ module: readFileSync(new URL("tokenizers_web_bg.wasm", import.meta.url)) });
