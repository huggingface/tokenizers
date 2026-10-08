import { readFileSync } from "node:fs";
import { initSync } from "./tokenizers_wasm.js";

export * from "./tokenizers_wasm.js";

// Node's fetch can't read file: URLs, which the default init relies on.
initSync({ module: readFileSync(new URL("tokenizers_wasm_bg.wasm", import.meta.url)) });
