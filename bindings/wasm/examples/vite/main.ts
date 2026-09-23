import init, { Tokenizer } from "tokenizers-wasm";
import "../style.css";

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const buttons = document.querySelectorAll<HTMLButtonElement>("[data-model]");
const form = byId<HTMLFormElement>("form");
const input = byId<HTMLTextAreaElement>("input");
const encodeButton = byId<HTMLButtonElement>("encode");
const status = byId<HTMLParagraphElement>("status");
const count = byId<HTMLSpanElement>("count");
const tokens = byId<HTMLDivElement>("tokens");

let tokenizer: Tokenizer | undefined;

async function load(button: HTMLButtonElement) {
  const model = button.dataset.model!;
  encodeButton.disabled = true;
  for (const other of buttons) {
    other.disabled = true;
    other.classList.toggle("on", other === button);
  }
  status.textContent = `Loading ${model}…`;

  try {
    const response = await fetch(`https://huggingface.co/${model}/resolve/main/tokenizer.json`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const loaded = Tokenizer.from_json(await response.text());
    // Tokenizers live in WebAssembly memory, so free the previous one instead of waiting for the GC.
    tokenizer?.free();
    tokenizer = loaded;
    status.textContent = `Loaded ${model}`;
  } catch (error) {
    status.textContent = `Could not load ${model}: ${(error as Error).message}`;
    return;
  } finally {
    for (const other of buttons) other.disabled = false;
    encodeButton.disabled = !tokenizer;
  }
  render();
}

function render() {
  const ids = tokenizer!.encode(input.value);
  const texts = tokenizer!.decode_tokens(ids);
  tokens.replaceChildren(...texts.map((text, i) => sticker(text, ids[i])));
  count.textContent = `${ids.length} tokens`;
}

function sticker(text: string, id: number) {
  const token = document.createElement("span");
  token.className = "token";
  const label = document.createElement("b");
  label.textContent = text;
  const number = document.createElement("small");
  number.textContent = String(id);
  token.append(label, number);
  return token;
}

await init();
for (const button of buttons) button.addEventListener("click", () => load(button));
form.addEventListener("submit", (event) => {
  event.preventDefault();
  render();
});
load(buttons[0]);
