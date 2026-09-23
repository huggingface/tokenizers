import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import init, { Tokenizer } from "tokenizers-wasm";
import "../style.css";

const MODELS = [
  ["openai-community/gpt2", "BPE"],
  ["google-bert/bert-base-uncased", "WordPiece"],
  ["google-t5/t5-small", "Unigram"],
  ["Qwen/Qwen3-0.6B", "BPE"],
];

function App() {
  const [model, setModel] = useState(MODELS[0][0]);
  const [tokenizer, setTokenizer] = useState<Tokenizer>();
  const [text, setText] = useState("Hello world! This text is tokenized in your browser.");

  useEffect(() => {
    let loaded: Tokenizer | undefined;
    let cancelled = false;
    fetch(`https://huggingface.co/${model}/resolve/main/tokenizer.json`)
      .then((response) => response.text())
      .then((json) => {
        if (cancelled) return;
        loaded = Tokenizer.from_json(json);
        setTokenizer(loaded);
      });
    // Tokenizers live in WebAssembly memory, so free the previous one instead of waiting for the GC.
    return () => {
      cancelled = true;
      setTokenizer(undefined);
      loaded?.free();
    };
  }, [model]);

  const ids = tokenizer?.encode(text) ?? new Uint32Array();
  const tokens = tokenizer?.decode_tokens(ids) ?? [];

  return (
    <main>
      <h1>
        Tokenizers x <mark>React</mark>
      </h1>

      <div className="chips">
        {MODELS.map(([name, kind]) => (
          <button key={name} className={name === model ? "chip on" : "chip"} onClick={() => setModel(name)}>
            {name.split("/")[1]} <small>{kind}</small>
          </button>
        ))}
      </div>
      <p className="status">{tokenizer ? `Loaded ${model}` : `Loading ${model}…`}</p>

      <div className="panel pad">
        <label className="label" htmlFor="input">
          Text
        </label>
        <textarea id="input" rows={5} spellCheck={false} value={text} onChange={(event) => setText(event.target.value)} />
      </div>

      <div className="panel">
        <div className="panel-head">
          <span>Tokens</span>
          <span>{tokenizer && `${ids.length} tokens`}</span>
        </div>
        <div className="pad tokens">
          {tokens.map((token, i) => (
            <span key={i} className="token">
              <b>{token}</b>
              <small>{ids[i]}</small>
            </span>
          ))}
        </div>
      </div>
    </main>
  );
}

await init();
createRoot(document.getElementById("root")!).render(<App />);
