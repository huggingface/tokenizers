# Examples

An example application, written in React and built/served with Vite,
that showcases the WebAssembly bindings for tokenizers.

From `bindings/wasm`:

```sh
make react
```

Or by hand, after `make build`:

```sh
cd examples/react && npm install && npm run dev
```

The examples link to `bindings/wasm/pkg`, so re-run `make build` after changing the Rust code.
