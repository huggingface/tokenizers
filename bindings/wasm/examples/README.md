# Examples

Two versions of the same small app, one in plain TypeScript (`vite`) and one in React (`react`),
both built with Vite.

From `bindings/wasm`:

```sh
make vite
make react
```

Or by hand, after `make build`:

```sh
cd examples/vite && npm install && npm run dev
```

The examples link to `bindings/wasm/pkg`, so re-run `make build` after changing the Rust code.
