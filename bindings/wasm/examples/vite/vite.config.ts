import { defineConfig, searchForWorkspaceRoot } from "vite";

// Required to load the wasm module locally
export default defineConfig({
  server: {
    fs: {
      allow: [searchForWorkspaceRoot(process.cwd()), "../../pkg"],
    },
  },
});
