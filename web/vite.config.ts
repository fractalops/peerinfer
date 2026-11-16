import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { browserNetworkPlugin } from "./build/browser-network.mjs";

export default defineConfig({
  base: "./",
  build: { target: "es2022" },
  worker: { format: "es" },
  // Reloading a live provider destroys its peer identity and in-memory seed store.
  server: {
    hmr: false,
    watch: null,
    host: "127.0.0.1",
    port: 5178,
    strictPort: true,
  },
  plugins: [
    svelte(),
    browserNetworkPlugin(),
    {
      name: "local-mlc-fixture",
      configureServer(server) {
        // MLC weights in fixtures/<model>/, served like Hugging Face's
        // resolve/main/ paths for ?weights= in development.
        server.middlewares.use("/mlc-fixture/", (req, res, next) => {
          const m = /^\/([\w.-]+)\/resolve\/main\/([\w.-]+)$/.exec(
            (req.url ?? "").split("?")[0],
          );
          if (!m) return next();
          const file = fileURLToPath(
            new URL(`../fixtures/${m[1]}/${m[2]}`, import.meta.url),
          );
          try {
            res.setHeader("Content-Length", statSync(file).size);
            res.setHeader("Content-Type", "application/octet-stream");
            createReadStream(file).pipe(res);
          } catch {
            res.statusCode = 404;
            res.end();
          }
        });
      },
    },
  ],
});
