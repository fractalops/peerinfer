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
      name: "local-model-fixture",
      configureServer(server) {
        server.middlewares.use("/model-fixture/", (req, res, next) => {
          const path = (req.url ?? "").split("?")[0].replace(/^\//, "");
          if (
            ![
              "onnx/model_q4.onnx",
              "config.json",
              "tokenizer.json",
              "tokenizer_config.json",
            ].includes(path)
          )
            return next();
          const file = fileURLToPath(
            new URL(
              `../fixtures/smollm2-135m/${path}`,
              import.meta.url,
            ),
          );
          try {
            res.setHeader("Content-Length", statSync(file).size);
            res.setHeader("Content-Type", "application/octet-stream");
            createReadStream(file).pipe(res);
          } catch {
            res.statusCode = 404;
            res.end("Run scripts/test-model-fixture.sh first.");
          }
        });
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
