# Reference

## Commands

### `peerinfer ask`

Sends one prompt and prints the reply to standard output while it arrives.

```
peerinfer ask [OPTIONS] [PROMPT]
```

If you omit `PROMPT`, the command reads the prompt from standard input.

| Option | Description |
|---|---|
| `-w, --worker <ID>` | Send the prompt to this worker. The pool is not used. |
| `--bootstrap <ID>` | Bootstrap node to join the pool. Repeat it or separate IDs with commas. Environment: `PEERINFER_BOOTSTRAP`. |
| `--pool <NAME>` | Pool name. Default: `peerinfer/pool/v1`. |
| `--system <TEXT>` | System message to send before the prompt. |
| `--max-tokens <N>` | Maximum number of tokens to generate. |
| `--temperature <T>` | Sampling temperature. |
| `--wait <S>` | Seconds to wait for a worker to appear in the pool. Default: 15. |
| `--connect-timeout <S>` | Seconds to wait for a connection to one worker. Default: 30. |
| `-v, --verbose` | Print the worker, model and token counts to standard error. |

Without `--worker`, the command joins the pool and orders the workers:

1. Idle workers, fastest first by reported speed.
2. Busy workers, shortest queue first.

Paused, loading and stale workers are not used. If a worker is busy, paused or cannot be reached, the command tries the next one. It tries at most 3 workers. If a worker fails after it has sent text, the command stops and does not retry, so you never see the same text twice.

The command exits with status 0 when the reply is complete and 1 on an error.

### `peerinfer peers`

Joins the pool, listens for 12 seconds, and prints the workers.

```
peerinfer peers [--wait <S>] [--bootstrap <ID>] [--pool <NAME>]
```

### `peerinfer bootstrap`

Runs an always-on pool member. Other members connect to it first. It prints joins, leaves and timeouts.

```
peerinfer bootstrap [--key-file <PATH>] [--bootstrap <ID>] [--pool <NAME>]
```

The node keeps its secret key in `--key-file`. The default is `~/.config/peerinfer/bootstrap.key`. The endpoint ID stays the same while the file exists. The command creates the file with mode 0600 if it does not exist. You can run more than one bootstrap node. Give each node the IDs of the others with `--bootstrap`.

### `peerinfer worker`

Answers pool requests with a model on this machine's GPU. With `--devices N`, the worker hosts a team: the model's layers are divided into `N` contiguous ranges that differ by at most one layer, the worker runs the first range, and it prints a `peerinfer join` command for the other devices.

```
peerinfer worker --model <DIR> [--model-id <ID>] [--devices <N>] [--key-file <PATH>] [--bootstrap <ID>] [--pool <NAME>]
```

| Option | Description |
|---|---|
| `--model <DIR>` | Model directory in MLC `q4f16_1` format: `mlc-chat-config.json`, `tensor-cache.json` or `ndarray-cache.json`, the weight shards and `tokenizer.json`. |
| `--model-id <ID>` | Model ID to announce. Clients choose workers by this ID. Default: the directory name. |
| `--devices <N>` | Devices that share the model, including this one. Default: 1. |
| `--key-file <PATH>` | Secret key file. The endpoint ID stays the same while the file exists. Default: `~/.config/peerinfer/worker.key`. |

The worker uses the chat template and stop tokens in `mlc-chat-config.json`. A request supports 2,048 tokens including the reply and at most 512 generated tokens. A team accepts requests only while every member is connected and has loaded its layers.

### `peerinfer join`

Runs one range of a team's layers on this machine's GPU.

```
peerinfer join <INVITATION> [--cache <DIR>]
```

The host assigns the next free layer range. The command receives from the host only the model files that its layers need, checks them against their BLAKE3 hashes, and saves them in `--cache` (default: `~/.cache/peerinfer/blobs`) for the next time. It runs until you press Ctrl-C or the host stops.

### Logs

Set `PEERINFER_LOG` to a tracing filter, for example `PEERINFER_LOG=peerinfer_net=debug`. Logs go to standard error.

## Web page

Choose a model and chat. Routing uses only non-stale workers advertising that exact model; idle workers are preferred, followed by short queues. It tries up to three candidates before any text arrives and never silently retries a partial answer. Stop cancels the response. Only completed turns are included in follow-ups; changing models starts a new conversation.

**Share compute** offers whole-model execution, and a team preview for SmolLM2 135M and 360M on 2 to 4 devices. GPU checks run before downloading. Downloads report progress and caches survive visits. Weights received from another device are content-verified; weight shards downloaded from Hugging Face are length-checked. Stop during preparation cancels loading. Pause interrupts active generation and rejects new jobs; resume makes the existing session available again. Stop ends the contribution and releases its GPU worker.

The **Network** dashboard shows availability, active and queued requests, reported completions and token totals, and a searchable worker table. The request chart and activity log record observations from the current tab session. Totals describe workers currently in view; they are not lifetime pool counters. Connection details show the local endpoint and pool. A shared team appears as one logical worker; it becomes ready only after both stages load. Select a worker for live details and its reported execution configuration. The diagram does not report individual remote stage health. Green dots indicate connected or available states; model availability respects worker queue capacity. An invited stage does not advertise a complete model. Worker availability is self-reported.

| Query | Description |
| --- | --- |
| `?bootstrap=ID1,ID2` | Overrides the build-time bootstrap list. An empty value creates a standalone local pool. |
| `?model=ID` | Selects a listed model: `SmolLM2-135M-Instruct-q4f16_1-MLC` (default), `SmolLM2-360M-Instruct-q4f16_1-MLC`, or `Qwen3-4B-q4f16_1-MLC`. |
| `?weights=URL` | Development builds only. Loads the selected SmolLM2 model's MLC weights from `URL` instead of Hugging Face. The development server serves `fixtures/<model>/` at `/mlc-fixture/<model>/resolve/main/`. |
| `#join=…` | Invitation capability for the second stage. Open it only for a host you trust. The fragment is not sent to the static host. |

Each tab has a fresh endpoint identity. Verified model caches use content hashes and revision keys independently of that identity. No conversation history or prompt data is persisted by the app. Browsers can suspend hidden tabs; keep contributors visible. A disconnected shared team must be restarted; it cannot currently recover KV state or replace a missing stage.

SmolLM2 135M and 360M run on the [peerinfer-core](https://github.com/fractalops/peerinfer-core) engine in a Web Worker. They use the pinned SmolLM2 tokenizer and chat template and stop at the end-of-message token. Precise mode decodes greedily; Creative mode samples with temperature over the whole vocabulary. A request supports 1,024 tokens including the reply and at most 512 generated tokens; the chat UI requests up to 256. The engine needs WebGPU with its default limits. It shares the downloaded weight cache with WebLLM, which runs Qwen3 4B.

In a team, each device runs a contiguous range of layers, and the page that started the team is the host. Browsers and computers running `peerinfer join` can be in the same team. A member receives only the model files its layers need, checks them against their BLAKE3 hashes and saves them for the next time. See [Team protocol](#team-protocol) for the messages.

## Pool membership

- Members join a gossip topic derived from the pool name.
- An idle worker sends a signed heartbeat every 10 seconds. Status changes wake the sender. Worker announcements are limited to one every 2 seconds, including replies to new members. Each message is at most 2 KB.
- Contributions report queue state and counters once a second. The page reads the roster when the Wasm node signals a change.
- Replayed and reordered status and leave messages are ignored. Sequence floors remain for ten minutes after a worker leaves or expires; a replay after that window can briefly reappear until expiry.
- Members check each signature against the sender's endpoint ID and drop messages that fail.
- A worker that sends nothing for 30 seconds is shown as stale. After 45 seconds it is removed.
- A worker that leaves on purpose sends a leave message and is removed at once.
- Observers do not announce themselves. Only workers appear in the list.

## Network

- Every member is an iroh endpoint. The endpoint ID is the public key of the node.
- Browser tabs connect through iroh relay servers. Browsers cannot send UDP, so they cannot make direct connections. The relay cannot read the traffic.
- The command-line tool and the bootstrap node use the public n0 relays and address lookup.

## Wire format

ALPN: `peerinfer/infer/0`. Each completion uses one bidirectional QUIC stream.

1. The client writes one `Request` frame and finishes its send side.
2. The worker writes zero or more `Chunk` frames.
3. The worker writes one `Done` or `Error` frame and finishes.

A frame is a 4-byte big-endian length and a [postcard](https://docs.rs/postcard) body. The largest frame is 1 MiB. The types are in `crates/net/src/proto.rs`.

| Error code | Meaning |
|---|---|
| `Busy` | The worker queue is full. |
| `Paused` | The worker is paused. |
| `BadRequest` | The request has no messages. |
| `Inference` | The model failed. |

If the stream ends before `Done` or `Error`, the client reports an incomplete completion.

## Team protocol

ALPN: `peerinfer/team/0`. A team runs one model on several devices. Each device runs a contiguous range of layers. The host runs the first range, answers pool requests and offers the model files. The types are in `crates/net/src/team.rs`.

The invitation is the host's endpoint address and a 16-byte secret, as postcard bytes in hexadecimal.

1. A member connects to the host and keeps the connection. On a new stream it sends `Join` with the secret. The host answers `Assigned` with the next free layer range and the manifest: the model ID, its number of layers, and the name, BLAKE3 hash and size of each file.
2. The member receives the files that its layers need with iroh-blobs (ALPN `/iroh-bytes/4`), which checks each file against its hash.
3. The member loads its layers and sends `Ready`.
4. For each request, the host opens one stream per message on the member's connection:
   - `Begin` with a run number, the temperature and the sampling seed. The member clears its context.
   - `Step` with the run number, the step number and the hidden states from the stage before it: `hidden_size` f32 values per token. A middle member answers `Hidden` with its output; the last member answers `Token`. The host sends the members' steps in layer order.
   - `End` with the run number.
5. A member sends `Leave`, or closes its connection, to leave the team.

Each message and each reply is one frame, in the same format as the completion protocol. The largest team frame is 64 MiB. A member rejects a `Step` for an unknown run or out of order, and a `Begin` with a run number it has seen before. The host waits up to 120 s for each reply. A team takes requests only while every member is connected and ready.

## Build from source

You need Rust with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` at the same version as the `wasm-bindgen` crate in `Cargo.lock`, and Node.js 22.18 or later. On macOS, also install LLVM (`brew install llvm`). Apple clang cannot compile C code for wasm32.

Build and run the command-line tool:

```
cargo run -p peerinfer -- --help
```

Build the web page:

```
./scripts/build-wasm.sh
cd web
npm ci
npm run dev
```

The development server does not watch files. A reload would end a running contribution and its invitation. Restart the server and reload the page after each change.

Run the tests and build the page:

```
cargo test --workspace --exclude peerinfer-web
npm --prefix web test
npm --prefix web run build
```

## Deployment

The `Pages` workflow builds the page and publishes it to GitHub Pages on each push to `main`. Set the repository variable `PEERINFER_BOOTSTRAP` to a comma-separated list of bootstrap endpoint IDs. The page then works without the `?bootstrap=` option.
