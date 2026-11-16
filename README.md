# peerinfer

peerinfer runs language models on browser GPUs and connects the browsers over [iroh](https://www.iroh.computer). You can chat in the browser, share your GPU with the pool, or split one model across two browsers.

[Open peerinfer](https://mfundo.wtf/peerinfer/)

## Chat

1. Open the page.
2. Choose a model.
3. Type a message.

The page sends your request to a ready worker for that model and streams the reply. You do not need a GPU to chat.

Messages are encrypted in transit. The worker that runs your request can read it. Conversations stay in the current tab. Reloading the page clears them.

## Share your GPU

Choose a model and click **Share compute**. The browser downloads the model once and caches it. Keep the tab visible while you share. **Pause** stops requests until you resume. **Stop** releases the GPU.

| Model | Download | Notes |
| --- | --- | --- |
| SmolLM2 135M | 182 MB | Runs whole or split across two browsers. Requests are limited to 1,024 tokens, including the reply. |
| SmolLM2 360M | 210 MB | Runs whole. |
| Qwen3 4B | 2.28 GB | Runs whole. Needs about 3.4 GB of GPU memory. |

The browser must support WebGPU.

## Split a model across two browsers

This is a preview. It works with SmolLM2 135M only.

1. On the first device, choose SmolLM2 135M, click **Share compute**, and choose **With another device**.
2. Open the invitation link on a second device that supports WebGPU.
3. On the second device, click **Join this team**.

The first device runs layers 0 to 14. The second device downloads from the first device only the weights it needs for layers 15 to 29, and runs them. When both devices are ready, the pair accepts chat requests as one worker.

If either device stops or disconnects, the pair ends. Start a new pair to continue. Each device keeps the weights it downloaded. Only pair devices you trust.

## Run your own pool

A pool needs an always-on bootstrap node so that browsers can find each other. On the server, install and start it:

```
cargo install --git https://github.com/fractalops/peerinfer peerinfer
peerinfer bootstrap
```

The command prints its endpoint ID. Open the page with `?bootstrap=<ENDPOINT_ID>`.

You can also send a prompt from the command line. On a laptop with peerinfer installed:

```
peerinfer ask --bootstrap <ENDPOINT_ID> "Explain peer-to-peer networking briefly."
```

See the [reference](docs/reference.md) for all commands, options and the wire format.

## License

MIT or Apache-2.0. Model weights keep their upstream licenses.
