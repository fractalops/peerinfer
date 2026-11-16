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
| SmolLM2 135M | 76 MB | Runs whole or split across two browsers. Requests are limited to 1,024 tokens, including the reply. |
| SmolLM2 360M | 210 MB | Runs whole or split across two browsers. Requests are limited to 1,024 tokens, including the reply. |
| Qwen3 4B | 2.28 GB | Runs whole. Needs about 3.4 GB of GPU memory. |

The browser must support WebGPU.

## Split a model across devices

This is a preview. It works with SmolLM2 135M and SmolLM2 360M, on 2 to 4 devices.

1. On the first device, choose a SmolLM2 model, click **Share compute**, choose **With other devices**, and choose the number of devices.
2. On each other device, open the invitation link and click **Join team**. On a computer with the command-line tool, you can run the `peerinfer join` command that the page shows instead.

Each device runs a range of the model's layers. The other devices receive from the first device only the weight files their layers need. When all devices are ready, the team accepts chat requests as one worker.

If a device stops or disconnects, the team ends. Start a new team to continue. Each device keeps the weights it received. Only invite devices you trust.

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

A machine with a GPU can answer requests too. On a computer with a model in MLC `q4f16_1` format, start a worker:

```
peerinfer worker --bootstrap <ENDPOINT_ID> --model SmolLM2-360M-Instruct-q4f16_1-MLC
```

To divide a model's layers between several computers, start the worker with `--devices 3`. It prints a `peerinfer join` command. On each of the other two computers, run that command.

See the [reference](docs/reference.md) for all commands, options and the wire format.

## License

MIT or Apache-2.0. Model weights keep their upstream licenses.
