#!/usr/bin/env bash
# Pinned SmolLM2 tokenizer files for the tokenizer integration test.
set -euo pipefail
cd "$(dirname "$0")/.."
root=fixtures/smollm2-135m
revision=12fd25f77366fa6b3b4b768ec3050bf629380bac
mkdir -p "$root"
for file in tokenizer.json tokenizer_config.json; do
  if [[ ! -s "$root/$file" ]]; then
    curl --fail --location --retry 3 \
      "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/$revision/$file" \
      --output "$root/$file.part"
    mv "$root/$file.part" "$root/$file"
  fi
done
