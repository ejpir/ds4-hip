# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

## CyberNeurova DeepSeek-V4-Flash GGUF Notes

CyberNeurova's abliterated Q2_K artifact is not a generic DeepSeek V3/V3.2-style
shape. It is a `general.architecture = deepseek4` GGUF based on
`deepseek-ai/DeepSeek-V4-Flash` with these fixed metadata values:

- Layers: 43
- Embedding dim: 4096
- Vocab: 129280
- Context: 1048576
- Attention heads: 64
- KV heads: 1
- Key/value dim: 512 / 512
- Q/O LoRA ranks: 1024 / 1024
- Output groups: 8
- Sliding window: 128
- MoE experts: 256 total, 6 active, 1 shared
- Expert FF dim: 2048
- Indexer: 64 heads, head dim 128, top-k 512
- Compression ratios: layers 0..1 uncompressed, then alternating ratio 4 and
  ratio 128 through layer 42. Ratio-4 layers have the indexer stream.

Important tensor/GEMM shapes in GGUF stored order:

- `token_embd.weight`, `output.weight`: Q8_0 `[4096, 129280]`
- Per layer attention:
  - `attn_q_a.weight`: Q8_0 `[4096, 1024]`
  - `attn_q_b.weight`: Q8_0 `[1024, 32768]` (`64 * 512`)
  - `attn_kv.weight`: Q8_0 `[4096, 512]`
  - `attn_output_a.weight`: Q8_0 `[4096, 8192]`
  - `attn_output_b.weight`: Q8_0 `[8192, 4096]`
- Attention compressor:
  - Ratio-4 layers, 21 layers: `attn_compressor_{gate,kv}.weight` Q8_0
    `[4096, 1024]`, `attn_compressor_ape.weight` Q8_0 `[1024, 4]`
  - Ratio-128 layers, 20 layers: `attn_compressor_{gate,kv}.weight` Q8_0
    `[4096, 512]`, `attn_compressor_ape.weight` Q8_0 `[512, 128]`
- Ratio-4 indexer tensors, 21 layers:
  - `indexer.attn_q_b.weight`: Q8_0 `[1024, 8192]` (`64 * 128`)
  - `indexer.proj.weight`: Q8_0 `[4096, 64]`
  - `indexer_compressor_{gate,kv}.weight`: Q8_0 `[4096, 256]`
  - `indexer_compressor_ape.weight`: Q8_0 `[256, 4]`
- Per layer MoE:
  - `ffn_gate_inp.weight`: F32 `[4096, 256]`
  - routed gate/up experts: Q2_K `[4096, 2048, 256]`
  - routed down experts: Q2_K `[2048, 4096, 256]`
  - shared gate/up experts: Q8_0 `[4096, 2048]`
  - shared down expert: Q8_0 `[2048, 4096]`
- Extras:
  - `ffn_gate_tid2eid.weight`: I32 `[6, 129280]` on layers 0..2
  - `exp_probs_b.bias`: F32 `[256]` on layers 3..42
  - HC function weights: Q8_0 `[16384, 24]` per layer/stage
  - output HC function: Q8_0 `[16384, 4]`

The CyberNeurova Q2_K file has 1328 tensors total: 661 Q8_0, 535 F32, 129 Q2_K,
and 3 I32. Unlike the DS4 preferred q2-imatrix mix, all routed gate/up/down
experts in this artifact are Q2_K; attention, embeddings, output, shared
experts, compressors, and indexer projections remain Q8_0.
