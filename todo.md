# DS4 HIP TODO

## Current baseline

- Target model: CyberNeurova DeepSeek V4 Flash Q2_K GGUF, ~92 GiB tensor payload.
- Target GPU: AMD Radeon 8060S / gfx1151, wave32.
- Default server mode should remain conservative: zero-copy mapped GGUF, managed tensors, ctx 4096.
- Full-copy mode is working and opt-in:
  - `DS4_HIP_COPY_MODEL=1`
  - staged pinned copy, `pread()` -> `hipHostMalloc` buffer -> `hipMemcpyAsync()` -> device allocation
  - observed: 92.02 GiB copied in ~22.7s, ~4.05 GiB/s, no 27 GiB stall.
- Current fast prefill baseline:
  - Q8 batch + MoE expert batch: ~17-19 tok/s on 774-1741 token prompts
  - full-copy + device tensors: ~20 tok/s on 774-token prompt
- Measured 1029-token prefill split with current fast paths and stage profiling:
  - routed Q2_K MoE: ~47.9%
  - dense attention/output Q8 projections: ~35.4%
  - shared expert Q8: ~5.7%
  - attention/cache/compressor/indexer: ~8.4%
  - other FFN/attention glue: ~2.7%
  - implication: for prefill, Q8 WMMA alone is useful but not enough; Q2_K MoE WMMA/grouped expert work must follow closely.
  - stage-count sanity: 129 records = 43 layers * 3 chunks; compressor 123 = 41 compressed layers * 3 chunks; indexer_setup 63 = 21 ratio-4 layers * 3 chunks.
  - output projection is the biggest dense stage by itself (~25.8%); split and optimize `attn_output_a/b` first.
- Decode bandwidth model:
  - estimated touched compressed traffic is ~8.8 GiB/token at short context.
  - realistic bandwidth ceiling is ~19-22 tok/s depending on sustained 180-212 GB/s.
  - current ~9-10 tok/s is roughly half of that ceiling; 12 tok/s is a sane near-term target.
  - current Q8_0 and Q2_K matmul paths dequant in-kernel, not via persistent BF16 scratch weights.
  - current decode routed MoE uses one gate/up kernel and one down kernel per layer, not one launch per expert; grouped/fused dispatch is still useful but less dramatic than a 6x launch-count reduction.
  - >20 tok/s needs lower per-token traffic: selective dense requant, stronger fusion/layout, speculative decode, or multi-stream batching.
- Measured 16-token decode split with stage profiling enabled (profiling slows absolute tok/s):
  - dense Q8/F32 matvec stages: ~65.0%
  - routed Q2_K MoE: ~19.6%
  - attention/cache/indexer: ~9.4%
  - norm/HC glue: ~6.0%
  - top stages: `attn_output` ~29.8%, `q_path` ~22.5%, `routed_moe` ~19.6%.
  - decode HIP MoE split: gate/up ~213 ms, down ~134 ms across 16 tokens; down-loop fusion is not the top decode bottleneck.
  - decode Q8 hot shapes: `1024->32768` q_b, `attn_output_a/b`, then lm_head/output.
- Isolated q_b microbench using 43 real GGUF `attn_q_b` tensors and the current default q8 warp-row kernel:
  - Q8_0 bytes per q_b tensor: 34.00 MiB, not 32 MiB.
  - mapped/zero-copy weights: ~0.481 ms/call, ~69.0 GiB/s weight-only bandwidth.
  - copied device weights: ~0.477 ms/call, ~69.6 GiB/s weight-only bandwidth.
  - q_b profile math: 303.96 ms was across 688 calls = 16 tokens * 43 layers, so profile-implied bandwidth is ~75 GiB/s, not ~5.6 GiB/s.
  - implication: q_b is underutilizing memory bandwidth, but not because of zero-copy alone; device-copy is nearly identical in this isolated test.
- Additional isolated Q8 decode-shape microbench with the same generic warp-row kernel:
  - `output_a` `4096->8192`: ~0.485 ms/call mapped, ~68.5 GiB/s; device copy ~68.7 GiB/s.
  - `output_b` `8192->4096`: ~0.534 ms/call mapped, ~62.2 GiB/s; device copy ~62.2 GiB/s.
  - The actual engine output path uses split-K/HC specialized kernels and profiles faster than generic: `attn_out_low` and `attn_out_b_hc_q8` are ~0.39 ms/call, ~86 GiB/s weight-only.
  - Existing `DS4_HIP_SPLITK_Q_B=1` decode test worsened q_b (~0.65 ms/call vs ~0.44 ms/call), so leave it off.
- Multi-output-row-per-wave q_b scalar microbench, `N in {1,2,4,8,16}`:
  - baseline generic kernel: ~0.480 ms/call, ~69.1 GiB/s.
  - templated `N=1`: ~0.421 ms/call, ~78.8 GiB/s.
  - `N=2`: ~0.443-0.459 ms/call, ~72-75 GiB/s.
  - `N=4`: ~0.467-0.475 ms/call, ~70-71 GiB/s.
  - `N=8`: ~0.584 ms/call, ~56.8 GiB/s.
  - `N=16`: ~0.94 ms/call, ~35 GiB/s.
  - result: naive multi-row-per-wave does not expose the expected bandwidth scaling; it reduces parallelism/register headroom and gets worse after N=1/2.
  - rocprofv3 counters collected one-at-a-time because the combined set exceeds gfx1151 counter collection capabilities.
  - counter summary for q_b multi-N:
    - N=1: VGPR 24, waves 32768, MemUnitBusy ~95.8%, L2 hit ~9.7%, VALU insts ~18.25M.
    - N=2: VGPR 24, waves 16384, MemUnitBusy ~96.7%, L2 hit ~9.6%, VALU insts ~13.27M.
    - N=4: VGPR 40, waves 8192, MemUnitBusy ~101%, L2 hit ~7.6%, VALU insts ~11.76M.
    - N=8: VGPR 64, waves 4096, MemUnitBusy ~129%, L2 hit ~2.8%, VALU insts ~10.98M.
    - N=16: VGPR 112, waves 2048, MemUnitBusy ~223%, L2 hit ~2.6%, VALU insts ~10.60M.
    - WRITE_SIZE and WriteUnitStalled are ~0 for all N.
  - interpretation: N-sweep reduces VALU work but collapses wave count/occupancy and raises register pressure; memory units stay busy but effective bandwidth falls.
  - next kernel experiment should preserve or increase wave/block concurrency while reusing activations, e.g. shared-x/multi-wave tile, split-K variants, or rocWMMA/tile-major layout rather than serial N rows inside one wave.
- Concurrent q_b kernel diagnostic with 2/4/8 nonblocking HIP streams:
  - serial mapped baseline: ~0.483 ms/call, ~68.8 GiB/s.
  - conc2: ~0.495 ms/call aggregate, ~67.1 GiB/s.
  - conc4: ~0.494 ms/call aggregate, ~67.2 GiB/s.
  - conc8: ~0.495 ms/call aggregate, ~67.1 GiB/s.
  - rocprof kernel trace for conc4 shows kernels do overlap (max overlap ~3), but each overlapped kernel stretches to ~1.4-1.9 ms, so aggregate bandwidth does not improve.
  - implication: single-kernel q_b is not merely queue-depth limited; concurrent independent q_b streams contend for the same bottleneck and do not reveal hidden 140+ GiB/s headroom.
  - For decode >12 tok/s, focus on reducing/reshaping traffic and fused/specialized kernels, not just more concurrent q_b work.
- GTT bulk bandwidth sanity check on 34 MiB device allocations:
  - `hipMemcpyDeviceToDevice`: ~200.6 GiB/s counting read+write traffic (~215 GB/s decimal).
  - simple kernel copy: ~202.4 GiB/s counting read+write traffic (~217 GB/s decimal).
  - simple read-only kernel: ~257 GiB/s if counted as payload bytes, likely cache/reuse/measurement-inflated but clearly far above q_b.
  - implication: DRAM/GTT bulk bandwidth is not capped at ~70 GiB/s; q_b/output Q8 kernels are access-pattern/instruction-structure limited. Layout/repack or more efficient Q8 kernel structure can still help.
- Vectorized-load q_b microbench on current GGUF Q8_0 layout:
  - baseline byte-per-lane: ~0.483 ms/call, ~68.8 GiB/s.
  - vec2 `uint16_t` loads, 16 active lanes: ~0.563 ms/call, ~59.0 GiB/s.
  - vec4 unaligned `uint32_t` loads, 8 active lanes: ~0.681 ms/call, ~48.7 GiB/s.
  - rocprof SQ_WAVES stays 32768 for all variants; VGPR vec2=16, vec4=24.
  - result: vectorized loads on the existing 34-byte Q8_0 block layout make q_b slower, likely because fewer active lanes/poorer coalescing/misaligned 32-bit loads outweigh lower instruction count.
  - implication: simple load widening is not enough; any vectorized path likely needs weight repack/aligned tile layout.
- Repacked q_b layout experiment:
  - layout: int8 weights contiguous/aligned `[out_dim, in_dim]`, fp16 scales separate `[out_dim, in_dim/32]`; logical bytes remain 34.00 MiB/tensor.
  - host repack of all 43 q_b tensors took ~0.115 s in the microbench (device copy not counted in that number).
  - repacked byte-per-lane kernel: ~0.336 ms/call, ~98.8 GiB/s logical bandwidth; ~1.43x faster than raw GGUF baseline (~0.483 ms/call, ~68.8 GiB/s).
  - repacked `uint16_t` vector loads: ~0.429 ms/call, ~77.3 GiB/s.
  - repacked `uint32_t` vector loads: ~0.503 ms/call, ~66.1 GiB/s.
  - repacked 16-byte `uint4` loads: ~0.777 ms/call, ~42.8 GiB/s.
  - result: layout separation/alignment helps substantially; vectorizing by reducing active lanes still hurts. Best current repacked q_b kernel is still byte-per-lane with separate scales.
  - implication: load-time repack for hot Q8 dense tensors is a promising decode lever even before rocWMMA. Need integrate behind an opt-in cache/path and test quality/correctness.

- Q2_K first repack microbench (`/tmp/q2_repack_bench.cpp`, layer0 routed experts, selected experts 0..5):
  - raw device Q2 gate/up + down, decode-like one token: ~0.57 ms/token-equivalent, ~80-89 GiB/s logical active-weight bandwidth.
  - separated arrays repack (`qs`, subblock scales, superblock d/dmin): ~0.66-0.67 ms, ~69-70 GiB/s; slower.
  - raw optimized unpack that loads each 2-bit byte once for four shifts was essentially flat (~0.57 ms).
  - implication: unlike Q8 q_b, naive Q2_K component separation is not a cheap win. Q2_K bottleneck is more dequant/instruction/control and expert/tile reuse than 84-byte alignment alone. For prefill, continue with expert-batched/WMMA/tile reuse rather than simple Q2 layout split.
- Q2_K routing skew measured with `DS4_HIP_MOE_ROUTING_DUMP` on a varied 1024-token prefill chunk (`/tmp/ds4-routing2.log`):
  - first 3 layers are close to flat: all 256 experts active, p50≈20-21, p90≈41-42, max≈71-100 for 6144 token-expert assignments.
  - later layers are heavy-tailed, not flat: active experts avg≈214/256, p50 avg≈11.9, p90≈66, p95≈100, p99≈283, max avg≈497 and worst max=980 assignments to one expert.
  - avg bucket counts over 43 layers: `>=32` ≈54 experts/layer, `>=64` ≈22, `>=128` ≈7.
  - assignment/work share from counts is better than raw expert count:
    - all layers: `>=32` ≈74.2%, `>=64` ≈50.8%, `>=128` ≈30.4%; top22 experts ≈50.5%.
    - later 40 layers: `>=32` ≈76.5%, `>=64` ≈54.2%, `>=128` ≈32.7%; top22 experts ≈52.8%.
    - first 3 flat layers: `>=32` only ≈43.8%, but `>=16` ≈86.4%.
  - implication: use two MoE prefill paths. Big buckets need tile-reuse/WMMA/LDS dequant; small buckets need a low-overhead direct/scalar path. Threshold 32 captures most later-layer assignments; threshold 16 may be needed for flat early layers if tile overhead is low enough.
- Added GGUF-backed Q2_K MoE microbench/replay helpers:
  - `tools/hip_q2_moe_wmma_microbench.cpp` can now mmap the GGUF, list matching tensors with `--gguf-find`, and benchmark one real routed expert slice with `--gguf` without loading the whole model.
  - Real CyberNeurova Q2_K tensor geometry: gate/up `[4096,2048,256]`, down `[2048,4096,256]`, one expert slice ≈2.625 MiB, one full routed tensor ≈672 MiB.
  - Q8_K activation/int8-dot diagnostic on real GGUF is a loser (~0.22-0.25x current) and should not be integrated.
  - Direct single-M-tile WMMA dequant is a loser; full `Q2_K -> KxN half` repack gives faster GEMM only for large M but repack cost kills it.
  - New multi-M WMMA prototype dequants a Q2_K B tile once and reuses it across 4/8 M tiles. Real GGUF down speedups: M64≈1.8x, M128≈2.2x, M512≈2.5x. Real gate+up dual-matrix speedups: M64≈1.0x, M128≈1.6x, M256≈2.1x, M512≈2.2x.
  - Weighted by the measured bucket distribution and current gate/down time share, a thresholded hot-bucket WMMA path is estimated at only ≈1.28-1.33x MoE-stage speedup unless the M64-127 gate/up case improves. Useful, but not an 80 t/s solution by itself.
  - Production-shaped bucket microbench with scattered `buckets[]` shows stronger per-bucket down speedups (M32≈1.7-2.0x, M64≈2.1-2.4x, M128≈2.5x, M512≈2.8-3.0x) and gate+up speedups only from M64+ (M32≈1.0x, M64≈1.3-1.4x, M128≈1.5-1.8x, M512≈2.4x).
  - Briefly tried an opt-in production WMMA-hot path in `ds4_hip.cpp`, but real graph was slower (long prompt baseline ≈31.6 t/s, WMMA hot threshold64/MTILES4 ≈29.4 t/s, threshold128/MTILES8 ≈30.1 t/s). The production path was reverted immediately; keep the idea in microbench until scheduling overhead is solved.
  - Fresh stage profile on current fast path shows large remaining bottlenecks for 2048-token chunks: attention output projection ≈0.51-0.53s/layer, routed MoE ≈0.34-0.38s/layer, attention ≈0.18s early and ≈0.55s later, q_path ≈0.21s/layer.
  - Re-tested Q8 shared-X token tiling on the long 4180-token prompt. The best measured setting is now `DS4_HIP_Q8_BATCH_TILE=32` with `DS4_HIP_Q8_BATCH_SHARED_X_BLOCKS=16` and RPB32.
  - Extended the `DS4_HIP_Q8_BATCH_FAST` path to batched prefill matmuls with small output dimensions instead of only `out_dim >= 1024`. This moved the expensive `4096->512`, `4096->256`, `4096->64`, and `16384->24` prefill shapes onto the shared-X token-tiled kernel while preserving default behavior when `DS4_HIP_Q8_BATCH_FAST` is unset.
  - Q8 profile on the long 4180-token prompt improved from ≈44.7s (old tile8/large-output-only path) to ≈19.9-21.8s with tile32+small-output batching. The former `4096->512` path dropped from ≈49-56ms/call to ≈4-5ms/call and `4096->256` from ≈26-29ms/call to ≈2.5-2.8ms/call. Unprofiled prefill improved from ≈31-32 t/s to ≈35-38 t/s; first token stayed `We` on the test prompt. Promoted tile32 into `DS4_SERVER_FAST_FULL`.
  - Added a chunked shared-X attention-output low projection path for the grouped Q8_0 `attn_output_a` batch matmul. The old grouped tile path had to stage `tile * 4096` floats and was limited to tile4; the chunked path stages `tile * 16 * 32` floats, so tile16 fits in 32 KiB and reuses each output row over more tokens. `DS4_HIP_Q8_BATCH_FAST` now defaults grouped attention-output low to tile16. Layer-4 output projection on the 2048-token chunk dropped from ≈474ms to ≈156-176ms, and full long-prompt prefill improved to ≈48-50 t/s with stable first token `We`.
  - Promoted the winning Q8 batched prefill path to HIP default-on. `DS4_HIP_Q8_BATCH_FAST=0` disables it, while normal and grouped Q8 prefill defaults are tile32/shared-X/RPB32 and grouped tile16/chunked shared-X respectively.
  - Added a default-on HIP indexer qmix score path. The old indexer score did `sum_h,d q[t,h,d] * weight[t,h] * comp[c,d]` independently for every compressed row. The new path first collapses each token to `qmix[t,d] = sum_h q[t,h,d] * weight[t,h]`, then scores `qmix[t,*] dot comp[c,*]` with a tiled 16x16 kernel. On layer32/pos2048/tokens2048/comp1024, score time dropped from ≈190ms to ≈3.4ms; the full indexed-attention stage dropped from ≈536ms to ≈348ms. Long-prompt prefill improved from ≈49.6 t/s with qmix disabled to ≈51-52.5 t/s, first token `We`. `DS4_HIP_INDEXER_QMIX_FAST=0` disables it for regression checks.
  - Added `tools/parse_moe_profile.py` to summarize routing counts, stage profile logs, and assignment-weighted what-if speedups.
- Added debug-only MoE split/range instrumentation:
  - `DS4_HIP_MOE_EXPERT_SPLIT_THRESHOLD=N` runs current expert-batch kernels as `<N` and `>=N` ranges with labels, useful with `DS4_HIP_MOE_PROFILE=1`.
  - This is not a speed path yet; it launches extra range-filtered kernels and only measures/validates the two-path split point.
  - On a 1003-token CLI prefill with threshold 64 and profiling syncs: base MoE profile total ≈25.13s, split64 total ≈24.74s (≈noise/slightly faster); unprofiled CLI base ≈1.72 t/s, split64 ≈1.69 t/s, split32 ≈1.70 t/s.
  - Split64 per-range profile: gate `<64` ≈5.46s, gate `>=64` ≈8.59s; down `<64` ≈3.58s, down `>=64` ≈7.10s. Big buckets are ~63% of measured MoE time for this prompt, matching the routing work-share data well enough to prioritize big-bucket kernels.
- Q8 repack prefill diagnostic:
  - `DS4_HIP_Q8_REPACK=1 DS4_HIP_Q8_MATMUL_PROFILE=1` on a 1003-token fast prefill proves q_b repack is eager-built but decode-only in the current selector.
  - All prefill Q8 calls (`tokens=1003`) logged as raw `q8_matmul`, including q_b `in=1024 out=32768`: 43 calls, total ≈3237 ms, avg ≈75.3 ms/layer under profiling.
  - `q8_matmul_repack` appears only for decode `tokens=1`: 43 calls, total ≈14.0 ms, avg ≈0.324 ms/layer.
  - Unprofiled fast CLI A/B on same 1003-token prompt: no-repack ≈19.95 t/s, q8-repack ≈20.02 t/s; effectively no prefill gain.
  - Conclusion: the earlier q_b repack/microbench win did not translate to prefill because `ds4_metal_matmul_q8_0_tensor()` only checked repack when `n_tok == 1`; batched prefill takes `DS4_HIP_Q8_BATCH_FAST` raw-layout kernels.
- Added a one-command max-performance/full-copy server preset:
  - `DS4_SERVER_FAST_FULL=1 scripts/start_ds4_server.sh`
  - expands to the current best prefill flags: raw/mixed attention, Q8 batch shared-X with RPB32/block16, Q2 MoE expert-batch with RPB16 shared-X/shared-mid.
  - also enables current decode/full-memory knobs: device tensors, staged full model copy, Q8 decode repack, and split16 decode repack.
  - removed server exposure for older non-winners: Q8 splitK knob and Q2 tile-list knobs.
  - pruned internal Q2 tile-list kernels/launch paths and older env-only Q8 splitK/repack-splitK branches. Kept raw automatic split-K helpers where they are part of current decode/attention output path.
  - Q8 shared-X tail validation passed prompt-graph tests for odd/non-tile token counts 19, 23, 27, 35, 43, 51, 75, 83, 139, and 147; all matched CPU/GPU top token.
  - Tried a fused lm_head/Q8 top-1 decode path that skipped materializing logits; it was not faster than the existing `4096->129280` matmul/top path (~6.6 ms either way for the projection), so it was removed rather than leaving a non-winning experiment.
- Removed nonperformant Q8 batched-prefill repack experiments:
  - Deleted the `DS4_HIP_Q8_REPACK_BATCH` / `DS4_SERVER_Q8_REPACK_BATCH` path and token-tiled repacked kernels. It was slower than raw-layout Q8 batch prefill.
  - Deleted the `DS4_HIP_Q8_REPACK_SPLIT16_BATCH` / `DS4_SERVER_Q8_REPACK_SPLIT16_BATCH` path and split16 token-tiled kernels. It was flat/slower than raw-layout Q8 batch prefill.
  - Kept decode-only Q8 repack and split16 repack paths for now; only the batched prefill variants were removed.
- Tried and removed a q_b-specific 4-row Q8 batched-prefill kernel:
  - tested fixed `1024 -> 32768` q_b with 4 output rows/wave.
  - result: slower than both baseline 2-row and shared-X. Profile on 1003-token prompt: baseline q_b ≈75.4 ms/layer, shared-X q_b ≈65.9 ms/layer, 4-row q_b ≈100.5 ms/layer, shared-X+4row ≈91.4 ms/layer. Removed immediately to keep the backend lean.
- Added optional LDS shared-X dense Q8 batched-prefill kernel:
  - env: `DS4_HIP_Q8_BATCH_SHARED_X=1`; server knob: `DS4_SERVER_Q8_BATCH_SHARED_X=1`. Also `DS4_HIP_Q8_BATCH_RPB`/`DS4_SERVER_Q8_BATCH_RPB` and `DS4_HIP_Q8_BATCH_SHARED_X_BLOCKS=8|16|32`.
  - kernel/helper: `ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<TOK_TILE,BLOCKS_TILE>` and `ds4_hip_launch_q8_0_batch_sharedx<BLOCKS_TILE>`.
  - implementation loads a `TOK_TILE x BLOCKS_TILE x 32` activation chunk into LDS, then all row waves in the block reuse it. This keeps raw Q8 layout and avoids decode-style q/scales split.
  - correctness smoke: `DS4_HIP_Q8_BATCH_FAST=1 DS4_HIP_Q8_BATCH_SHARED_X=1 --metal-graph-prompt-test` kept same top.
  - 1003-token fast prefill with normal MoE: base ~20.00 t/s; shared-X RPB16/block16 ~21.65; RPB32/block16 ~22.26; block8/block32 were similar/slightly lower; tile4 bad; tile16 similar to tile8.
  - Q8 profile on 1003-token prompt: Q8 total ≈16.02s -> ≈11.09s. Big wins: `8192->4096` ≈108 ms/layer -> ≈55 ms/layer, `4096->2048` ≈26.6 -> ≈10.8, `4096->1024` ≈14.3 -> ≈6.1, `2048->4096` ≈21.3 -> ≈11.2. `1024->32768` only modestly improved ≈78.8 -> ≈72.3.
  - Combined with Q2 LDS MoE RPB16: 1003-token prefill ≈28.35 -> ≈31.76 t/s; 1905-token prompt ≈30.68 t/s with both. Keep opt-in pending server stability.
- Added optional tile-list MoE scheduling:
  - env: `DS4_HIP_MOE_EXPERT_TILE_LIST=1`; server knob: `DS4_SERVER_MOE_EXPERT_TILE_LIST=1`.
  - builds a compact `(expert,p0)` tile list after bucketization and launches row kernels over tile index rather than expert index, parallelizing huge hot buckets instead of looping all p0 tiles inside one row block.
  - supports `DS4_HIP_MOE_EXPERT_TILE=4|8|16`; tile 8 remains best. Tile 16 and split big=16 were slower in unprofiled CLI tests.
  - correctness: `DS4_HIP_MOE_EXPERT_TILE_LIST=1 --metal-graph-full-test` passes with same representative diff/top as baseline.
  - profile on 1003-token prompt: base MoE total ≈25.12s vs tile-list8 ≈22.73s (~9.5% MoE-kernel reduction: gate 14.53s→14.16s, down 10.59s→8.56s).
  - fast-prefill CLI A/B with raw/mixed/Q8/MoE fast flags: base ≈19.0–19.4 tok/s, tile-list8 ≈19.9–20.2 tok/s. Keep opt-in until longer prompt/server A/B.
  - added hybrid big-bucket mode: `DS4_HIP_MOE_EXPERT_TILE_LIST_THRESHOLD=N` / `DS4_SERVER_MOE_EXPERT_TILE_LIST_THRESHOLD=N`; current expert-batch handles buckets `<N`, tile-list handles buckets `>=N`.
  - 1003-token fast-prefill A/B after hybrid: base 19.99 t/s, all tile-list 20.98 t/s, hybrid32 20.99 t/s, hybrid64 20.84 t/s.
  - 1905-token fast-prefill A/B: base 19.73 t/s, hybrid32 20.08 t/s. Gain is real but modest/variable; threshold 32 is still the better default candidate if we enable this later.
  - tried down-only tile-list mode (`DS4_HIP_MOE_EXPERT_TILE_LIST_DOWN_ONLY=1`) to keep baseline gate/up and only parallelize down. 1003-token prompt: base 19.99, down-all 20.83, down32 20.72, down64 20.62, full hybrid32 20.89. 1905-token prompt was noisy/negative: base 19.70, down-all 19.55, hybrid32 19.53. Conclusion: tile-list/down-only remains opt-in; no default change until server/longer-prompt repeat says otherwise.
- Added optional LDS activation tile reuse in Q2_K expert-batch MoE:
  - env: `DS4_HIP_MOE_EXPERT_SHARED_X=1` for gate/up and `DS4_HIP_MOE_EXPERT_SHARED_MID=1` for down, intended with `DS4_HIP_MOE_GATE_RPB=4 DS4_HIP_MOE_DOWN_RPB=4`.
  - server knobs: `DS4_SERVER_MOE_GATE_RPB`, `DS4_SERVER_MOE_DOWN_RPB`, `DS4_SERVER_MOE_EXPERT_SHARED_X`, `DS4_SERVER_MOE_EXPERT_SHARED_MID`.
  - kernels: `ds4_hip_moe_q2_gate_up_expert_batch_sharedx_kernel<PAIR_TILE>` and `ds4_hip_moe_q2_down_expert_batch_sharedmid_kernel<PAIR_TILE>` cache the current `PAIR_TILE x 256` activation/mid tile in LDS so multiple row waves in a block reuse it.
  - correctness: prompt graph and full first-token graph passed with `GATE_RPB=4 DOWN_RPB=4 SHARED_X=1 SHARED_MID=1`.
  - 1003-token fast prefill: base ≈19.99 t/s; `GATE_RPB=4 DOWN_RPB=4` without LDS ≈20.0 t/s/noise; shared-x only ≈21.0; shared-mid only ≈20.5; shared both RPB4 ≈22.9–24.7 t/s depending on profile/noise.
  - RPB sweep with shared both: RPB2 was bad/noisy (~17.4), RPB4 ~23, RPB8 ~25.7, RPB16 ~27–29, RPB32 slightly worse (~28.1). Tile 8 remains best vs tile 4/16 for RPB8.
  - MoE-profiled 1003-token run: base gate/up expert ≈14.13s and down expert ≈10.35s; RPB16 shared gate/up ≈5.01s and down ≈3.59s. Overall profiled prefill ≈19.4 -> ≈27.3 t/s.
  - 1905-token fast prefill: base ≈18.88 t/s, RPB16 shared ≈26.54 t/s.
  - Added shared tile-list variants (`ds4_hip_moe_q2_gate_up_tile_list_sharedx_kernel`, `ds4_hip_moe_q2_down_tile_list_sharedmid_kernel`) so tile-list can combine with LDS. Correctness passed, but performance was flat/slightly worse than pure shared expert-batch: 1003-token shared base 24.54 vs tile-all 24.43/hybrid32 24.47/down-only32 24.33; 1905 shared base 24.15 vs tile-all 24.02. Scheduling no longer helps much once LDS row-block reuse is active.
  - Keep opt-in pending longer server stability, but this is the first substantial Q2_K prefill win beyond scheduling. Best current candidate: `DS4_HIP_MOE_GATE_RPB=16 DS4_HIP_MOE_DOWN_RPB=16 DS4_HIP_MOE_EXPERT_SHARED_X=1 DS4_HIP_MOE_EXPERT_SHARED_MID=1`.

## Loader follow-ups

Keep full-copy opt-in. It consumes ~92 GiB and leaves limited headroom, so zero-copy remains the safe default.

- [ ] Add explicit full-copy loader mode selection:
  - `DS4_HIP_COPY_MODEL_MODE=auto|staged|host_direct|mmap_memcpy`
  - default under `DS4_HIP_COPY_MODEL=1`: `auto`
  - keep old direct mmap->`hipMemcpyAsync` path only as a debug fallback, not default.
- [ ] Probe the actual large model allocation after `hipMalloc(map_size)`, not only a tiny test allocation:
  - call `hipPointerGetAttributes(&attr, device_start)`
  - if `attr.hostPointer != NULL`, allow CPU-direct `pread()` into `attr.hostPointer`
  - kernels should use `attr.devicePointer`/the HIP device pointer, not the host pointer
  - log: allocation type, `isManaged`, `hostPointer`, `devicePointer`, selected loader mode.
- [ ] Keep staged pinned copy as fallback and as the default on the current ROCm stack:
  - current observed `hipMalloc` attrs: device memory, `hostPointer=NULL`
  - CPU writes to the `hipMalloc` pointer are not valid here.
- [ ] Replace two independent pinned staging allocations with one pinned slab:
  - `hipHostMalloc(pool, nbuf * chunk_bytes, hipHostMallocDefault)`
  - slice into `stage[i] = pool + i * chunk_bytes`
  - env: `DS4_HIP_COPY_MODEL_BUFFERS=2|3`, default `2` or `3` after A/B.
- [ ] A/B chunk sizes:
  - `DS4_HIP_COPY_MODEL_CHUNK_MB=64,128,256,512`
  - record copy time, average GiB/s, peak RSS, and post-load HIP free memory.
- [ ] Keep `posix_fadvise(fd, copied_range, POSIX_FADV_DONTNEED)` after chunks.
- [ ] Treat `madvise(MADV_DONTNEED)` as optional/lower priority; the staged path uses `pread`, so fd-level fadvise is the important page-cache hint.
- [ ] Add a short full-copy smoke test script that stops the server, runs `--inspect`, and confirms:
  - allocation succeeds
  - chosen loader mode is logged
  - GPU probe succeeds
  - server can restart in zero-copy mode afterward.

## Q8_0 layout/repack plan

This is now the highest-priority optimization. The q_b repack experiment showed a 1.43x decode-kernel speedup from layout alone, without WMMA or quantization changes.

- [x] Add an opt-in repacked Q8_0 tensor cache:
  - env: `DS4_HIP_Q8_REPACK=1`
  - keep default off until full validation passes
  - cache entries keyed by `(model_map, weight_offset, in_dim, out_dim)`.
  - current default repack window is 32-40 MiB, intentionally targeting `attn_q_b`; override with `DS4_HIP_Q8_REPACK_MIN_MB` / `DS4_HIP_Q8_REPACK_MAX_MB` for experiments.
- [x] Repack format:
  - `int8 weights[out_dim][in_dim]`, contiguous/aligned
  - `uint16_t scales[out_dim][in_dim/32]`, contiguous fp16 scale bits
  - preserve logical Q8_0 math exactly.
- [x] Add repacked Q8 matmul kernel:
  - same byte-per-lane/wave-row structure as current kernel
  - two pointers: `q` and `scales`
  - no vec2/vec4 path initially; vectorized loads regressed in microbench.
- [x] Apply first to hot decode tensors:
  - `attn_q_b`
  - eager GGUF scan/repack of q_b tensors at model registration when `DS4_HIP_Q8_REPACK=1`
  - server wrapper knob: `DS4_SERVER_Q8_REPACK=1`
- [x] Add a split-K-aware Q8_0 repack experiment for output/shared-down:
  - env: `DS4_HIP_Q8_REPACK_SPLIT16=1`
  - server wrapper knob: `DS4_SERVER_Q8_REPACK_SPLIT16=1`
  - layout is split-major 16-block records: `[split][row]` records of 544 bytes = 16 fp16 scales (32B) + 16x32 int8 weights (512B), aligned.
  - eager GGUF scan/repack covers `attn_output_a`, `attn_output_b`, and `ffn_down_shexp` when enabled.
- [ ] Apply next only after stronger measured win:
  - `output.weight` / lm_head if memory permits.
- [ ] Then apply to remaining dense/shared Q8 tensors:
  - `attn_q_a`, `attn_kv`
  - shared expert Q8 tensors
  - small router/indexer Q8 tensors only if profiling says useful.
- [ ] Validation:
  - synthetic Q8 correctness vs raw Q8 kernel
  - graph prompt/full tests
  - decode stage profile before/after
  - prefill stage profile before/after.
- Current integrated q_b-only observations:
  - eager q_b repack: 43 tensors, ~1.43 GiB device memory, ~0.32-0.33 s during model registration.
  - q_b decode kernel profile, warmed: raw ~0.440 ms/layer, repacked ~0.358-0.364 ms/layer.
  - short CLI `-n 32` without profiling: raw ~8.88 tok/s, q_b eager repack ~9.15 tok/s; modest because only q_b is enabled and output/lm_head still dominate.
  - generic repack for small Q8 shapes (`4096->1024`, `4096->512`, `4096->256`) was slower, so default min size excludes them.
  - experimental separate-scale split-K output/shared paths are gated behind `DS4_HIP_Q8_REPACK_SPLITK=1`; initial profiles were slower (`attn_out_low`/`attn_out_b_hc_q8` ~1.2-1.3 ms vs raw ~0.38 ms), so keep off by default.
- Split-major output/shared observations:
  - standalone `/tmp/q8_split_repack_bench.cpp` on all 43 tensors:
    - `attn_output_a` raw-device split16+sum: ~0.205 ms/call, split-major: ~0.170 ms/call (~195 GiB/s logical).
    - `attn_output_b` raw-device split16+sum: ~0.209 ms/call, split-major: ~0.175 ms/call (~190 GiB/s logical).
  - integrated eager repack with `DS4_HIP_Q8_REPACK=1 DS4_HIP_Q8_REPACK_SPLIT16=1`:
    - q_b=43 tensors, ~1.43 GiB; split16=129 tensors, ~3.21 GiB; total extra ~4.64 GiB.
    - eager repack time ~0.9-1.7 s in CLI runs.
    - warmed profiles: q_b ~0.34-0.35 ms/layer, `attn_out_low` ~0.37 ms, `attn_out_b_hc_q8` ~0.36-0.37 ms, shared down ~0.10 ms.
    - CLI `-n 64`: q_b+split16 eager ~9.4 tok/s vs q_b-only ~9.2 tok/s and raw ~8.9 tok/s on short prompt; useful but not the expected large decode jump.
  - lm_head/output generic repack test (`4096->129280`) did not help: raw ~6.5 ms, repacked ~6.5-6.9 ms, so leave excluded.
- [ ] Expected impact:
  - decode: ~10 tok/s -> ~12-13 tok/s if dense Q8 stages see similar gains
  - prefill: ~20 tok/s -> ~24-26 tok/s before Q2_K/WMMA work.

## rocWMMA / dense MLA plan

### Objective

Use rocWMMA/WMMA on gfx1151 after the Q8_0 repack path. Repack is the cheaper first win for decode; WMMA remains important for prefill and later dense/expert kernels.

The GGUF uses MLA-style factorized attention:

- `attn_q_a`: `[4096, 1024]` Q8_0, ~4.25 MiB/layer
- `attn_q_b`: `[1024, 32768]` Q8_0, ~34 MiB/layer
- `attn_kv`: `[4096, 512]` Q8_0, ~2.125 MiB/layer
- `attn_output_a`: `[4096, 8192]` Q8_0, ~34 MiB/layer
- `attn_output_b`: `[8192, 4096]` Q8_0, ~34 MiB/layer

These dense Q8_0 matrices are touched every decode token and are the largest decode target after correcting the bandwidth model. For prefill, the measured split is more balanced: routed Q2_K MoE is currently the largest stage (~48%), while dense/shared Q8 projection work is ~41%. The first Q2_K down-projection WMMA microbench changed the priority: direct Q2_K dequant-to-LDS WMMA is slower, while pre-repacked half `KxN` is only about a 1.25x microbench win. Dense Q8 WMMA on output projection and q-path is now Phase 1 because it is simpler, validates WMMA on cleaner kernels, has a higher expected speedup, and should save more absolute time in prefill.

Realistic prefill trajectory:

- current HIP fast path: ~32 tok/s
- dense Q8 WMMA on output projection and q-path: ~36-40 tok/s
- Q2_K MoE WMMA after a validated repack design: ~38-42 tok/s
- further polish, fusion, and attention work: ~42-50 tok/s

### Phase 0: rocWMMA capability/smoke

- [x] Add a tiny standalone rocWMMA smoke source, initially outside the main engine:
  - source: `tools/hip_rocwmma_smoke.cpp`
  - build: `make hip-rocwmma-smoke`
  - includes `<rocwmma/rocwmma.hpp>` and `<rocwmma/rocwmma-version.hpp>`
  - verified on AMD Radeon 8060S: `gfx1151`, wavefront size 32, rocWMMA 2.1.0.
- [x] Build one FP16 GEMM kernel using rocWMMA fragments:
  - 16x16x16 fragments, FP16 inputs, FP32 accumulation
  - compared against CPU reference on small shape and simple HIP reference on large shape.
- [x] Confirm generated ISA contains WMMA instructions:
  - extracted code object with `llvm-objdump --offloading hip-rocwmma-smoke`
  - confirmed `v_wmma_f32_16x16x16_f16` in gfx1151 disassembly.
- [x] Baseline smoke timings:
  - `128x128x128`: rocWMMA ~0.013 ms / ~0.32 TFLOP/s, max abs vs CPU ~1.1e-5
  - `1024x1024x1024`: rocWMMA ~0.44 ms / ~4.9 TFLOP/s, max abs vs HIP naive ~2.1e-4
  - This is only a one-wave-per-16x16-tile smoke kernel, not a tuned GEMM.

### Phase 1: dense Q8 WMMA first target

Priority revision after the Q2_K down microbench: dense Q8 WMMA on output projection should happen before Q2_K MoE integration.

Reasons:

- higher expected first-step speedup: about 1.5-2x on the targeted dense kernels vs about 1.25x for the pre-repacked Q2_K down microbench
- simpler integration: no expert dispatch, no bucket thresholds, simpler Q8_0 dequant/repack
- cleanest validation target for WMMA infrastructure
- larger expected absolute prefill saving: roughly 4.5s vs 1.7s on a 31s prefill
- lets Q2_K get another microbench/repack pass before engine integration

Immediate targets:

- [x] Build dense Q8 WMMA microbench for output projection first:
  - source: `tools/hip_q8_wmma_microbench.cpp`
  - build: `make hip-q8-wmma-bench`
  - compares current Q8 shared-X batched prefill path against direct Q8 dequant WMMA, packed half WMMA, and multi-N packed half WMMA.
- [x] Test output projection and q-path at `M=128` tokens:
  - `attn_output_a`-like `4096 -> 8192`: current ~7.72 ms, packed multi-N WMMA ~3.13 ms, ~2.46x
  - `attn_output_b`-like `8192 -> 4096`: current ~7.37 ms, packed multi-N WMMA ~2.81 ms, ~2.63x
  - `attn_q_b`-like `1024 -> 32768`: current ~9.35 ms, packed multi-N WMMA ~4.00 ms, ~2.34x
  - `attn_q_a`-like `4096 -> 1024`: current ~0.75 ms, packed multi-N WMMA ~0.35 ms, ~2.16x
  - rel RMS vs current FP32 path is ~2.9e-4.
- [x] Integrate packed multi-N WMMA for prefill behind `DS4_HIP_Q8_WMMA_FAST=1`:
  - engine-side eager Q8 FP16 `KxN` repack for Q-side tensors: `attn_q_a`, `attn_q_b`, and indexer `attn_q_b`
  - production path uses FP32-accumulate WMMA ISA (`v_wmma_f32_16x16x16_f16`) and a two-pass activation split to reduce FP16 input rounding drift
  - current shared-X path remains fallback for decode, unsupported shapes, small batches (`DS4_HIP_Q8_WMMA_MIN_TOKENS`, default 64), and when the flag is off.
- [x] Replace slow host repack with GPU repack kernel:
  - q-side zero-copy Q8 WMMA repack now builds 107 tensors / 3.35 GiB in about 1.7-2.6 s
  - full output-projection WMMA repack was pruned from the production path because it caused worse greedy drift with little useful graph speedup.
- [x] Add chunk-level prefill instrumentation:
  - `DS4_HIP_PREFILL_CHUNK_PROFILE=1` / `DS4_METAL_GRAPH_PREFILL_CHUNK_PROFILE=1`
  - optional stage filters: `DS4_METAL_LAYER_STAGE_PROFILE_POS`, `DS4_METAL_LAYER_STAGE_PROFILE_LAYER`, `DS4_METAL_Q_STAGE_PROFILE_POS`, `DS4_METAL_Q_STAGE_PROFILE_LAYER`.
- [x] Fix long-prompt chunk slowdown unrelated to Q8 WMMA:
  - old indexer top-k path used one thread per token and took ~5.1 s per ratio-4 layer at `pos=2048`, `n_comp=768`, `top_k=512`
  - new parallel iterative top-k path reduces this to ~4-5 ms per layer and now covers `n_comp<=8192`, `top_k<=1024` so 32k contexts do not fall back to the serial path
  - 4180-token prefill recovered from ~8 t/s to ~29 t/s without WMMA and ~31 t/s with WMMA.
- [ ] Next dense Q8 step:
  - Q8 WMMA remains experimental opt-in, not part of `DS4_SERVER_FAST_FULL`, because greedy continuations still diverge on close logit margins even with q-side/xsplit gating
  - quantify stochastic sampling variance vs WMMA drift at nonzero temperature before deciding whether the opt-in path is acceptable for non-greedy serving
  - graph/profile per-stage savings with `DS4_HIP_Q8_MATMUL_PROFILE=1`
  - tune packed multi-N tile count/VGPR occupancy if profiling shows headroom
  - validate longer generations and server stability.
- [ ] Keep one small profile loop around current decode/prefill to measure real graph savings:
  - dense MLA/projection Q8_0 matmuls
  - routed Q2_K MoE kernels
  - raw/mixed attention fast paths
  - output/lm_head and launch overhead.

### Phase 2: Q8_0 WMMA microbench before integration

- [x] Create a standalone microbench for DS4-like shapes:
  - A = activations `[tokens, K]`
  - B = weights `[K, out_dim]`, derived from GGUF Q8_0 row layout
  - C = output `[tokens, out_dim]`
- [ ] Test dense MLA shape families first, with output projection as the first performance target:
  - `4096 -> 8192` (`attn_output_a`, grouped by 8 output groups)
  - `8192 -> 4096` (`attn_output_b`)
  - `1024 -> 32768` (`attn_q_b`, large Q up-projection)
  - `4096 -> 1024` (`attn_q_a`)
  - `4096 -> 512` (`attn_kv`)
  - decode token tile: 1
  - prefill token tiles: 8, 16, 32, 64, 128, 512.
- [ ] Then test shared expert Q8 shapes:
  - `4096 -> 2048`
  - `2048 -> 4096`.
- [ ] Compare against current kernels with:
  - `DS4_HIP_Q8_BATCH_FAST=1`
  - tile 8/16
  - 2-row path enabled/disabled.
- [ ] Track:
  - max/rms error vs current FP32 path
  - elapsed ms
  - effective TFLOP/s
  - effective GB/s
  - decode tok/s and prefill tok/s when integrated.

### Phase 3: Q8_0 WMMA kernel

- [ ] Implement an opt-in Q8_0 WMMA path:
  - env: `DS4_HIP_Q8_WMMA_FAST=1`
  - call site: `ds4_metal_matmul_q8_0_tensor(...)`
  - support decode `n_tok == 1` for dense MLA projections and prefill `n_tok >= 8`
  - fall back to current kernels for unsafe dimensions/tails.
- [ ] Data plan:
  - convert activation tile `float -> half` in a small pre-kernel or inside the WMMA kernel
  - dequantize Q8_0 weight K/N tile into LDS as FP16
  - rocWMMA multiply FP16 x FP16 with FP32 accumulation
  - store FP32 outputs to preserve downstream behavior.
- [ ] Tiling sketch:
  - C tile: 16 tokens x 16 output rows initially for prefill
  - decode path may need a different N-tiled/vector shape because M=1 underuses WMMA
  - K tile: 16 or 32, depending on rocWMMA fragment constraints and LDS pressure
  - one or multiple waves per block; start simple, then try cooperative schedulers.
- [ ] Keep correctness gates strict:
  - synthetic Q8 tests
  - prompt/full graph tests
  - logit/top-token comparison on 774-token prompt
  - tolerate small FP16 accumulation differences only if output quality/top tokens remain stable.

### Phase 4: MLA projection fusion and load-time layout/repack

WMMA wants tile-friendly contiguous B tiles. GGUF row layout is not necessarily ideal. The first fusion targets are the MLA projection chains, not a single giant all-in-one attention kernel.

- [ ] First fusion target: `attn_output_a -> attn_output_b` chain.
  - this is currently the largest dense prefill stage
  - keep the 8192-wide intermediate tiled and local where practical.
- [ ] Second fusion target: `q_a -> q_b` projection chain.
  - avoid round-tripping the 1024-wide latent and/or 32768-wide Q tensor through global memory where practical
  - keep intermediate tiles in registers/LDS.
- [ ] Keep attention softmax/cache logic as a separate kernel initially; full MLA+attention fusion can come later if profiling justifies it.
- [ ] Design an on-device/repacked Q8 layout for WMMA:
  - tile-major by output rows and K blocks
  - packed scales adjacent to quant data or pre-expanded to FP16 scale vectors
  - alignment suitable for coalesced LDS fills.
- [ ] Add per-tensor optional repack metadata without changing the public `ds4_metal.h` API.
- [ ] Repack only hot Q8 tensors first:
  - `attn_q_b`, `attn_output_a`, `attn_output_b`
  - then `attn_q_a`, `attn_kv`
  - shared expert Q8 tensors
  - `e_proj`, `h_proj` if they are measured hot.
- [ ] Do not repack giant/rare tensors until profiling proves value.

### Phase 5: dense requant experiments

Do not jump straight from Q8_0 to Q4_K for attention. Attention, especially `attn_q_b`, is sensitive.

- [ ] Add a cheap quality experiment for Q6_K-style dense requant before Q4_K kernel work.
  - Note: the engine currently validates dense attention tensors as Q8_0, so Q6_K requires parser/type support and reference kernels or an external eval path first.
- [ ] Suggested mixed-precision policy to test:
  - keep Q8_0: `attn_q_b`, `lm_head/output.weight`
  - Q6_K candidates: `attn_output_a`, `attn_output_b`, shared expert Q8 tensors
  - Q4_K candidates only after Q6_K passes: `attn_q_a`, `attn_kv` and other smaller/tolerant tensors.
- [ ] Validate with graph tests, prompt logits/top-token checks, and longer qualitative prompts; quick max-diff tests are not enough for attention quantization.

### Phase 6: Q2_K MoE WMMA

Harder and still important, but lower priority for single-stream decode than dense MLA. Likely bigger payoff for prefill and batched workloads.

- [x] Start with routed down projection microbench:
  - source: `tools/hip_q2_moe_wmma_microbench.cpp`
  - build: `make hip-q2-moe-wmma-bench`
  - compares current-style Q2_K expert down against direct WMMA and pre-repacked half-KxN WMMA.
- [x] Implement Q2_K tile dequant into LDS FP16 in the microbench:
  - B tile = expert weight tile
  - A tile = selected token activations converted from float to FP16
  - FP32 accumulation
- [x] Initial result on synthetic `M pairs x K mid x N out = 64 x 2048 x 4096`:
  - direct Q2_K dequant-to-LDS WMMA: ~0.94 ms, slower than current-like ~0.71 ms
  - pre-repacked `KxN` half WMMA: ~0.57 ms, about 1.25x faster than current-like
  - error vs current-like FP32 path: rel RMS ~3e-4
  - conclusion: do not integrate direct dequant WMMA; Q2_K WMMA needs load-time/tile-major repack or a better cooperative dequant layout.
- [ ] Reuse existing expert-bucketed prefill path:
  - only run WMMA for experts with enough `(token, expert)` pairs
  - keep current scalar/rowwise kernels for small buckets and decode.
- [ ] Consider fused gate/up later:
  - compute gate and up for the same expert bucket
  - apply SiLU/gating
  - feed down projection.
- [ ] Env gates:
  - `DS4_HIP_MOE_WMMA_FAST=1`
  - `DS4_HIP_MOE_WMMA_MIN_PAIRS=N`.

### Phase 7: profiling and success criteria

- [ ] Profile with rocprof/rocprofv3 on one decode step and one prefill chunk:
  - current dense Q8 MLA/projection kernels
  - current Q8 batch baseline
  - Q8 WMMA microbench
  - integrated Q8 WMMA path
  - MLA projection fusion kernels
  - MoE expert batch baseline
  - future Q2_K WMMA path.
- [ ] Classify bottleneck per kernel:
  - VALU/WMMA utilization
  - memory bandwidth
  - LDS bank conflicts
  - occupancy/register pressure
  - zero-copy vs full-copy behavior.
- [ ] Success criteria before enabling in server wrapper:
  - `make ds4`
  - `make ds4-server`
  - `make test`
  - graph prompt/full tests
  - synthetic Q8/Q2 correctness
  - no server stability regressions
  - measurable decode and/or prefill improvement over current baselines.

## Near-term execution order

The backend now has a stable fast path and the known non-winning experiments
have been removed. The next roadmap is:

1. rocWMMA on dense Q8_0 projections.
   - Production opt-in target is now q-side only: `attn_q_a`, `attn_q_b`, and indexer q.
   - Output projection WMMA was pruned from production because greedy drift was worse and graph speedup was not compelling.
   - Use tile-major/repacked layouts instead of raw GGUF rows.
   - Expected prefill move is modest for q-side only; larger gains likely require Q2_K/expert work.
2. rocWMMA on Q2_K routed experts.
   - First down-projection microbench is in place.
   - Direct dequant-to-LDS WMMA is slower; pre-repacked half `KxN` WMMA is faster.
   - Next: refine a load-time/tile-major Q2_K expert repack path before engine integration.
   - Expected after dense Q8: ~38-42 tok/s if Q2_K repack/WMMA lands.
3. hipBLASLt-quality custom kernels.
   - Focus on occupancy, LDS/coalescing, register pressure, tile scheduling,
     and real graph performance rather than just adding WMMA instructions.
   - Keep opt-in fallback gates until kernels beat the current fast path.
4. Decode/inference optimization.
   - Re-profile after the WMMA/repack work.
   - Optimize single-stream decode and server inference: dense projection
     fusion, output/head behavior, speculative/top-only paths where useful, and
     lower memory traffic per generated token.
5. K/V cache work.
   - After compute kernels, improve live KV layout, disk KV reuse, long-context
     cache movement, and server/cache ergonomics.
