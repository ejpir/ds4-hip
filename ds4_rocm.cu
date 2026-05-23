#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

#include "rocm/ds4_rocm_q8.cuh"

__global__ static void dequant_q8_0_to_f16_transpose_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const unsigned char *blk = w + (row * blocks + b) * 34u;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[i * out_dim + row] = __hmul(scale, __float2half((float)q));
}

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

extern "C" int ds4_gpu_embed_token_hc_q8_0_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!out_hc || !model_map || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t blocks = (n_embd + 31u) / 32u;
    const uint64_t weight_bytes = (uint64_t)n_vocab * blocks * 34u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        out_hc->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd_q8_0");
    if (!wptr) return 0;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    embed_token_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const unsigned char *)wptr,
            token,
            n_embd,
            n_hc);
    return cuda_ok(cudaGetLastError(), "embed token q8_0 launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t blocks = (n_embd + 31u) / 32u;
    const uint64_t weight_bytes = (uint64_t)n_vocab * blocks * 34u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd_q8_0");
    if (!wptr) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const int32_t *)tokens_t->ptr,
            (const unsigned char *)wptr,
            n_vocab,
            n_tokens,
            n_embd,
            n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens q8_0 launch");
}

#include "rocm/ds4_rocm_matmul.cuh"

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    if (getenv("DS4_CUDA_FP8_KV_SERIAL_GROUPS") != NULL) {
        fp8_kv_quantize_serial_groups_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    } else {
        const uint32_t groups = (n_nope + 63u) / 64u;
        fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    }
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#ifdef __HIP_PLATFORM_AMD__
static __half *moe_dense_weight_f16_cached(
        const char *base,
        uint32_t expert,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t expert_bytes,
        uint64_t row_bytes,
        __half *scratch_w,
        const char *label,
        int *ok) {
    const uint64_t elems = (uint64_t)out_dim * in_dim;
    const uint64_t bytes = elems * sizeof(__half);
    const uint32_t cache_limit_mb = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DENSE_HOT_CACHE_MB",
                                                              "DS4_HIP_MOE_DENSE_HOT_CACHE_MB",
                                                              6144u, 0u, 1048576u);
    const uint64_t cache_limit_bytes = (uint64_t)cache_limit_mb * 1024ull * 1024ull;
    const int dense_no_cache = cuda_env_flag_any3("DS4_CUDA_MOE_DENSE_HOT_NO_CACHE",
                                                  "DS4_HIP_MOE_DENSE_HOT_NO_CACHE", NULL);
    if (!dense_no_cache) {
        for (size_t ci = 0; ci < g_moe_dense_hot_cache.size(); ci++) {
            cuda_moe_dense_hot_cache_entry &ce = g_moe_dense_hot_cache[ci];
            if (ce.base == base && ce.expert == expert && ce.in_dim == in_dim &&
                ce.out_dim == out_dim && ce.expert_bytes == expert_bytes && ce.row_bytes == row_bytes) {
                return ce.ptr;
            }
        }
        if (cache_limit_bytes == 0 || g_moe_dense_hot_cache_bytes + bytes <= cache_limit_bytes) {
            __half *cached = NULL;
            if (cudaMalloc((void **)&cached, bytes) == cudaSuccess && cached != NULL) {
                moe_q2K_dequant_expert_f16_kernel<<<(elems + 255u) / 256u, 256>>>(
                        cached, base, expert, in_dim, out_dim, expert_bytes, row_bytes);
                if (!cuda_ok(cudaGetLastError(), label)) {
                    (void)cudaFree(cached);
                    if (ok) *ok = 0;
                    return scratch_w;
                }
                cuda_moe_dense_hot_cache_entry ce;
                ce.base = base;
                ce.expert = expert;
                ce.in_dim = in_dim;
                ce.out_dim = out_dim;
                ce.expert_bytes = expert_bytes;
                ce.row_bytes = row_bytes;
                ce.ptr = cached;
                ce.bytes = bytes;
                g_moe_dense_hot_cache.push_back(ce);
                g_moe_dense_hot_cache_bytes += bytes;
                return cached;
            }
        }
    }
    moe_q2K_dequant_expert_f16_kernel<<<(elems + 255u) / 256u, 256>>>(
            scratch_w, base, expert, in_dim, out_dim, expert_bytes, row_bytes);
    if (!cuda_ok(cudaGetLastError(), label) && ok) *ok = 0;
    return scratch_w;
}
#endif

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    const int iq2_path = (gate_type == 16u && down_type == 10u);
    const int q2k_path = (gate_type == 10u && down_type == 10u);
    if (!q4k_path && !iq2_path && !q2k_path) return 0;
    if (q4k_path && (n_tokens != 1u || n_expert != 6u)) return 0;
    const uint64_t gate_bytes = 256ull * gate_expert_bytes;
    const uint64_t down_bytes = 256ull * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u;
        const uint32_t use_expert_tiles = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL;
        const uint32_t expert_tile_m = getenv("DS4_CUDA_MOE_TILE4") ? 4u : 8u;
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = use_expert_tiles &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL));
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_down_tile16 = use_atomic_down && expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u : 1024u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_atomic_down && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
            const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + 256u;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + 256u) : 0u;
            const uint64_t tile_offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? 257ull * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile_experts, tile_starts, tile_offsets, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (use_decode_lut_gate && q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (q2k_path && n_expert == 6u &&
        n_tokens >= cuda_parse_u32_env_alias("DS4_CUDA_MOE_EXPERT_MIN_TOKENS",
                                             "DS4_HIP_MOE_EXPERT_MIN_TOKENS",
                                             32u, 1u, 65535u) &&
        getenv("DS4_CUDA_NO_MOE_Q2_EXPERT_BATCH") == NULL &&
        !cuda_runtime_config()->graph_dump) {
        const uint32_t pair_count = n_tokens * n_expert;
        const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
        const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
        const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
        const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
        const uint64_t hot_gate_bytes = 256ull * sizeof(uint32_t);
        const uint64_t hot_down_bytes = 256ull * sizeof(uint32_t);
        const uint64_t med_gate_bytes = 256ull * sizeof(uint32_t);
        const uint64_t med_down_bytes = 256ull * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = 256ull * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = 256ull * sizeof(uint32_t);
        const uint32_t wmma_tile_capacity = (pair_count + 127u) / 128u + 256u;
        const uint64_t wmma_tile_bytes = (uint64_t)wmma_tile_capacity * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = (getenv("DS4_CUDA_MOE_WMMA_HOT") != NULL ||
                                  getenv("DS4_HIP_MOE_WMMA_HOT") != NULL) &&
                                 expert_in_dim % 16u == 0u && expert_mid_dim % 16u == 0u && out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
#ifdef __HIP_PLATFORM_AMD__
        const int moe_dense_hot = moe_wmma_hot &&
            cuda_env_flag_any3("DS4_CUDA_MOE_DENSE_HOT", "DS4_HIP_MOE_DENSE_HOT", NULL) &&
            g_hipblaslt_ready && expert_in_dim == 2048u && expert_mid_dim == 2048u && out_dim == 4096u;
        const int moe_wmma_f16_down_req =
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_F16_DOWN", "DS4_HIP_MOE_WMMA_F16_DOWN", NULL);
        const int moe_wmma_f16_down_all_req =
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_F16_DOWN_ALL", "DS4_HIP_MOE_WMMA_F16_DOWN_ALL", NULL);
        const int moe_wmma_direct_sum_req =
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_DIRECT_SUM", "DS4_HIP_MOE_WMMA_DIRECT_SUM", NULL);
        const int moe_wmma_f16_mid_all_req =
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_F16_MID_ALL", "DS4_HIP_MOE_WMMA_F16_MID_ALL", NULL);
        const int moe_wmma_f16_mid = moe_wmma_hot && !moe_dense_hot &&
            (cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_F16_MID", "DS4_HIP_MOE_WMMA_F16_MID", NULL) ||
             moe_wmma_f16_mid_all_req || moe_wmma_f16_down_req || moe_wmma_f16_down_all_req || moe_wmma_direct_sum_req) &&
            n_expert == 6u && expert_in_dim % 16u == 0u && expert_mid_dim % 16u == 0u && out_dim % 16u == 0u;
        const int moe_wmma_f16_mid_all = moe_wmma_f16_mid && moe_wmma_f16_mid_all_req;
        const int moe_wmma_direct_sum = moe_wmma_f16_mid && moe_wmma_direct_sum_req;
        const int moe_wmma_f16_down = moe_wmma_f16_mid && !moe_wmma_direct_sum && moe_wmma_f16_down_req;
        const int moe_wmma_f16_down_all = moe_wmma_f16_mid && !moe_wmma_direct_sum && moe_wmma_f16_down_all_req;
        const int moe_wmma_x_f16_req =
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_X_F16", "DS4_HIP_MOE_WMMA_X_F16", NULL);
#else
        const int moe_dense_hot = 0;
        const int moe_wmma_f16_mid = 0;
        const int moe_wmma_f16_mid_all = 0;
        const int moe_wmma_direct_sum = 0;
        const int moe_wmma_f16_down = 0;
        const int moe_wmma_f16_down_all = 0;
        const int moe_wmma_x_f16_req = 0;
#endif
        const int moe_wmma_f16_down_any = moe_wmma_f16_down || moe_wmma_f16_down_all;
        const int moe_wmma_x_f16 = moe_wmma_f16_mid && moe_wmma_x_f16_req;
        const uint64_t f16_mid_bytes = moe_wmma_f16_mid ? (uint64_t)pair_count * expert_mid_dim * sizeof(__half) : 0ull;
        const uint64_t f16_down_bytes = moe_wmma_f16_down_any ? (uint64_t)pair_count * out_dim * sizeof(__half) : 0ull;
        const uint64_t f16_pair_mask_bytes = (moe_wmma_f16_down && !moe_wmma_f16_down_all) ? (uint64_t)pair_count * sizeof(uint8_t) : 0ull;
        const uint64_t wmma_x_bytes = moe_wmma_x_f16 ? (uint64_t)n_tokens * expert_in_dim * sizeof(__half) : 0ull;
        const uint64_t dense_x_bytes = moe_dense_hot ? (uint64_t)n_tokens * expert_in_dim * sizeof(__half) : 0ull;
        const uint64_t dense_gate_w_bytes = moe_dense_hot ? (uint64_t)expert_mid_dim * expert_in_dim * sizeof(__half) : 0ull;
        const uint64_t dense_up_w_bytes = dense_gate_w_bytes;
        const uint64_t dense_down_w_bytes = moe_dense_hot ? (uint64_t)out_dim * expert_mid_dim * sizeof(__half) : 0ull;
        const uint64_t dense_gate_bytes = moe_dense_hot ? (uint64_t)n_tokens * expert_mid_dim * sizeof(__half) : 0ull;
        const uint64_t dense_up_bytes = dense_gate_bytes;
        const uint64_t dense_down_bytes_h = moe_dense_hot ? (uint64_t)n_tokens * out_dim * sizeof(__half) : 0ull;
        auto align256 = [](uint64_t v) -> uint64_t { return (v + 255ull) & ~255ull; };
        const uint64_t base_scratch_end = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes + hot_gate_bytes + hot_down_bytes + med_gate_bytes + med_down_bytes + f16_low_gate_bytes + f16_low_down_bytes + 4ull * wmma_tile_bytes;
        const uint64_t f16_mid_off = align256(base_scratch_end);
        const uint64_t f16_down_off = align256(f16_mid_off + f16_mid_bytes);
        const uint64_t f16_pair_mask_off = align256(f16_down_off + f16_down_bytes);
        const uint64_t wmma_x_off = align256(f16_pair_mask_off + f16_pair_mask_bytes);
        const uint64_t dense_base = align256(wmma_x_off + wmma_x_bytes);
        const uint64_t dense_x_off = dense_base;
        const uint64_t dense_gate_w_off = align256(dense_x_off + dense_x_bytes);
        const uint64_t dense_up_w_off = align256(dense_gate_w_off + dense_gate_w_bytes);
        const uint64_t dense_down_w_off = align256(dense_up_w_off + dense_up_w_bytes);
        const uint64_t dense_gate_off = align256(dense_down_w_off + dense_down_w_bytes);
        const uint64_t dense_up_off = align256(dense_gate_off + dense_gate_bytes);
        const uint64_t dense_down_off = align256(dense_up_off + dense_up_bytes);
        const uint64_t scratch_bytes = moe_dense_hot
            ? align256(dense_down_off + dense_down_bytes_h)
            : dense_base;
        uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        const uint64_t wmma_list_base = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
        const uint64_t wmma_tile_base = wmma_list_base + hot_gate_bytes + hot_down_bytes + med_gate_bytes + med_down_bytes + f16_low_gate_bytes + f16_low_down_bytes;
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_medium_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_medium_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + med_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + med_gate_bytes + med_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + med_gate_bytes + med_down_bytes + f16_low_gate_bytes);
        uint32_t *wmma_gate_tile_experts_dev = (uint32_t *)(scratch + wmma_tile_base);
        uint32_t *wmma_gate_tile_starts_dev = (uint32_t *)(scratch + wmma_tile_base + wmma_tile_bytes);
        uint32_t *wmma_down_tile_experts_dev = (uint32_t *)(scratch + wmma_tile_base + 2ull * wmma_tile_bytes);
        uint32_t *wmma_down_tile_starts_dev = (uint32_t *)(scratch + wmma_tile_base + 3ull * wmma_tile_bytes);
        __half *wmma_mid_h = moe_wmma_f16_mid ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_f16_down_any ? (__half *)(scratch + f16_down_off) : NULL;
        uint8_t *wmma_f16_pair_mask_dev = (moe_wmma_f16_down && !moe_wmma_f16_down_all) ? (uint8_t *)(scratch + f16_pair_mask_off) : NULL;
        __half *wmma_x_h = moe_wmma_x_f16 ? (__half *)(scratch + wmma_x_off) : NULL;
        __half *dense_x = moe_dense_hot ? (__half *)(scratch + dense_x_off) : NULL;
        __half *dense_gate_w = moe_dense_hot ? (__half *)(scratch + dense_gate_w_off) : NULL;
        __half *dense_up_w = moe_dense_hot ? (__half *)(scratch + dense_up_w_off) : NULL;
        __half *dense_down_w = moe_dense_hot ? (__half *)(scratch + dense_down_w_off) : NULL;
        __half *dense_gate_h = moe_dense_hot ? (__half *)(scratch + dense_gate_off) : NULL;
        __half *dense_up_h = moe_dense_hot ? (__half *)(scratch + dense_up_off) : NULL;
        __half *dense_down_h = moe_dense_hot ? (__half *)(scratch + dense_down_off) : NULL;
        const uint32_t profile_q2_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t q2_prof[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_q2_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&q2_prof[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(q2_prof[j]);
                    memset(q2_prof, 0, sizeof(q2_prof));
                    break;
                }
            }
            if (q2_prof[0]) (void)cudaEventRecord(q2_prof[0], 0);
        }
        ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected->ptr,
                    pair_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    sorted_pairs,
                    cursors,
                    (const int32_t *)selected->ptr,
                    pair_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_gate_hot_count = 0u, wmma_down_hot_count = 0u;
        uint32_t wmma_gate_hot_max = 0u, wmma_down_hot_max = 0u;
        uint32_t wmma_gate_medium_count = 0u, wmma_down_medium_count = 0u;
        uint32_t wmma_gate_medium_max = 0u, wmma_down_medium_max = 0u;
        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint8_t wmma_f16_hot_mask[256] = {0};
        uint32_t h_counts[256] = {0};
        uint32_t h_gate_hot[256] = {0};
        uint32_t h_down_hot[256] = {0};
        uint32_t h_gate_medium[256] = {0};
        uint32_t h_down_medium[256] = {0};
        uint32_t h_f16_hot[256] = {0};
        uint32_t h_f16_low[256] = {0};
        uint32_t wmma_gate_hot_threshold = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_GATE_HOT",
                                                                    "DS4_HIP_MOE_WMMA_GATE_HOT",
                                                                    16u, 1u, 65535u);
        uint32_t wmma_down_hot_threshold = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_DOWN_HOT",
                                                                    "DS4_HIP_MOE_WMMA_DOWN_HOT",
                                                                    16u, 1u, 65535u);
        uint32_t wmma_mtiles = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_MTILES",
                                                        "DS4_HIP_MOE_WMMA_MTILES",
                                                        8u, 4u, 16u);
        if (wmma_mtiles != 4u && wmma_mtiles != 8u && wmma_mtiles != 16u) wmma_mtiles = 8u;
        uint32_t wmma_gate_medium_threshold = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_GATE_MEDIUM",
                                                                       "DS4_HIP_MOE_WMMA_GATE_MEDIUM",
                                                                       12u, 1u, 65535u);
        uint32_t wmma_down_medium_threshold = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_DOWN_MEDIUM",
                                                                       "DS4_HIP_MOE_WMMA_DOWN_MEDIUM",
                                                                       12u, 1u, 65535u);
        uint32_t wmma_medium_mtiles = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_MEDIUM_MTILES",
                                                               "DS4_HIP_MOE_WMMA_MEDIUM_MTILES",
                                                               4u, 4u, 8u);
        if (wmma_medium_mtiles != 4u && wmma_medium_mtiles != 8u) wmma_medium_mtiles = 4u;
        const uint32_t moe_wmma_medium = moe_wmma_hot &&
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_MEDIUM", "DS4_HIP_MOE_WMMA_MEDIUM", NULL);
        uint32_t wmma_f16_split_threshold = cuda_parse_u32_env_alias("DS4_CUDA_MOE_WMMA_F16_SPLIT_MIN",
                                                                     "DS4_HIP_MOE_WMMA_F16_SPLIT_MIN",
                                                                     64u, 1u, 65535u);
        const uint32_t moe_wmma_f16_split = moe_wmma_f16_mid &&
            (!moe_wmma_f16_down || moe_wmma_f16_down_all) &&
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_F16_SPLIT", "DS4_HIP_MOE_WMMA_F16_SPLIT", NULL);
        const uint32_t moe_slot_partial = moe_wmma_f16_down_all && !moe_wmma_direct_sum && !moe_dense_hot && !moe_wmma_medium &&
            cuda_env_flag_any3("DS4_CUDA_MOE_SLOT_PARTIAL", "DS4_HIP_MOE_SLOT_PARTIAL", NULL);
        const uint32_t moe_wmma_split_hot = moe_wmma_hot &&
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_SPLIT_HOT", "DS4_HIP_MOE_WMMA_SPLIT_HOT", NULL);
        const uint32_t moe_wmma_tile_hot = moe_wmma_hot &&
            cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_TILE_HOT", "DS4_HIP_MOE_WMMA_TILE_HOT", NULL);
        uint32_t wmma_gate_tile_count = 0u, wmma_down_tile_count = 0u;
        uint32_t dense_hot_experts[8] = {0};
        uint32_t dense_hot_counts[8] = {0};
        uint8_t dense_hot_mask[256] = {0};
        uint32_t dense_hot_n = 0u;
        uint32_t route_nz = 0u, route_max = 0u;
        uint32_t route_lt12_e = 0u, route_lt12_p = 0u;
        uint32_t route_m12_27_e = 0u, route_m12_27_p = 0u;
        uint32_t route_28_63_e = 0u, route_28_63_p = 0u;
        uint32_t route_64_127_e = 0u, route_64_127_p = 0u;
        uint32_t route_128_255_e = 0u, route_128_255_p = 0u;
        uint32_t route_ge256_e = 0u, route_ge256_p = 0u;
        if (moe_wmma_hot || moe_dense_hot || profile_q2_moe) {
            if (!cuda_ok(cudaMemcpy(h_counts, counts, 256u * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < 256u; e++) {
                const uint32_t c = h_counts[e];
                if (c != 0u) {
                    route_nz++;
                    if (c > route_max) route_max = c;
                }
                if (c < 12u) {
                    if (c != 0u) route_lt12_e++;
                    route_lt12_p += c;
                } else if (c < 28u) {
                    route_m12_27_e++;
                    route_m12_27_p += c;
                } else if (c < 64u) {
                    route_28_63_e++;
                    route_28_63_p += c;
                } else if (c < 128u) {
                    route_64_127_e++;
                    route_64_127_p += c;
                } else if (c < 256u) {
                    route_128_255_e++;
                    route_128_255_p += c;
                } else {
                    route_ge256_e++;
                    route_ge256_p += c;
                }
            }
            if (moe_dense_hot && moe_wmma_hot) {
                const uint32_t dense_min = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DENSE_HOT_MIN",
                                                                     "DS4_HIP_MOE_DENSE_HOT_MIN",
                                                                     wmma_gate_hot_threshold, 1u, 65535u);
                uint32_t dense_top = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DENSE_HOT_TOP",
                                                              "DS4_HIP_MOE_DENSE_HOT_TOP",
                                                              1u, 1u, 8u);
                for (uint32_t pick = 0; pick < dense_top; pick++) {
                    uint32_t best_e = UINT32_MAX;
                    uint32_t best_c = 0u;
                    for (uint32_t e = 0; e < 256u; e++) {
                        const uint32_t c = h_counts[e];
                        if (!dense_hot_mask[e] && c >= dense_min && c > best_c) {
                            best_c = c;
                            best_e = e;
                        }
                    }
                    if (best_e == UINT32_MAX) break;
                    dense_hot_mask[best_e] = 1u;
                    dense_hot_experts[dense_hot_n] = best_e;
                    dense_hot_counts[dense_hot_n] = best_c;
                    dense_hot_n++;
                }
            }
            if (moe_wmma_f16_mid) {
                const uint32_t f16_min = wmma_gate_hot_threshold > wmma_down_hot_threshold
                    ? wmma_gate_hot_threshold : wmma_down_hot_threshold;
                for (uint32_t e = 0; e < 256u; e++) {
                    const uint32_t c = h_counts[e];
                    if (!dense_hot_mask[e] && c >= f16_min) {
                        wmma_f16_hot_mask[e] = 1u;
                        if (moe_wmma_f16_split && c < wmma_f16_split_threshold) {
                            h_f16_low[wmma_f16_low_count++] = e;
                            if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                        } else {
                            h_f16_hot[wmma_f16_hot_count++] = e;
                            if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                        }
                    }
                }
            }
            for (uint32_t e = 0; e < 256u; e++) {
                const uint32_t c = h_counts[e];
                if (!dense_hot_mask[e] && !wmma_f16_hot_mask[e] && c >= wmma_gate_hot_threshold) {
                    h_gate_hot[wmma_gate_hot_count++] = e;
                    if (c > wmma_gate_hot_max) wmma_gate_hot_max = c;
                } else if (moe_wmma_medium && !dense_hot_mask[e] && !wmma_f16_hot_mask[e] &&
                           c >= wmma_gate_medium_threshold && c < wmma_gate_hot_threshold) {
                    h_gate_medium[wmma_gate_medium_count++] = e;
                    if (c > wmma_gate_medium_max) wmma_gate_medium_max = c;
                }
                if (!dense_hot_mask[e] && !wmma_f16_hot_mask[e] && c >= wmma_down_hot_threshold) {
                    h_down_hot[wmma_down_hot_count++] = e;
                    if (c > wmma_down_hot_max) wmma_down_hot_max = c;
                } else if (moe_wmma_medium && !dense_hot_mask[e] && !wmma_f16_hot_mask[e] &&
                           c >= wmma_down_medium_threshold && c < wmma_down_hot_threshold) {
                    h_down_medium[wmma_down_medium_count++] = e;
                    if (c > wmma_down_medium_max) wmma_down_medium_max = c;
                }
            }
            if (wmma_gate_hot_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_gate_hot,
                                    wmma_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma gate hot copy")) return 0;
            if (wmma_down_hot_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_down_hot,
                                    wmma_down_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma down hot copy")) return 0;
            if (wmma_gate_medium_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_gate_medium_dev, h_gate_medium,
                                    wmma_gate_medium_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma gate medium copy")) return 0;
            if (wmma_down_medium_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_down_medium_dev, h_down_medium,
                                    wmma_down_medium_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma down medium copy")) return 0;
            if (moe_wmma_tile_hot) {
                std::vector<uint32_t> gate_tile_experts;
                std::vector<uint32_t> gate_tile_starts;
                std::vector<uint32_t> down_tile_experts;
                std::vector<uint32_t> down_tile_starts;
                gate_tile_experts.reserve(wmma_tile_capacity);
                gate_tile_starts.reserve(wmma_tile_capacity);
                down_tile_experts.reserve(wmma_tile_capacity);
                down_tile_starts.reserve(wmma_tile_capacity);
                for (uint32_t e = 0; e < 256u; e++) {
                    const uint32_t c = h_counts[e];
                    if (!dense_hot_mask[e] && !wmma_f16_hot_mask[e] && c >= wmma_gate_hot_threshold) {
                        for (uint32_t s = 0; s < c; s += 128u) {
                            gate_tile_experts.push_back(e);
                            gate_tile_starts.push_back(s);
                        }
                    }
                    if (!dense_hot_mask[e] && !wmma_f16_hot_mask[e] && c >= wmma_down_hot_threshold) {
                        for (uint32_t s = 0; s < c; s += 128u) {
                            down_tile_experts.push_back(e);
                            down_tile_starts.push_back(s);
                        }
                    }
                }
                wmma_gate_tile_count = (uint32_t)gate_tile_experts.size();
                wmma_down_tile_count = (uint32_t)down_tile_experts.size();
                if (wmma_gate_tile_count > wmma_tile_capacity || wmma_down_tile_count > wmma_tile_capacity) {
                    fprintf(stderr, DS4_GPU_LOG_PREFIX "MoE q2 WMMA tile list overflow gate=%u down=%u cap=%u\n",
                            wmma_gate_tile_count, wmma_down_tile_count, wmma_tile_capacity);
                    return 0;
                }
                if (wmma_gate_tile_count != 0u) {
                    if (!cuda_ok(cudaMemcpy(wmma_gate_tile_experts_dev, gate_tile_experts.data(),
                                            wmma_gate_tile_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                 "routed_moe q2 wmma gate tile expert copy")) return 0;
                    if (!cuda_ok(cudaMemcpy(wmma_gate_tile_starts_dev, gate_tile_starts.data(),
                                            wmma_gate_tile_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                 "routed_moe q2 wmma gate tile start copy")) return 0;
                }
                if (wmma_down_tile_count != 0u) {
                    if (!cuda_ok(cudaMemcpy(wmma_down_tile_experts_dev, down_tile_experts.data(),
                                            wmma_down_tile_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                 "routed_moe q2 wmma down tile expert copy")) return 0;
                    if (!cuda_ok(cudaMemcpy(wmma_down_tile_starts_dev, down_tile_starts.data(),
                                            wmma_down_tile_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                 "routed_moe q2 wmma down tile start copy")) return 0;
                }
            }
        }
        if (q2_prof[1]) (void)cudaEventRecord(q2_prof[1], 0);

        uint32_t gate_tile = cuda_parse_u32_env_alias("DS4_CUDA_MOE_GATE_TILE", "DS4_HIP_MOE_GATE_TILE", 4u, 4u, 16u);
        uint32_t down_tile = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DOWN_TILE", "DS4_HIP_MOE_DOWN_TILE", 4u, 4u, 16u);
        if (gate_tile != 4u && gate_tile != 8u && gate_tile != 16u) gate_tile = 4u;
        if (down_tile != 4u && down_tile != 8u && down_tile != 16u) down_tile = 4u;
        uint32_t gate_rpb = cuda_parse_u32_env_alias("DS4_CUDA_MOE_GATE_RPB", "DS4_HIP_MOE_GATE_RPB", 16u, 1u, 16u);
        uint32_t down_rpb = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DOWN_RPB", "DS4_HIP_MOE_DOWN_RPB", 16u, 1u, 16u);
        if (gate_rpb == 0u) gate_rpb = 1u;
        if (down_rpb == 0u) down_rpb = 1u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = (size_t)gate_tile * 256u * sizeof(float);
        const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
        const uint32_t gate_scalar_max = ((moe_wmma_hot && (wmma_gate_medium_count != 0u || wmma_gate_hot_count != 0u || wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)) || dense_hot_n != 0u)
            ? (wmma_gate_medium_count != 0u ? wmma_gate_medium_threshold : wmma_gate_hot_threshold) : 0u;
        const uint32_t down_scalar_max = ((moe_wmma_hot && (wmma_down_medium_count != 0u || wmma_down_hot_count != 0u || wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)) || dense_hot_n != 0u)
            ? (wmma_down_medium_count != 0u ? wmma_down_medium_threshold : wmma_down_hot_threshold) : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, 256u, 1);
        if (gate_tile == 4u) {
            if (moe_wmma_f16_mid_all) {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4,true><<<gate_grid, gate_threads, gate_shmem>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            }
        } else if (gate_tile == 8u) {
            if (moe_wmma_f16_mid_all) {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<8,true><<<gate_grid, gate_threads, gate_shmem>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<8><<<gate_grid, gate_threads, gate_shmem>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            }
        } else {
            if (moe_wmma_f16_mid_all) {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<16,true><<<gate_grid, gate_threads, gate_shmem>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<16><<<gate_grid, gate_threads, gate_shmem>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, 1u, gate_scalar_max, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, clamp);
            }
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#ifdef __HIP_PLATFORM_AMD__
        if (moe_dense_hot && moe_wmma_hot && dense_hot_n != 0u) {
            int dense_ok = 1;
            for (uint32_t di = 0; di < dense_hot_n; di++) {
                const uint32_t dense_hot_expert = dense_hot_experts[di];
                const uint32_t dense_hot_count = dense_hot_counts[di];
                const uint64_t x_elems = (uint64_t)dense_hot_count * expert_in_dim;
                const uint64_t mid_elems = (uint64_t)dense_hot_count * expert_mid_dim;
                const uint64_t down_elems = (uint64_t)dense_hot_count * out_dim;
                __half *gate_w_h = moe_dense_weight_f16_cached(gate_w, dense_hot_expert, expert_in_dim, expert_mid_dim,
                                                                gate_expert_bytes, gate_row_bytes, dense_gate_w,
                                                                "routed_moe q2 dense gate dequant launch", &dense_ok);
                if (!dense_ok) return 0;
                __half *up_w_h = moe_dense_weight_f16_cached(up_w, dense_hot_expert, expert_in_dim, expert_mid_dim,
                                                              gate_expert_bytes, gate_row_bytes, dense_up_w,
                                                              "routed_moe q2 dense up dequant launch", &dense_ok);
                if (!dense_ok) return 0;
                __half *down_w_h = moe_dense_weight_f16_cached(down_w, dense_hot_expert, expert_mid_dim, out_dim,
                                                                down_expert_bytes, down_row_bytes, dense_down_w,
                                                                "routed_moe q2 dense down dequant launch", &dense_ok);
                if (!dense_ok) return 0;
                moe_dense_gather_x_f16_kernel<<<(x_elems + 255u) / 256u, 256>>>(
                        dense_x, (const float *)x->ptr, offsets, sorted_pairs, dense_hot_expert,
                        dense_hot_count, expert_in_dim);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 dense gather x launch")) return 0;
                if (!hipblaslt_gemm_tn_f16_out_f16(dense_gate_h, gate_w_h, dense_x,
                                                   expert_mid_dim, dense_hot_count, expert_in_dim,
                                                   "moe dense gate")) return 0;
                if (!hipblaslt_gemm_tn_f16_out_f16(dense_up_h, up_w_h, dense_x,
                                                   expert_mid_dim, dense_hot_count, expert_in_dim,
                                                   "moe dense up")) return 0;
                moe_dense_swiglu_f16_kernel<<<(mid_elems + 255u) / 256u, 256>>>(
                        dense_gate_h, dense_gate_h, dense_up_h, (const float *)weights->ptr,
                        offsets, sorted_pairs, dense_hot_expert, dense_hot_count, expert_mid_dim, clamp);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 dense swiglu launch")) return 0;
                if (!hipblaslt_gemm_tn_f16_out_f16(dense_down_h, down_w_h, dense_gate_h,
                                                   out_dim, dense_hot_count, expert_mid_dim,
                                                   "moe dense down")) return 0;
                moe_dense_scatter_down_f16_kernel<<<(down_elems + 255u) / 256u, 256>>>(
                        (float *)down->ptr, dense_down_h, offsets, sorted_pairs, dense_hot_expert,
                        dense_hot_count, out_dim);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 dense down scatter launch")) return 0;
            }
        }
#endif
        if (q2_prof[2]) (void)cudaEventRecord(q2_prof[2], 0);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_f16_mid && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            if (moe_wmma_x_f16) {
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, NULL, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_f16_mid && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            if (moe_wmma_x_f16) {
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                        NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, NULL, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
        if (moe_wmma_medium && wmma_gate_medium_count != 0u) {
            constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
            if (wmma_medium_mtiles == 8u) {
                constexpr uint32_t mt8 = 8u;
                const dim3 block(32u * mt8, 1u, 1u);
                const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_gate_medium_max + mt8 * bm - 1u) / (mt8 * bm),
                                wmma_gate_medium_count);
                const size_t shmem_n2 = (mt8 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                        (4u * mt8 * bm * bn) * sizeof(float);
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_medium_dev, wmma_gate_medium_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                constexpr uint32_t mt4 = 4u;
                const dim3 block(32u * mt4, 1u, 1u);
                const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_gate_medium_max + mt4 * bm - 1u) / (mt4 * bm),
                                wmma_gate_medium_count);
                const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                        (4u * mt4 * bm * bn) * sizeof(float);
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_medium_dev, wmma_gate_medium_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma medium gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_gate_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const size_t shmem = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                 (2u * mt * bm * bn) * sizeof(float);
            if (wmma_f16_hot_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_gate_hot,
                                    wmma_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma gate hot recopy")) return 0;
            if (moe_wmma_tile_hot && wmma_gate_tile_count != 0u) {
                if (!cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_NO_GATE_N2", "DS4_HIP_MOE_WMMA_NO_GATE_N2", NULL)) {
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn), wmma_gate_tile_count, 1u);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    moe_gate_up_mid_q2K_hottile_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                            counts, offsets, sorted_pairs, wmma_gate_tile_experts_dev, wmma_gate_tile_starts_dev,
                            wmma_gate_tile_count, expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                } else {
                    const dim3 grid(expert_mid_dim / bn, wmma_gate_tile_count, 1u);
                    moe_gate_up_mid_q2K_hottile_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                            (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                            counts, offsets, sorted_pairs, wmma_gate_tile_experts_dev, wmma_gate_tile_starts_dev,
                            wmma_gate_tile_count, expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                }
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma tile hot gate/up launch")) return 0;
            } else if (moe_wmma_split_hot) {
                const uint32_t bounds[] = {128u, 256u, 512u, 1024u, 2048u, 4096u, 65536u};
                uint32_t lo = wmma_gate_hot_threshold;
                for (uint32_t bi = 0; bi < sizeof(bounds) / sizeof(bounds[0]); bi++) {
                    const uint32_t hi = bounds[bi];
                    if (hi <= lo) continue;
                    uint32_t h_bucket[256];
                    uint32_t bucket_count = 0u;
                    uint32_t bucket_max = 0u;
                    for (uint32_t e = 0; e < 256u; e++) {
                        if (dense_hot_mask[e] || wmma_f16_hot_mask[e]) continue;
                        const uint32_t c = h_counts[e];
                        if (c >= lo && c < hi) {
                            h_bucket[bucket_count++] = e;
                            if (c > bucket_max) bucket_max = c;
                        }
                    }
                    if (bucket_count != 0u) {
                        if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_bucket,
                                                bucket_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                     "routed_moe q2 wmma split gate hot copy")) return 0;
                        const dim3 grid(expert_mid_dim / bn, (bucket_max + mt * bm - 1u) / (mt * bm), bucket_count);
                        moe_gate_up_mid_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                                (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                                counts, offsets, sorted_pairs, wmma_gate_hot_dev, bucket_count,
                                expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma split hot gate/up launch")) return 0;
                    }
                    lo = hi;
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt16 = 16u, bm16 = 16u, bn16 = 16u, bk16 = 16u;
                const dim3 block16(32u * mt16, 1u, 1u);
                const dim3 grid(expert_mid_dim / bn16, (wmma_gate_hot_max + mt16 * bm16 - 1u) / (mt16 * bm16), wmma_gate_hot_count);
                const size_t shmem16 = (mt16 * bm16 * bk16 + 2u * bk16 * bn16) * sizeof(half) +
                                       (2u * mt16 * bm16 * bn16) * sizeof(float);
                moe_gate_up_mid_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block16, shmem16>>>(
                        (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_gate_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot gate/up mt16 launch")) return 0;
            } else if (wmma_mtiles == 4u) {
                constexpr uint32_t mt4 = 4u, bm4 = 16u, bn4 = 16u, bk4 = 16u;
                const dim3 block4(32u * mt4, 1u, 1u);
                const dim3 grid(expert_mid_dim / bn4, (wmma_gate_hot_max + mt4 * bm4 - 1u) / (mt4 * bm4), wmma_gate_hot_count);
                const size_t shmem4 = (mt4 * bm4 * bk4 + 2u * bk4 * bn4) * sizeof(half) +
                                      (2u * mt4 * bm4 * bn4) * sizeof(float);
                moe_gate_up_mid_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block4, shmem4>>>(
                        (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_gate_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot gate/up mt4 launch")) return 0;
            } else if (!cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_NO_GATE_N2", "DS4_HIP_MOE_WMMA_NO_GATE_N2", NULL)) {
                const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                wmma_gate_hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                        (4u * mt * bm * bn) * sizeof(float);
                moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                        (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_gate_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot gate/up n2 launch")) return 0;
            } else {
                const dim3 grid(expert_mid_dim / bn, (wmma_gate_hot_max + mt * bm - 1u) / (mt * bm), wmma_gate_hot_count);
                moe_gate_up_mid_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                        (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_gate_hot_count,
                        expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot gate/up launch")) return 0;
            }
        }
#endif
        if (q2_prof[3]) (void)cudaEventRecord(q2_prof[3], 0);

        if (moe_wmma_direct_sum &&
            !cuda_ok(cudaMemset(out->ptr, 0, (uint64_t)n_tokens * out_dim * sizeof(float)),
                     "routed_moe q2 direct sum output clear")) return 0;
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, 256u, 1);
        if (down_tile == 4u) {
            if (moe_wmma_direct_sum) {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<4,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<4,false,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            } else if (moe_wmma_f16_down_all) {
                if (moe_wmma_f16_mid_all) {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<4,true,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<4,true,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                } else {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                }
            } else {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<4,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            }
        } else if (down_tile == 8u) {
            if (moe_wmma_direct_sum) {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<8,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<8,false,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            } else if (moe_wmma_f16_down_all) {
                if (moe_wmma_f16_mid_all) {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<8,true,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<8,true,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                } else {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                }
            } else {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<8,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            }
        } else {
            if (moe_wmma_direct_sum) {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<16,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<16,false,false,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            } else if (moe_wmma_f16_down_all) {
                if (moe_wmma_f16_mid_all) {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<16,true,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<16,true,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                } else {
                    if (moe_slot_partial) {
                        moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes, n_tokens);
                    } else {
                        moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                                NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                                counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                                down_expert_bytes, down_row_bytes);
                    }
                }
            } else {
                if (moe_wmma_f16_mid_all) {
                    moe_down_q2K_expert_batch_sharedmid_kernel<16,true><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, 1u, down_scalar_max, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes);
                }
            }
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert down launch")) return 0;
        if (q2_prof[4]) (void)cudaEventRecord(q2_prof[4], 0);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_f16_mid && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            if (moe_wmma_direct_sum) {
                moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false,true><<<grid, block, shmem_n2>>>(
                        (float *)out->ptr, NULL, down_w, NULL, wmma_mid_h,
                        counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            } else if (moe_wmma_f16_down_any) {
                if (moe_slot_partial) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true,false,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_tokens);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else {
                moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                        (float *)down->ptr, NULL, down_w, NULL, wmma_mid_h,
                        counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_f16_mid && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            if (moe_wmma_direct_sum) {
                moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false,true><<<grid, block, shmem_n2>>>(
                        (float *)out->ptr, NULL, down_w, NULL, wmma_mid_h,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            } else if (moe_wmma_f16_down_any) {
                if (moe_slot_partial) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true,false,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_tokens);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                            counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else {
                moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                        (float *)down->ptr, NULL, down_w, NULL, wmma_mid_h,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
            if (moe_wmma_f16_down && !moe_wmma_f16_down_all) {
                if (!cuda_ok(cudaMemset(wmma_f16_pair_mask_dev, 0, f16_pair_mask_bytes),
                             "routed_moe q2 wmma f16 pair mask clear")) return 0;
                const dim3 mark_grid((wmma_f16_hot_max + 255u) / 256u, wmma_f16_hot_count, 1u);
                moe_mark_hot_pairs_kernel<<<mark_grid, 256>>>(
                        wmma_f16_pair_mask_dev, counts, offsets, sorted_pairs,
                        wmma_down_hot_dev, wmma_f16_hot_count);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16 pair mask mark")) return 0;
            }
        }
        if (moe_wmma_medium && wmma_down_medium_count != 0u) {
            constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
            if (wmma_medium_mtiles == 8u) {
                constexpr uint32_t mt8 = 8u;
                const dim3 block(32u * mt8, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_down_medium_max + mt8 * bm - 1u) / (mt8 * bm),
                                wmma_down_medium_count);
                const size_t shmem_n2 = (mt8 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt8 * bm * bn) * sizeof(float);
                if (moe_wmma_direct_sum) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,false,true><<<grid, block, shmem_n2>>>(
                            (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (moe_wmma_f16_down_all) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else {
                constexpr uint32_t mt4 = 4u;
                const dim3 block(32u * mt4, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_down_medium_max + mt4 * bm - 1u) / (mt4 * bm),
                                wmma_down_medium_count);
                const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt4 * bm * bn) * sizeof(float);
                if (moe_wmma_direct_sum) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,false,true><<<grid, block, shmem_n2>>>(
                            (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (moe_wmma_f16_down_all) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, wmma_down_medium_dev, wmma_down_medium_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            }
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma medium down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_down_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            if (wmma_f16_hot_count != 0u &&
                !cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_down_hot,
                                    wmma_down_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma down hot recopy")) return 0;
            if (moe_wmma_direct_sum) {
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_down_hot_max + mt * bm - 1u) / (mt * bm),
                                wmma_down_hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,false,true><<<grid, block, shmem_n2>>>(
                        (float *)out->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down direct-sum launch")) return 0;
            } else if (moe_wmma_f16_down_all) {
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_down_hot_max + mt * bm - 1u) / (mt * bm),
                                wmma_down_hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                        NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down f16-all launch")) return 0;
            } else if (moe_wmma_tile_hot && wmma_down_tile_count != 0u) {
                if (!cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_NO_DOWN_N2", "DS4_HIP_MOE_WMMA_NO_DOWN_N2", NULL)) {
                    const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn), wmma_down_tile_count, 1u);
                    const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                            (2u * mt * bm * bn) * sizeof(float);
                    moe_down_q2K_hottile_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, down_w, (const float *)mid->ptr,
                            counts, offsets, sorted_pairs, wmma_down_tile_experts_dev, wmma_down_tile_starts_dev,
                            wmma_down_tile_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    const dim3 grid(out_dim / bn, wmma_down_tile_count, 1u);
                    moe_down_q2K_hottile_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                            (float *)down->ptr, down_w, (const float *)mid->ptr,
                            counts, offsets, sorted_pairs, wmma_down_tile_experts_dev, wmma_down_tile_starts_dev,
                            wmma_down_tile_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma tile hot down launch")) return 0;
            } else if (moe_wmma_split_hot) {
                const uint32_t bounds[] = {128u, 256u, 512u, 1024u, 2048u, 4096u, 65536u};
                uint32_t lo = wmma_down_hot_threshold;
                for (uint32_t bi = 0; bi < sizeof(bounds) / sizeof(bounds[0]); bi++) {
                    const uint32_t hi = bounds[bi];
                    if (hi <= lo) continue;
                    uint32_t h_bucket[256];
                    uint32_t bucket_count = 0u;
                    uint32_t bucket_max = 0u;
                    for (uint32_t e = 0; e < 256u; e++) {
                        if (dense_hot_mask[e] || wmma_f16_hot_mask[e]) continue;
                        const uint32_t c = h_counts[e];
                        if (c >= lo && c < hi) {
                            h_bucket[bucket_count++] = e;
                            if (c > bucket_max) bucket_max = c;
                        }
                    }
                    if (bucket_count != 0u) {
                        if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_bucket,
                                                bucket_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                                     "routed_moe q2 wmma split down hot copy")) return 0;
                        const dim3 grid(out_dim / bn, (bucket_max + mt * bm - 1u) / (mt * bm), bucket_count);
                        moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                                (float *)down->ptr, down_w, (const float *)mid->ptr,
                                counts, offsets, sorted_pairs, wmma_down_hot_dev, bucket_count,
                                expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma split hot down launch")) return 0;
                    }
                    lo = hi;
                }
            } else if (!cuda_env_flag_any3("DS4_CUDA_MOE_WMMA_NO_DOWN_N2", "DS4_HIP_MOE_WMMA_NO_DOWN_N2", NULL)) {
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (wmma_down_hot_max + mt * bm - 1u) / (mt * bm),
                                wmma_down_hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                        (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down n2 launch")) return 0;
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt16 = 16u, bm16 = 16u, bn16 = 16u, bk16 = 16u;
                const dim3 block16(32u * mt16, 1u, 1u);
                const dim3 grid(out_dim / bn16, (wmma_down_hot_max + mt16 * bm16 - 1u) / (mt16 * bm16), wmma_down_hot_count);
                const size_t shmem16 = (mt16 * bm16 * bk16 + bk16 * bn16) * sizeof(half) +
                                       (mt16 * bm16 * bn16) * sizeof(float);
                moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block16, shmem16>>>(
                        (float *)down->ptr, down_w, (const float *)mid->ptr,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down mt16 launch")) return 0;
            } else if (wmma_mtiles == 4u) {
                constexpr uint32_t mt4 = 4u, bm4 = 16u, bn4 = 16u, bk4 = 16u;
                const dim3 block4(32u * mt4, 1u, 1u);
                const dim3 grid(out_dim / bn4, (wmma_down_hot_max + mt4 * bm4 - 1u) / (mt4 * bm4), wmma_down_hot_count);
                const size_t shmem4 = (mt4 * bm4 * bk4 + bk4 * bn4) * sizeof(half) +
                                      (mt4 * bm4 * bn4) * sizeof(float);
                moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block4, shmem4>>>(
                        (float *)down->ptr, down_w, (const float *)mid->ptr,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down mt4 launch")) return 0;
            } else {
                const dim3 grid(out_dim / bn, (wmma_down_hot_max + mt * bm - 1u) / (mt * bm), wmma_down_hot_count);
                moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                        (float *)down->ptr, down_w, (const float *)mid->ptr,
                        counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_down_hot_count,
                        expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma hot down launch")) return 0;
            }
        }
#endif
        if (q2_prof[5]) (void)cudaEventRecord(q2_prof[5], 0);
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_direct_sum) {
            ok = 1;
        } else if (moe_wmma_f16_down_all) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                if (moe_slot_partial) {
                    moe_sum_slot_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                            (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
                } else {
                    moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                            (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
                }
            } else if (moe_slot_partial) {
                moe_sum_slot_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else if (moe_wmma_f16_down && wmma_f16_hot_count != 0u) {
            moe_sum_f16_hot_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, wmma_down_h,
                    wmma_f16_pair_mask_dev, out_dim, n_expert, n_tokens);
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert sum launch");
        if (q2_prof[6]) {
            (void)cudaEventRecord(q2_prof[6], 0);
            if (cudaEventSynchronize(q2_prof[6]) == cudaSuccess) {
                float ms_bucket = 0.0f, ms_gate_scalar = 0.0f, ms_gate_wmma = 0.0f;
                float ms_down_scalar = 0.0f, ms_down_wmma = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_bucket, q2_prof[0], q2_prof[1]);
                (void)cudaEventElapsedTime(&ms_gate_scalar, q2_prof[1], q2_prof[2]);
                (void)cudaEventElapsedTime(&ms_gate_wmma, q2_prof[2], q2_prof[3]);
                (void)cudaEventElapsedTime(&ms_down_scalar, q2_prof[3], q2_prof[4]);
                (void)cudaEventElapsedTime(&ms_down_wmma, q2_prof[4], q2_prof[5]);
                (void)cudaEventElapsedTime(&ms_sum, q2_prof[5], q2_prof[6]);
                (void)cudaEventElapsedTime(&ms_total, q2_prof[0], q2_prof[6]);
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "MoE q2 expert profile tokens=%u pairs=%u hot_gate=%u/%u tiles=%u hot_down=%u/%u tiles=%u med_gate=%u/%u med_down=%u/%u f16_low=%u/%u f16_hot=%u/%u route_nz=%u max=%u lt12=%u/%u m12_27=%u/%u c28_63=%u/%u c64_127=%u/%u c128_255=%u/%u ge256=%u/%u direct=%u slot=%u f16_mid_all=%u f16_all=%u bucket=%.3f gate_scalar=%.3f gate_wmma=%.3f down_scalar=%.3f down_wmma=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, wmma_gate_hot_count, wmma_gate_hot_max, wmma_gate_tile_count,
                        wmma_down_hot_count, wmma_down_hot_max, wmma_down_tile_count,
                        wmma_gate_medium_count, wmma_gate_medium_max, wmma_down_medium_count, wmma_down_medium_max,
                        wmma_f16_low_count, wmma_f16_low_max, wmma_f16_hot_count, wmma_f16_hot_max,
                        route_nz, route_max, route_lt12_e, route_lt12_p,
                        route_m12_27_e, route_m12_27_p,
                        route_28_63_e, route_28_63_p, route_64_127_e, route_64_127_p,
                        route_128_255_e, route_128_255_p, route_ge256_e, route_ge256_p,
                        (uint32_t)moe_wmma_direct_sum, (uint32_t)moe_slot_partial,
                        (uint32_t)moe_wmma_f16_mid_all, (uint32_t)moe_wmma_f16_down_all,
                        ms_bucket, ms_gate_scalar, ms_gate_wmma,
                        ms_down_scalar, ms_down_wmma, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(q2_prof[i]);
        }
        return ok;
    }

    if (q2k_path && n_expert == 6u &&
        getenv("DS4_CUDA_NO_OLDHIP_MOE_Q2_ROWS") == NULL) {
        uint32_t rows_per_block = cuda_parse_u32_env_alias("DS4_CUDA_MOE_DECODE_RPB", "DS4_HIP_MOE_DECODE_RPB", 8u, 1u, 32u);
        if (rows_per_block != 1u && rows_per_block != 2u && rows_per_block != 4u &&
            rows_per_block != 8u && rows_per_block != 16u && rows_per_block != 32u) rows_per_block = 8u;
        const uint32_t threads = rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        dim3 gate_grid((expert_mid_dim + rows_per_block - 1u) / rows_per_block, n_tokens * n_expert, 1);
        moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, threads>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp,
                store_gate_up);
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch")) return 0;
        dim3 down_grid((out_dim + rows_per_block - 1u) / rows_per_block, n_tokens, 1);
        moe_down_q2K_sum_rows_w32_kernel<<<down_grid, threads>>>(
                (float *)out->ptr,
                down_w,
                (const float *)mid->ptr,
                (const int32_t *)selected->ptr,
                n_tokens,
                expert_mid_dim,
                out_dim,
                down_expert_bytes,
                down_row_bytes);
        return cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows down launch");
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_expert, clamp, x, 1);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t n_tokens) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_expert, clamp, x, n_tokens);
}
extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (getenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED") == NULL) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        const int use_fused = n_rows == 1 ||
                              cuda_env_flag_any3("DS4_CUDA_HC_SPLIT_NORM_BATCH_FUSED",
                                                 "DS4_HIP_HC_SPLIT_NORM_BATCH_FUSED",
                                                 NULL);
        if (use_fused) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                    3ull * sizeof(float), "hc_scale");
            const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                    mix_bytes, "hc_base");
            const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            if (n_rows == 1 &&
                getenv("DS4_CUDA_OLDHIP_HC_SPLIT_NORM") != NULL &&
                getenv("DS4_CUDA_NO_OLDHIP_HC_SPLIT_NORM") == NULL &&
                cuda_offset_in_env_range(norm_weight_offset,
                                         "DS4_CUDA_OLDHIP_HC_SPLIT_NORM_OFFSETS",
                                         "DS4_CUDA_OLDHIP_HC_SPLIT_NORM_MIN_OFFSET",
                                         "DS4_CUDA_OLDHIP_HC_SPLIT_NORM_MAX_OFFSET")) {
                hc_split4_weighted_sum_norm_oldhip_kernel<<<(uint32_t)n_rows, 256>>>(
                        (float *)out->ptr,
                        (float *)norm_out->ptr,
                        (float *)split->ptr,
                        (const float *)mix->ptr,
                        (const float *)residual_hc->ptr,
                        scale,
                        base,
                        norm_w,
                        n_embd, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
                return cuda_ok(cudaGetLastError(), "hc split weighted sum norm oldhip launch");
            }
            hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
                    (float *)out->ptr,
                    (float *)norm_out->ptr,
                    (float *)split->ptr,
                    (const float *)mix->ptr,
                    (const float *)residual_hc->ptr,
                    scale,
                    base,
                    norm_w,
                    n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    if (block_out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)split->ptr,
                                                    n_embd,
                                                    n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_split4 launch");
    }
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}
extern "C" int ds4_gpu_hc_expand_split_half_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out_h || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    if (block_out_h->bytes < (uint64_t)n_tokens * n_embd * sizeof(__half)) return 0;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_half_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                         (const __half *)block_out_h->ptr,
                                                         (const float *)residual_hc->ptr,
                                                         (const float *)split->ptr,
                                                         n_embd,
                                                         n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_split_half4 launch");
    }
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_half_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                         (const __half *)block_out_h->ptr,
                                                         (const float *)residual_hc->ptr,
                                                         base + n_hc,
                                                         base + 2u * n_hc,
                                                         n_embd, n_hc, n_tokens,
                                                         mix_hc, mix_hc);
    return cuda_ok(cudaGetLastError(), "hc_expand_split_half launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    if (block_out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        block_add->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_add_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                        (const float *)block_out->ptr,
                                                        (const float *)block_add->ptr,
                                                        (const float *)residual_hc->ptr,
                                                        (const float *)split->ptr,
                                                        n_embd,
                                                        n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_add_split4 launch");
    }
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add_h, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add_h || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    if (block_out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        block_add_h->bytes < (uint64_t)n_tokens * n_embd * sizeof(__half)) return 0;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_add_half_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                             (const float *)block_out->ptr,
                                                             (const __half *)block_add_h->ptr,
                                                             (const float *)residual_hc->ptr,
                                                             (const float *)split->ptr,
                                                             n_embd,
                                                             n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_add_split_half4 launch");
    }
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_add_half_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                             (const float *)block_out->ptr,
                                                             (const __half *)block_add_h->ptr,
                                                             (const float *)residual_hc->ptr,
                                                             base + n_hc,
                                                             base + 2u * n_hc,
                                                             n_embd, n_hc, n_tokens,
                                                             mix_hc, mix_hc);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split_half_add launch");
}
extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}
