static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm_hipblaslt.cuh"
#endif
static int g_quality_mode;

enum {
    DS4_ROCM_N_EXPERT = 256u,
    DS4_ROCM_N_EXPERT_USED = 6u,
    DS4_ROCM_COMPRESSOR_MAX_RATIO = 128u
};
#define DS4_ROCM_EXPERT_WEIGHT_SCALE 1.5f
#define DS4_ROCM_EXPERT_WEIGHT_SCALE_TOL 1.0e-6f

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_moe_resident_layer {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_bytes;
    uint64_t down_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t resident_bytes;
};

struct cuda_moe_expert_cache_entry {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t expert;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t bytes;
    uint64_t last_use;
};

struct cuda_moe_layer_expert_cache {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t capacity;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t bytes;
    uint64_t last_use;
    std::vector<uint32_t> expert_for_slot;
    std::vector<uint64_t> last_use_for_slot;
};

struct cuda_model_image {
    const void *host_base;
    uint64_t size;
    char *device_ptr;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f16_transpose_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::vector<cuda_model_image> g_model_images;
static std::vector<cuda_moe_resident_layer> g_moe_resident_layers;
static std::vector<cuda_moe_expert_cache_entry> g_moe_expert_cache;
static std::vector<cuda_moe_layer_expert_cache> g_moe_layer_expert_cache;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f16_transpose_range> g_q8_f16_transpose_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_transpose_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_moe_resident_bytes;
static uint64_t g_moe_expert_cache_bytes;
static uint64_t g_moe_layer_expert_cache_bytes;
static uint64_t g_moe_expert_cache_tick;
static uint64_t g_q8_f16_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static char *g_moe_sparse_gate;
static char *g_moe_sparse_up;
static char *g_moe_sparse_down;
static uint64_t g_moe_sparse_gate_bytes;
static uint64_t g_moe_sparse_up_bytes;
static uint64_t g_moe_sparse_down_bytes;
static char *g_moe_compact_gate;
static char *g_moe_compact_up;
static char *g_moe_compact_down;
static uint64_t g_moe_compact_gate_bytes;
static uint64_t g_moe_compact_up_bytes;
static uint64_t g_moe_compact_down_bytes;
static int32_t *g_moe_compact_selected;
static uint64_t g_moe_compact_selected_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;

static int cuda_ok(cudaError_t err, const char *what);

static int cuda_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int cuda_u64_mul3_checked(uint64_t a, uint64_t b, uint64_t c, uint64_t *out) {
    uint64_t tmp = 0;
    return cuda_u64_mul_checked(a, b, &tmp) && cuda_u64_mul_checked(tmp, c, out);
}

static int cuda_model_range_fits(uint64_t model_size, uint64_t offset, uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static int cuda_tensor_has_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static int cuda_tensor_has_elems(const ds4_gpu_tensor *t, uint64_t elems, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul_checked(elems, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems2(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul3_checked(a, b, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems3(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t c, uint64_t elem_size) {
    uint64_t ab = 0, elems = 0, bytes = 0;
    return cuda_u64_mul_checked(a, b, &ab) &&
           cuda_u64_mul_checked(ab, c, &elems) &&
           cuda_u64_mul_checked(elems, elem_size, &bytes) &&
           cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_f32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(float));
}

static int cuda_tensor_has_i32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(int32_t));
}

static int cuda_tensor_has_f16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(__half));
}

static int cuda_tensor_has_u16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(uint16_t));
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static void cuda_model_range_release_all(void);
static void cuda_moe_layer_expert_cache_release(void);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f16_transpose_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void cuda_shared_gate_up_async_cleanup(void);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_ROCM_ATTENTION_SCORE_CAP - DS4_ROCM_ATTENTION_RAW_SCORE_CAP;
}

static int cuda_model_image_find(const void *model_map) {
    if (!model_map) return -1;
    for (size_t i = 0; i < g_model_images.size(); i++) {
        if (g_model_images[i].host_base == model_map) return (int)i;
    }
    return -1;
}

static const char *cuda_model_image_ptr(const void *model_map, uint64_t offset) {
    const int idx = cuda_model_image_find(model_map);
    if (idx < 0) return NULL;
    const cuda_model_image &img = g_model_images[(size_t)idx];
    if (offset > img.size) return NULL;
    return img.device_ptr + offset;
}

static int cuda_model_image_owned(const void *model_map) {
    return cuda_model_image_find(model_map) >= 0;
}

static uint64_t cuda_model_image_bytes(void) {
    uint64_t bytes = 0;
    for (const cuda_model_image &img : g_model_images) bytes += img.size;
    return bytes;
}

static void cuda_model_image_release_all(void) {
    for (const cuda_model_image &img : g_model_images) {
        if (img.device_ptr) (void)cudaFree(img.device_ptr);
    }
    g_model_images.clear();
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    const char *owned = cuda_model_image_ptr(model_map, offset);
    if (owned) return owned;
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_copy_uncached(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const char *src = (const char *)model_map + offset;
    err = cudaMemcpy(dev, src, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return NULL;
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    if (cuda_model_image_owned(model_map)) return cuda_model_ptr(model_map, offset);

    if (model_map != g_model_host_base) {
        return cuda_model_range_copy_uncached(model_map, offset, bytes, what);
    }

    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
    if (fd_ptr) return fd_ptr;

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported && model_map == g_model_host_base) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                return dev_ptr;
            }
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    void *dev = NULL;
    err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (cuda_model_image_owned(model_map)) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_transpose_ranges.clear();
    g_q8_f16_transpose_by_offset.clear();
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static int cuda_env_present(const char *env) {
    if (env != NULL) return env[0] != '\0' && strcmp(env, "0") != 0;
    return 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_parse_u64_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    return (uint64_t)v;
}

static int cuda_env_flag(const char *name) {
    const char *env = name ? getenv(name) : NULL;
    return env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
}

static int cuda_env_flag_any3(const char *a, const char *b, const char *c) {
    return (a && cuda_env_flag(a)) ||
           (b && cuda_env_flag(b)) ||
           (c && cuda_env_flag(c));
}

static uint32_t cuda_parse_u32_env_alias(const char *primary, const char *alias,
                                         uint32_t def, uint32_t min_v, uint32_t max_v) {
    int have = 0;
    uint64_t v = cuda_parse_u64_env(primary, &have);
    if (!have && alias) v = cuda_parse_u64_env(alias, &have);
    if (!have) return def;
    if (v < min_v) v = min_v;
    if (v > max_v) v = max_v;
    return (uint32_t)v;
}

static uint32_t cuda_rows_per_block_or_default(uint32_t v, uint32_t def) {
    return (v == 1u || v == 2u || v == 4u || v == 8u || v == 16u || v == 32u) ? v : def;
}

struct ds4_rocm_runtime_config {
    int initialized;
    int q8_prequant_decode;
    int disable_splitk_attn_out_low;
    int disable_shared_gate_up_fused_w32;
    int attention_output_cublas_all;
    int shared_down_cublas;
    int graph_dump;
    uint32_t q8_decode_rpb;
    uint32_t q8_hc_decode_rpb;
    uint32_t attn_out_low_decode_rpb;
    uint32_t moe_decode_rpb;
    int oldhip_attention_decode;
};

static ds4_rocm_runtime_config g_rocm_cfg;

static const ds4_rocm_runtime_config *cuda_runtime_config(void) {
    if (!g_rocm_cfg.initialized) {
        g_rocm_cfg.q8_prequant_decode = !g_quality_mode;
        g_rocm_cfg.disable_splitk_attn_out_low = !g_quality_mode;
        g_rocm_cfg.disable_shared_gate_up_fused_w32 = !g_quality_mode;
        g_rocm_cfg.attention_output_cublas_all = !g_quality_mode;
        g_rocm_cfg.shared_down_cublas = !g_quality_mode;
        g_rocm_cfg.graph_dump = cuda_env_present(getenv("DS4_METAL_GRAPH_DUMP_PREFIX"));
        g_rocm_cfg.q8_decode_rpb = g_quality_mode ? 8u : 1u;
        g_rocm_cfg.q8_hc_decode_rpb = g_quality_mode ? 8u : 16u;
        g_rocm_cfg.attn_out_low_decode_rpb = g_quality_mode ? 8u : 32u;
        g_rocm_cfg.moe_decode_rpb = g_quality_mode ? 8u : 1u;
        g_rocm_cfg.oldhip_attention_decode = !g_quality_mode;
        g_rocm_cfg.initialized = 1;
    }
    return &g_rocm_cfg;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        !cuda_runtime_config()->attention_output_cublas_all) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static const __half *cuda_q8_f16_transpose_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_transpose_by_offset.find(offset);
    if (exact != g_q8_f16_transpose_by_offset.end()) {
        const cuda_q8_f16_transpose_range &r = g_q8_f16_transpose_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;
    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;
    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 transpose cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("transpose allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_transpose_kernel<<<(n + 255u) / 256u, 256>>>(dev,
                                                                     (const unsigned char *)q8,
                                                                     in_dim,
                                                                     out_dim,
                                                                     blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 transpose dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("transpose launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_transpose_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_transpose_by_offset[offset] = g_q8_f16_transpose_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static uint32_t cuda_prefill_warmup_tokens(void) {
    uint32_t n_tok = 2048u;
    const char *chunk_env = getenv("DS4_METAL_PREFILL_CHUNK");
    if (chunk_env && chunk_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(chunk_env, &end, 10);
        if (end != chunk_env && *end == '\0' && v > 0 && v <= 4096u) n_tok = (uint32_t)v;
    }
    return n_tok;
}

static void cuda_q8_f16_warmup_attention_output_a_gemm(const __half *out_a_f16,
                                                       uint64_t group_dim,
                                                       uint64_t rank,
                                                       uint32_t n_groups) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_a_f16 || group_dim == 0 || rank == 0 || n_groups == 0) return;
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (!cfg->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t heads_h_count = (uint64_t)n_groups * n_tok * group_dim;
    const uint64_t low_h_count = (uint64_t)n_groups * n_tok * rank;
    const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
    const uint64_t low_h_off = (heads_h_bytes + 255ull) & ~255ull;
    if (low_h_count > (UINT64_MAX - low_h_off) / sizeof(__half)) return;
    void *tmp = cuda_tmp_alloc(low_h_off + low_h_count * sizeof(__half), "attention output a warmup");
    if (!tmp) return;
    __half *heads_h = (__half *)tmp;
    __half *low_h = (__half *)((char *)tmp + low_h_off);
    if (cudaMemset(heads_h, 0, (size_t)heads_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   (int)rank,
                                                   (int)n_tok,
                                                   (int)group_dim,
                                                   &alpha,
                                                   out_a_f16,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)rank * (long long)group_dim,
                                                   heads_h,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)n_tok * (long long)group_dim,
                                                   &beta,
                                                   low_h,
                                                   CUDA_R_16F,
                                                   (int)(n_groups * rank),
                                                   (long long)rank,
                                                   (int)n_groups,
                                                   CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static void cuda_q8_f16_warmup_attention_output_b_gemm(const __half *out_b_f16_t,
                                                       uint64_t low_dim,
                                                       uint64_t out_dim) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_b_f16_t || low_dim == 0 || out_dim == 0) return;
    if (!cuda_runtime_config()->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t low_h_count = (uint64_t)n_tok * low_dim;
    const uint64_t out_count = (uint64_t)n_tok * out_dim;
    const uint64_t low_h_bytes = low_h_count * sizeof(__half);
    const uint64_t out_off = (low_h_bytes + 255ull) & ~255ull;
    if (out_count > (UINT64_MAX - out_off) / sizeof(float)) return;
    void *tmp = cuda_tmp_alloc(out_off + out_count * sizeof(float), "attention output b warmup");
    if (!tmp) return;
    __half *low_h = (__half *)tmp;
    float *out = (float *)((char *)tmp + out_off);
    if (cudaMemset(low_h, 0, (size_t)low_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)low_dim,
                                     &alpha,
                                     out_b_f16_t,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     low_h,
                                     CUDA_R_16F,
                                     (int)low_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r" DS4_GPU_LOG_PREFIX "loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    return 64ull * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (!model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    uint64_t alloc_bytes = bytes;
    if (g_model_direct_align > 1u) {
        const uint64_t pad = g_model_direct_align - 1u;
        if (alloc_bytes > UINT64_MAX - pad) return 0;
        alloc_bytes += pad;
    }
    if (alloc_bytes > (uint64_t)SIZE_MAX) return 0;
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static void cuda_model_range_evict_for_bytes(uint64_t bytes, const char *what) {
    if (!cuda_env_flag("DS4_CUDA_WEIGHT_CACHE_EVICT") || bytes == 0) return;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (limit == UINT64_MAX || bytes > limit) return;
    if (g_model_range_bytes <= limit - bytes) return;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "evicting %.2f GiB model range cache for %s group %.2f MiB (limit %.2f GiB)\n",
                (double)g_model_range_bytes / 1073741824.0,
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)limit / 1073741824.0);
    }
    (void)cudaDeviceSynchronize();
    cuda_model_range_release_all();
}

static int cuda_moe_sparse_alloc_one(char **ptr, uint64_t *have, uint64_t need, const char *what) {
    if (need == 0) return 0;
    if (*ptr && *have >= need) return 1;
    (void)cudaDeviceSynchronize();
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *have = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)need);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "sparse MoE alloc failed for %s (%.2f GiB): %s\n",
                what ? what : "weights",
                (double)need / 1073741824.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (char *)dev;
    *have = need;
    return 1;
}

static int cuda_moe_sparse_buffers_prepare(uint64_t gate_bytes, uint64_t down_bytes) {
    return cuda_moe_sparse_alloc_one(&g_moe_sparse_gate, &g_moe_sparse_gate_bytes, gate_bytes, "gate") &&
           cuda_moe_sparse_alloc_one(&g_moe_sparse_up, &g_moe_sparse_up_bytes, gate_bytes, "up") &&
           cuda_moe_sparse_alloc_one(&g_moe_sparse_down, &g_moe_sparse_down_bytes, down_bytes, "down");
}

static void cuda_moe_sparse_buffers_release(void) {
    if (g_moe_sparse_gate) (void)cudaFree(g_moe_sparse_gate);
    if (g_moe_sparse_up) (void)cudaFree(g_moe_sparse_up);
    if (g_moe_sparse_down) (void)cudaFree(g_moe_sparse_down);
    g_moe_sparse_gate = NULL;
    g_moe_sparse_up = NULL;
    g_moe_sparse_down = NULL;
    g_moe_sparse_gate_bytes = 0;
    g_moe_sparse_up_bytes = 0;
    g_moe_sparse_down_bytes = 0;
}

static int cuda_moe_compact_buffers_prepare(uint64_t gate_bytes, uint64_t down_bytes) {
    return cuda_moe_sparse_alloc_one(&g_moe_compact_gate, &g_moe_compact_gate_bytes, gate_bytes, "compact_gate") &&
           cuda_moe_sparse_alloc_one(&g_moe_compact_up, &g_moe_compact_up_bytes, gate_bytes, "compact_up") &&
           cuda_moe_sparse_alloc_one(&g_moe_compact_down, &g_moe_compact_down_bytes, down_bytes, "compact_down");
}

static int cuda_moe_compact_selected_prepare(uint64_t bytes) {
    if (bytes == 0) return 0;
    if (g_moe_compact_selected && g_moe_compact_selected_bytes >= bytes) return 1;
    (void)cudaDeviceSynchronize();
    if (g_moe_compact_selected) {
        (void)cudaFree(g_moe_compact_selected);
        g_moe_compact_selected = NULL;
        g_moe_compact_selected_bytes = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compact MoE selected alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_moe_compact_selected = (int32_t *)dev;
    g_moe_compact_selected_bytes = bytes;
    return 1;
}

static void cuda_moe_compact_buffers_release(void) {
    if (g_moe_compact_gate) (void)cudaFree(g_moe_compact_gate);
    if (g_moe_compact_up) (void)cudaFree(g_moe_compact_up);
    if (g_moe_compact_down) (void)cudaFree(g_moe_compact_down);
    if (g_moe_compact_selected) (void)cudaFree(g_moe_compact_selected);
    g_moe_compact_gate = NULL;
    g_moe_compact_up = NULL;
    g_moe_compact_down = NULL;
    g_moe_compact_selected = NULL;
    g_moe_compact_gate_bytes = 0;
    g_moe_compact_up_bytes = 0;
    g_moe_compact_down_bytes = 0;
    g_moe_compact_selected_bytes = 0;
}

static int cuda_model_copy_range_to_device_buffer(
        const void *model_map,
        uint64_t model_size,
        char *dst,
        uint64_t src_offset,
        uint64_t bytes,
        const char *what) {
    if (!dst || bytes == 0) return bytes == 0;
    if (src_offset > model_size || bytes > model_size - src_offset) return 0;

    if (g_model_fd < 0) {
        cudaError_t err = cudaMemcpy(dst, (const char *)model_map + src_offset,
                                     (size_t)bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "sparse MoE copy failed for %s %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, src_offset, bytes);
        return 1;
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return 0;

    uint64_t copied = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[0], g_model_stage_bytes,
                                   src_offset + copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "sparse MoE read failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    (double)bytes / 1048576.0,
                    strerror(errno));
            return 0;
        }
        cudaError_t err = cudaMemcpy(dst + copied, payload, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "sparse MoE upload failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(src_offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size, src_offset + copied, n);
        copied += n;
    }
    return 1;
}

static uint64_t cuda_moe_expert_cache_budget_bytes(void) {
    int have = 0;
    uint64_t gb = cuda_parse_u64_env("DS4_CUDA_MOE_EXPERT_CACHE_GB", &have);
    if (!have) gb = cuda_parse_u64_env("DS4_HIP_MOE_EXPERT_CACHE_GB", &have);
    if (have) {
        if (gb > UINT64_MAX / 1073741824ull) return UINT64_MAX;
        return gb * 1073741824ull;
    }
    uint64_t bytes = cuda_parse_mib_env("DS4_CUDA_MOE_EXPERT_CACHE_MB", &have);
    if (!have) bytes = cuda_parse_mib_env("DS4_HIP_MOE_EXPERT_CACHE_MB", &have);
    return have ? bytes : 0;
}

static void cuda_moe_expert_cache_release(void) {
    cuda_moe_layer_expert_cache_release();
    for (const cuda_moe_expert_cache_entry &e : g_moe_expert_cache) {
        if (e.gate_ptr) (void)cudaFree(e.gate_ptr);
        if (e.up_ptr) (void)cudaFree(e.up_ptr);
        if (e.down_ptr) (void)cudaFree(e.down_ptr);
    }
    g_moe_expert_cache.clear();
    g_moe_expert_cache_bytes = 0;
    g_moe_expert_cache_tick = 0;
    cuda_moe_compact_buffers_release();
}

static int cuda_moe_expert_cache_find_index(
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert,
        size_t *idx_out) {
    for (size_t i = 0; i < g_moe_expert_cache.size(); i++) {
        const cuda_moe_expert_cache_entry &e = g_moe_expert_cache[i];
        if (e.gate_offset == gate_offset && e.up_offset == up_offset && e.down_offset == down_offset &&
            e.gate_expert_bytes == gate_expert_bytes && e.down_expert_bytes == down_expert_bytes &&
            e.expert == expert && e.gate_ptr && e.up_ptr && e.down_ptr) {
            if (idx_out) *idx_out = i;
            return 1;
        }
    }
    return 0;
}

static void cuda_moe_expert_cache_evict_index(size_t idx) {
    if (idx >= g_moe_expert_cache.size()) return;
    cuda_moe_expert_cache_entry &e = g_moe_expert_cache[idx];
    if (e.gate_ptr) (void)cudaFree(e.gate_ptr);
    if (e.up_ptr) (void)cudaFree(e.up_ptr);
    if (e.down_ptr) (void)cudaFree(e.down_ptr);
    if (g_moe_expert_cache_bytes >= e.bytes) g_moe_expert_cache_bytes -= e.bytes;
    else g_moe_expert_cache_bytes = 0;
    g_moe_expert_cache.erase(g_moe_expert_cache.begin() + (ptrdiff_t)idx);
}

static int cuda_moe_expert_cache_evict_for(uint64_t bytes, uint64_t budget) {
    if (bytes > budget) return 0;
    if (g_moe_expert_cache_bytes <= budget - bytes) return 1;
    (void)cudaDeviceSynchronize();
    while (!g_moe_expert_cache.empty() && g_moe_expert_cache_bytes > budget - bytes) {
        size_t victim = 0;
        uint64_t best_use = g_moe_expert_cache[0].last_use;
        for (size_t i = 1; i < g_moe_expert_cache.size(); i++) {
            if (g_moe_expert_cache[i].last_use < best_use) {
                best_use = g_moe_expert_cache[i].last_use;
                victim = i;
            }
        }
        cuda_moe_expert_cache_evict_index(victim);
    }
    return g_moe_expert_cache_bytes <= budget - bytes;
}

static int cuda_moe_expert_cache_add(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert,
        size_t *idx_out) {
    const uint64_t budget = cuda_moe_expert_cache_budget_bytes();
    if (budget == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint64_t entry_bytes = gate_expert_bytes;
    if (entry_bytes > UINT64_MAX - gate_expert_bytes) return 0;
    entry_bytes += gate_expert_bytes;
    if (entry_bytes > UINT64_MAX - down_expert_bytes) return 0;
    entry_bytes += down_expert_bytes;
    if (!cuda_moe_expert_cache_evict_for(entry_bytes, budget)) return 0;

    char *gate_ptr = NULL;
    char *up_ptr = NULL;
    char *down_ptr = NULL;
    cudaError_t err = cudaMalloc((void **)&gate_ptr, (size_t)gate_expert_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&up_ptr, (size_t)gate_expert_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&down_ptr, (size_t)down_expert_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "MoE expert cache alloc skipped (expert %.2f MiB): %s\n",
                (double)entry_bytes / 1048576.0,
                cudaGetErrorString(err));
        if (gate_ptr) (void)cudaFree(gate_ptr);
        if (up_ptr) (void)cudaFree(up_ptr);
        if (down_ptr) (void)cudaFree(down_ptr);
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t gate_src = gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src = up_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src = down_offset + (uint64_t)expert * down_expert_bytes;
    if (!cuda_model_copy_range_to_device_buffer(model_map, model_size, gate_ptr, gate_src, gate_expert_bytes, "expert_cache_gate") ||
        !cuda_model_copy_range_to_device_buffer(model_map, model_size, up_ptr, up_src, gate_expert_bytes, "expert_cache_up") ||
        !cuda_model_copy_range_to_device_buffer(model_map, model_size, down_ptr, down_src, down_expert_bytes, "expert_cache_down")) {
        (void)cudaFree(gate_ptr);
        (void)cudaFree(up_ptr);
        (void)cudaFree(down_ptr);
        return 0;
    }

    cuda_moe_expert_cache_entry ce;
    ce.gate_offset = gate_offset;
    ce.up_offset = up_offset;
    ce.down_offset = down_offset;
    ce.gate_expert_bytes = gate_expert_bytes;
    ce.down_expert_bytes = down_expert_bytes;
    ce.expert = expert;
    ce.gate_ptr = gate_ptr;
    ce.up_ptr = up_ptr;
    ce.down_ptr = down_ptr;
    ce.bytes = entry_bytes;
    ce.last_use = ++g_moe_expert_cache_tick;
    g_moe_expert_cache.push_back(ce);
    g_moe_expert_cache_bytes += entry_bytes;
    if (idx_out) *idx_out = g_moe_expert_cache.size() - 1u;
    if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE", "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "MoE expert cached expert=%u entry=%.2f MiB total=%.2f/%.2f GiB\n",
                expert,
                (double)entry_bytes / 1048576.0,
                (double)g_moe_expert_cache_bytes / 1073741824.0,
                (double)budget / 1073741824.0);
    }
    return 1;
}

static void cuda_moe_layer_expert_cache_release(void) {
    for (const cuda_moe_layer_expert_cache &c : g_moe_layer_expert_cache) {
        if (c.gate_ptr) (void)cudaFree(c.gate_ptr);
        if (c.up_ptr) (void)cudaFree(c.up_ptr);
        if (c.down_ptr) (void)cudaFree(c.down_ptr);
    }
    g_moe_layer_expert_cache.clear();
    g_moe_layer_expert_cache_bytes = 0;
}

static int cuda_moe_layer_expert_cache_find(
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        size_t *idx_out) {
    for (size_t i = 0; i < g_moe_layer_expert_cache.size(); i++) {
        const cuda_moe_layer_expert_cache &c = g_moe_layer_expert_cache[i];
        if (c.gate_offset == gate_offset && c.up_offset == up_offset && c.down_offset == down_offset &&
            c.gate_expert_bytes == gate_expert_bytes && c.down_expert_bytes == down_expert_bytes &&
            c.gate_ptr && c.up_ptr && c.down_ptr) {
            if (idx_out) *idx_out = i;
            return 1;
        }
    }
    return 0;
}

static void cuda_moe_layer_expert_cache_evict_index(size_t idx) {
    if (idx >= g_moe_layer_expert_cache.size()) return;
    cuda_moe_layer_expert_cache &c = g_moe_layer_expert_cache[idx];
    if (c.gate_ptr) (void)cudaFree(c.gate_ptr);
    if (c.up_ptr) (void)cudaFree(c.up_ptr);
    if (c.down_ptr) (void)cudaFree(c.down_ptr);
    if (g_moe_layer_expert_cache_bytes >= c.bytes) g_moe_layer_expert_cache_bytes -= c.bytes;
    else g_moe_layer_expert_cache_bytes = 0;
    g_moe_layer_expert_cache.erase(g_moe_layer_expert_cache.begin() + (ptrdiff_t)idx);
}

static int cuda_moe_layer_expert_cache_evict_for(uint64_t bytes, uint64_t budget) {
    if (bytes > budget) return 0;
    if (g_moe_layer_expert_cache_bytes <= budget - bytes) return 1;

    /*
     * Decode visits MoE layers in a fixed cyclic order.  Plain LRU is a worst
     * case when the layer-cache budget cannot hold every non-resident layer:
     * layer 10 evicts layer 9, layer 11 evicts layer 10, ..., and the next
     * token starts by missing layer 9 again.  Keep the already-admitted layer
     * caches stable by default and let uncached layers fall through to the
     * one-shot sparse path.  This turns partial budgets into useful pinned
     * layer coverage instead of a full-cache zero-hit scan.
     *
     * Set DS4_CUDA_MOE_LAYER_CACHE_EVICT=1 or DS4_HIP_MOE_LAYER_CACHE_EVICT=1
     * to restore the old LRU eviction behaviour for experiments.
     */
    if (!cuda_env_flag_any3("DS4_CUDA_MOE_LAYER_CACHE_EVICT",
                            "DS4_HIP_MOE_LAYER_CACHE_EVICT", NULL)) {
        if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE",
                               "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "MoE layer expert cache full: preserving %.2f/%.2f GiB, bypassing %.2f MiB new layer cache\n",
                    (double)g_moe_layer_expert_cache_bytes / 1073741824.0,
                    (double)budget / 1073741824.0,
                    (double)bytes / 1048576.0);
        }
        return 0;
    }

    (void)cudaDeviceSynchronize();
    while (!g_moe_layer_expert_cache.empty() && g_moe_layer_expert_cache_bytes > budget - bytes) {
        size_t victim = 0;
        uint64_t best_use = g_moe_layer_expert_cache[0].last_use;
        for (size_t i = 1; i < g_moe_layer_expert_cache.size(); i++) {
            if (g_moe_layer_expert_cache[i].last_use < best_use) {
                best_use = g_moe_layer_expert_cache[i].last_use;
                victim = i;
            }
        }
        cuda_moe_layer_expert_cache_evict_index(victim);
    }
    return g_moe_layer_expert_cache_bytes <= budget - bytes;
}

static int cuda_moe_layer_expert_cache_get(
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t min_capacity,
        size_t *idx_out) {
    if (!idx_out) return 0;
    if (cuda_moe_layer_expert_cache_find(gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes,
                                         idx_out)) return 1;
    const uint64_t budget = cuda_moe_expert_cache_budget_bytes();
    if (budget == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint32_t capacity = cuda_parse_u32_env_alias(
            "DS4_CUDA_MOE_EXPERT_CACHE_SLOTS", "DS4_HIP_MOE_EXPERT_CACHE_SLOTS",
            16u, 1u, DS4_ROCM_N_EXPERT);
    const uint32_t target_layers = cuda_parse_u32_env_alias(
            "DS4_CUDA_MOE_EXPERT_CACHE_TARGET_LAYERS", "DS4_HIP_MOE_EXPERT_CACHE_TARGET_LAYERS",
            0u, 0u, 4096u);
    if (target_layers != 0u) {
        uint64_t per_slot_bytes = gate_expert_bytes;
        if (per_slot_bytes <= UINT64_MAX - gate_expert_bytes) per_slot_bytes += gate_expert_bytes;
        else per_slot_bytes = 0;
        if (per_slot_bytes != 0 && per_slot_bytes <= UINT64_MAX - down_expert_bytes) per_slot_bytes += down_expert_bytes;
        else per_slot_bytes = 0;
        if (per_slot_bytes != 0) {
            uint64_t target_layer_bytes = budget / (uint64_t)target_layers;
            uint32_t target_capacity = (uint32_t)(target_layer_bytes / per_slot_bytes);
            if (target_capacity > DS4_ROCM_N_EXPERT) target_capacity = DS4_ROCM_N_EXPERT;
            if (target_capacity != 0u && capacity > target_capacity) {
                if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE",
                                       "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX "MoE layer expert cache auto-slots: requested=%u target_layers=%u budget=%.2f GiB -> slots=%u\n",
                            capacity,
                            target_layers,
                            (double)budget / 1073741824.0,
                            target_capacity);
                }
                capacity = target_capacity;
            }
        }
    }
    if (capacity < min_capacity) capacity = min_capacity;
    if (capacity > DS4_ROCM_N_EXPERT) capacity = DS4_ROCM_N_EXPERT;

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked((uint64_t)capacity, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked((uint64_t)capacity, down_expert_bytes, &down_bytes)) return 0;
    uint64_t cache_bytes = gate_bytes;
    if (cache_bytes > UINT64_MAX - gate_bytes) return 0;
    cache_bytes += gate_bytes;
    if (cache_bytes > UINT64_MAX - down_bytes) return 0;
    cache_bytes += down_bytes;
    if (!cuda_moe_layer_expert_cache_evict_for(cache_bytes, budget)) return 0;

    char *gate_ptr = NULL;
    char *up_ptr = NULL;
    char *down_ptr = NULL;
    cudaError_t err = cudaMalloc((void **)&gate_ptr, (size_t)gate_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&up_ptr, (size_t)gate_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&down_ptr, (size_t)down_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "MoE layer expert cache alloc skipped (slots=%u %.2f MiB): %s\n",
                capacity,
                (double)cache_bytes / 1048576.0,
                cudaGetErrorString(err));
        if (gate_ptr) (void)cudaFree(gate_ptr);
        if (up_ptr) (void)cudaFree(up_ptr);
        if (down_ptr) (void)cudaFree(down_ptr);
        (void)cudaGetLastError();
        return 0;
    }

    cuda_moe_layer_expert_cache c;
    c.gate_offset = gate_offset;
    c.up_offset = up_offset;
    c.down_offset = down_offset;
    c.gate_expert_bytes = gate_expert_bytes;
    c.down_expert_bytes = down_expert_bytes;
    c.capacity = capacity;
    c.gate_ptr = gate_ptr;
    c.up_ptr = up_ptr;
    c.down_ptr = down_ptr;
    c.bytes = cache_bytes;
    c.last_use = ++g_moe_expert_cache_tick;
    c.expert_for_slot.assign(capacity, UINT32_MAX);
    c.last_use_for_slot.assign(capacity, 0u);
    g_moe_layer_expert_cache.push_back(c);
    g_moe_layer_expert_cache_bytes += cache_bytes;
    *idx_out = g_moe_layer_expert_cache.size() - 1u;
    if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE", "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "MoE layer expert cache allocated slots=%u %.2f MiB total=%.2f/%.2f GiB\n",
                capacity,
                (double)cache_bytes / 1048576.0,
                (double)g_moe_layer_expert_cache_bytes / 1073741824.0,
                (double)budget / 1073741824.0);
    }
    return 1;
}

static int cuda_model_layer_moe_expert_cache_ptrs(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert,
        const char **gate_w,
        const char **up_w,
        const char **down_w,
        ds4_gpu_tensor *remapped_selected) {
    if (cuda_moe_expert_cache_budget_bytes() == 0) return 0;
    if (!model_map || !gate_w || !up_w || !down_w || !selected || !selected->ptr || !remapped_selected) return 0;
    if (g_model_device_owned) return 0;
    if (n_tokens != 1u || n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT) return 0;
    if (!cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t))) return 0;
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    if (!cuda_model_range_fits(model_size, gate_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, down_bytes)) return 0;
    if (gate_expert_bytes > gate_bytes / DS4_ROCM_N_EXPERT ||
        down_expert_bytes > down_bytes / DS4_ROCM_N_EXPERT) return 0;

    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (pair_count == 0 || pair_count > (uint64_t)UINT32_MAX) return 0;
    std::vector<int32_t> h_selected((size_t)pair_count);
    cudaError_t err = cudaMemcpy(h_selected.data(), selected->ptr,
                                 (size_t)(pair_count * sizeof(int32_t)),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    uint8_t used[DS4_ROCM_N_EXPERT] = {0};
    uint8_t slot_for_expert[DS4_ROCM_N_EXPERT];
    memset(slot_for_expert, 0xff, sizeof(slot_for_expert));
    uint32_t experts[DS4_ROCM_N_EXPERT];
    uint32_t unique = 0;
    for (uint64_t i = 0; i < pair_count; i++) {
        const int32_t e = h_selected[(size_t)i];
        if (e < 0 || e >= (int32_t)DS4_ROCM_N_EXPERT) return 0;
        if (!used[e]) {
            used[e] = 1;
            experts[unique++] = (uint32_t)e;
        }
    }
    if (unique == 0) return 0;
    const uint32_t max_unique = cuda_parse_u32_env_alias(
            "DS4_CUDA_MOE_EXPERT_CACHE_MAX_EXPERTS", "DS4_HIP_MOE_EXPERT_CACHE_MAX_EXPERTS",
            16u, 1u, DS4_ROCM_N_EXPERT);
    if (unique > max_unique) return 0;

    size_t cache_idx = 0;
    if (!cuda_moe_layer_expert_cache_get(gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes,
                                         unique, &cache_idx)) return 0;
    cuda_moe_layer_expert_cache &cache = g_moe_layer_expert_cache[cache_idx];
    cache.last_use = ++g_moe_expert_cache_tick;

    uint32_t hits = 0;
    uint32_t misses = 0;
    uint64_t uploaded = 0;
    for (uint32_t i = 0; i < unique; i++) {
        const uint32_t expert = experts[i];
        uint32_t slot = UINT32_MAX;
        for (uint32_t s = 0; s < cache.capacity; s++) {
            if (cache.expert_for_slot[s] == expert) {
                slot = s;
                break;
            }
        }
        if (slot != UINT32_MAX) {
            hits++;
        } else {
            for (uint32_t s = 0; s < cache.capacity; s++) {
                if (cache.expert_for_slot[s] == UINT32_MAX) {
                    slot = s;
                    break;
                }
            }
            if (slot == UINT32_MAX) {
                uint64_t best_use = UINT64_MAX;
                for (uint32_t s = 0; s < cache.capacity; s++) {
                    const uint32_t old_expert = cache.expert_for_slot[s];
                    if (old_expert < DS4_ROCM_N_EXPERT && used[old_expert]) continue;
                    if (cache.last_use_for_slot[s] < best_use) {
                        best_use = cache.last_use_for_slot[s];
                        slot = s;
                    }
                }
            }
            if (slot == UINT32_MAX) return 0;

            const uint64_t gate_src = gate_offset + (uint64_t)expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + (uint64_t)expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + (uint64_t)expert * down_expert_bytes;
            char *dst_gate = cache.gate_ptr + (uint64_t)slot * gate_expert_bytes;
            char *dst_up = cache.up_ptr + (uint64_t)slot * gate_expert_bytes;
            char *dst_down = cache.down_ptr + (uint64_t)slot * down_expert_bytes;
            if (!cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_gate, gate_src, gate_expert_bytes, "layer_expert_cache_gate") ||
                !cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_up, up_src, gate_expert_bytes, "layer_expert_cache_up") ||
                !cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_down, down_src, down_expert_bytes, "layer_expert_cache_down")) {
                return 0;
            }
            cache.expert_for_slot[slot] = expert;
            misses++;
            uploaded += gate_expert_bytes + gate_expert_bytes + down_expert_bytes;
        }
        cache.last_use_for_slot[slot] = ++g_moe_expert_cache_tick;
        slot_for_expert[expert] = (uint8_t)slot;
    }

    std::vector<int32_t> h_remap((size_t)pair_count);
    for (uint64_t i = 0; i < pair_count; i++) h_remap[(size_t)i] = (int32_t)slot_for_expert[h_selected[(size_t)i]];
    const uint64_t selected_bytes = pair_count * sizeof(int32_t);
    if (!cuda_moe_compact_selected_prepare(selected_bytes)) return 0;
    err = cudaMemcpy(g_moe_compact_selected, h_remap.data(), (size_t)selected_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE", "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "MoE layer expert cache decode unique=%u hits=%u misses=%u upload=%.2f MiB layer_cache=%.2f GiB\n",
                unique, hits, misses,
                (double)uploaded / 1048576.0,
                (double)g_moe_layer_expert_cache_bytes / 1073741824.0);
    }

    *remapped_selected = *selected;
    remapped_selected->ptr = g_moe_compact_selected;
    remapped_selected->bytes = selected_bytes;
    *gate_w = cache.gate_ptr;
    *up_w = cache.up_ptr;
    *down_w = cache.down_ptr;
    return 1;
}

static int cuda_model_cached_moe_expert_layer_ptrs(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert,
        const char **gate_w,
        const char **up_w,
        const char **down_w,
        ds4_gpu_tensor *remapped_selected) {
    if (!cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_LEGACY", "DS4_HIP_MOE_EXPERT_CACHE_LEGACY", NULL)) {
        return cuda_model_layer_moe_expert_cache_ptrs(model_map, model_size,
                                                      gate_offset, up_offset, down_offset,
                                                      gate_bytes, down_bytes,
                                                      gate_expert_bytes, down_expert_bytes,
                                                      selected, n_tokens, n_expert,
                                                      gate_w, up_w, down_w,
                                                      remapped_selected);
    }
    if (cuda_moe_expert_cache_budget_bytes() == 0) return 0;
    if (!model_map || !gate_w || !up_w || !down_w || !selected || !selected->ptr || !remapped_selected) return 0;
    if (g_model_device_owned) return 0;
    if (n_tokens != 1u || n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT) return 0;
    if (!cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t))) return 0;
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    if (!cuda_model_range_fits(model_size, gate_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, down_bytes)) return 0;
    if (gate_expert_bytes > gate_bytes / DS4_ROCM_N_EXPERT ||
        down_expert_bytes > down_bytes / DS4_ROCM_N_EXPERT) return 0;

    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (pair_count == 0 || pair_count > (uint64_t)UINT32_MAX) return 0;
    std::vector<int32_t> h_selected((size_t)pair_count);
    cudaError_t err = cudaMemcpy(h_selected.data(), selected->ptr,
                                 (size_t)(pair_count * sizeof(int32_t)),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    uint8_t used[DS4_ROCM_N_EXPERT] = {0};
    uint8_t slot_for[DS4_ROCM_N_EXPERT];
    memset(slot_for, 0xff, sizeof(slot_for));
    uint32_t experts[DS4_ROCM_N_EXPERT];
    uint32_t unique = 0;
    for (uint64_t i = 0; i < pair_count; i++) {
        const int32_t e = h_selected[(size_t)i];
        if (e < 0 || e >= (int32_t)DS4_ROCM_N_EXPERT) return 0;
        if (!used[e]) {
            used[e] = 1;
            slot_for[e] = (uint8_t)unique;
            experts[unique++] = (uint32_t)e;
        }
    }
    if (unique == 0) return 0;
    const uint32_t max_unique = cuda_parse_u32_env_alias(
            "DS4_CUDA_MOE_EXPERT_CACHE_MAX_EXPERTS", "DS4_HIP_MOE_EXPERT_CACHE_MAX_EXPERTS",
            16u, 1u, DS4_ROCM_N_EXPERT);
    if (unique > max_unique) return 0;

    std::vector<int32_t> h_remap((size_t)pair_count);
    for (uint64_t i = 0; i < pair_count; i++) h_remap[(size_t)i] = (int32_t)slot_for[h_selected[(size_t)i]];

    uint64_t compact_gate_bytes = 0;
    uint64_t compact_down_bytes = 0;
    if (!cuda_u64_mul_checked((uint64_t)unique, gate_expert_bytes, &compact_gate_bytes) ||
        !cuda_u64_mul_checked((uint64_t)unique, down_expert_bytes, &compact_down_bytes)) return 0;
    if (!cuda_moe_compact_buffers_prepare(compact_gate_bytes, compact_down_bytes)) return 0;
    const uint64_t selected_bytes = pair_count * sizeof(int32_t);
    if (!cuda_moe_compact_selected_prepare(selected_bytes)) return 0;
    err = cudaMemcpy(g_moe_compact_selected, h_remap.data(), (size_t)selected_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    uint32_t hits = 0;
    uint32_t misses = 0;
    uint32_t bypass = 0;
    uint64_t uploaded = 0;
    for (uint32_t slot = 0; slot < unique; slot++) {
        const uint32_t expert = experts[slot];
        const char *cache_gate = NULL;
        const char *cache_up = NULL;
        const char *cache_down = NULL;
        size_t idx = 0;
        if (cuda_moe_expert_cache_find_index(gate_offset, up_offset, down_offset,
                                             gate_expert_bytes, down_expert_bytes,
                                             expert, &idx)) {
            cuda_moe_expert_cache_entry &ce = g_moe_expert_cache[idx];
            ce.last_use = ++g_moe_expert_cache_tick;
            cache_gate = ce.gate_ptr;
            cache_up = ce.up_ptr;
            cache_down = ce.down_ptr;
            hits++;
        } else if (cuda_moe_expert_cache_add(model_map, model_size,
                                             gate_offset, up_offset, down_offset,
                                             gate_expert_bytes, down_expert_bytes,
                                             expert, &idx)) {
            cuda_moe_expert_cache_entry &ce = g_moe_expert_cache[idx];
            cache_gate = ce.gate_ptr;
            cache_up = ce.up_ptr;
            cache_down = ce.down_ptr;
            misses++;
            uploaded += gate_expert_bytes + gate_expert_bytes + down_expert_bytes;
        }

        char *dst_gate = g_moe_compact_gate + (uint64_t)slot * gate_expert_bytes;
        char *dst_up = g_moe_compact_up + (uint64_t)slot * gate_expert_bytes;
        char *dst_down = g_moe_compact_down + (uint64_t)slot * down_expert_bytes;
        if (cache_gate && cache_up && cache_down) {
            err = cudaMemcpy(dst_gate, cache_gate, (size_t)gate_expert_bytes, cudaMemcpyDeviceToDevice);
            if (err == cudaSuccess) err = cudaMemcpy(dst_up, cache_up, (size_t)gate_expert_bytes, cudaMemcpyDeviceToDevice);
            if (err == cudaSuccess) err = cudaMemcpy(dst_down, cache_down, (size_t)down_expert_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "MoE expert cache device copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        } else {
            const uint64_t gate_src = gate_offset + (uint64_t)expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + (uint64_t)expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + (uint64_t)expert * down_expert_bytes;
            if (!cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_gate, gate_src, gate_expert_bytes, "compact_moe_gate") ||
                !cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_up, up_src, gate_expert_bytes, "compact_moe_up") ||
                !cuda_model_copy_range_to_device_buffer(model_map, model_size, dst_down, down_src, down_expert_bytes, "compact_moe_down")) {
                return 0;
            }
            bypass++;
            uploaded += gate_expert_bytes + gate_expert_bytes + down_expert_bytes;
        }
    }

    if (cuda_env_flag_any3("DS4_CUDA_MOE_EXPERT_CACHE_VERBOSE", "DS4_HIP_MOE_EXPERT_CACHE_VERBOSE", NULL)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "MoE expert cache decode unique=%u hits=%u misses=%u bypass=%u upload=%.2f MiB cache=%.2f GiB\n",
                unique, hits, misses, bypass,
                (double)uploaded / 1048576.0,
                (double)g_moe_expert_cache_bytes / 1073741824.0);
    }

    *remapped_selected = *selected;
    remapped_selected->ptr = g_moe_compact_selected;
    remapped_selected->bytes = selected_bytes;
    *gate_w = g_moe_compact_gate;
    *up_w = g_moe_compact_up;
    *down_w = g_moe_compact_down;
    return 1;
}

static uint64_t cuda_moe_resident_cache_budget_bytes(void) {
    int have = 0;
    uint64_t gb = cuda_parse_u64_env("DS4_CUDA_MOE_RESIDENT_CACHE_GB", &have);
    if (!have) gb = cuda_parse_u64_env("DS4_HIP_MOE_RESIDENT_CACHE_GB", &have);
    if (have) {
        if (gb > UINT64_MAX / 1073741824ull) return UINT64_MAX;
        return gb * 1073741824ull;
    }
    uint64_t bytes = cuda_parse_mib_env("DS4_CUDA_MOE_RESIDENT_CACHE_MB", &have);
    if (!have) bytes = cuda_parse_mib_env("DS4_HIP_MOE_RESIDENT_CACHE_MB", &have);
    return have ? bytes : 0;
}

static void cuda_moe_resident_layers_release(void) {
    for (const cuda_moe_resident_layer &r : g_moe_resident_layers) {
        if (r.gate_ptr) (void)cudaFree(r.gate_ptr);
        if (r.up_ptr) (void)cudaFree(r.up_ptr);
        if (r.down_ptr) (void)cudaFree(r.down_ptr);
    }
    g_moe_resident_layers.clear();
    g_moe_resident_bytes = 0;
}

static int cuda_model_resident_moe_layer_ptrs(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (!gate_w || !up_w || !down_w) return 0;
    for (const cuda_moe_resident_layer &r : g_moe_resident_layers) {
        if (r.gate_offset == gate_offset && r.up_offset == up_offset && r.down_offset == down_offset &&
            r.gate_bytes == gate_bytes && r.down_bytes == down_bytes &&
            r.gate_ptr && r.up_ptr && r.down_ptr) {
            *gate_w = r.gate_ptr;
            *up_w = r.up_ptr;
            *down_w = r.down_ptr;
            return 1;
        }
    }

    const uint64_t budget = cuda_moe_resident_cache_budget_bytes();
    if (budget == 0) return 0;
    if (!model_map || gate_bytes == 0 || down_bytes == 0) return 0;
    if (!cuda_model_range_fits(model_size, gate_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, down_bytes)) return 0;
    uint64_t layer_bytes = gate_bytes;
    if (layer_bytes > UINT64_MAX - gate_bytes) return 0;
    layer_bytes += gate_bytes;
    if (layer_bytes > UINT64_MAX - down_bytes) return 0;
    layer_bytes += down_bytes;
    if (layer_bytes > budget || g_moe_resident_bytes > budget - layer_bytes) return 0;

    char *gate_ptr = NULL;
    char *up_ptr = NULL;
    char *down_ptr = NULL;
    cudaError_t err = cudaMalloc((void **)&gate_ptr, (size_t)gate_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&up_ptr, (size_t)gate_bytes);
    if (err == cudaSuccess) err = cudaMalloc((void **)&down_ptr, (size_t)down_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "resident MoE allocation skipped (%.2f GiB layer): %s\n",
                (double)layer_bytes / 1073741824.0,
                cudaGetErrorString(err));
        if (gate_ptr) (void)cudaFree(gate_ptr);
        if (up_ptr) (void)cudaFree(up_ptr);
        if (down_ptr) (void)cudaFree(down_ptr);
        (void)cudaGetLastError();
        return 0;
    }

    if (!cuda_model_copy_range_to_device_buffer(model_map, model_size, gate_ptr, gate_offset, gate_bytes, "resident_moe_gate") ||
        !cuda_model_copy_range_to_device_buffer(model_map, model_size, up_ptr, up_offset, gate_bytes, "resident_moe_up") ||
        !cuda_model_copy_range_to_device_buffer(model_map, model_size, down_ptr, down_offset, down_bytes, "resident_moe_down")) {
        (void)cudaFree(gate_ptr);
        (void)cudaFree(up_ptr);
        (void)cudaFree(down_ptr);
        return 0;
    }

    g_moe_resident_layers.push_back({gate_offset, up_offset, down_offset,
                                     gate_bytes, down_bytes,
                                     gate_ptr, up_ptr, down_ptr, layer_bytes});
    g_moe_resident_bytes += layer_bytes;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "resident MoE cached layer %.2f GiB (total %.2f/%.2f GiB)\n",
            (double)layer_bytes / 1073741824.0,
            (double)g_moe_resident_bytes / 1073741824.0,
            (double)budget / 1073741824.0);
    *gate_w = gate_ptr;
    *up_w = up_ptr;
    *down_w = down_ptr;
    return 1;
}

static int cuda_model_sparse_moe_layer_ptrs(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (!cuda_env_flag_any3("DS4_CUDA_MOE_SPARSE_CACHE", "DS4_HIP_MOE_SPARSE_CACHE", NULL)) return 0;
    if (!model_map || !gate_w || !up_w || !down_w || !selected || !selected->ptr) return 0;
    if (g_model_device_owned) return 0;
    if (n_tokens == 0 || n_expert == 0 || n_expert > DS4_ROCM_N_EXPERT) return 0;
    if (!cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t))) return 0;
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    if (!cuda_model_range_fits(model_size, gate_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, down_bytes)) return 0;
    if (gate_expert_bytes > gate_bytes / DS4_ROCM_N_EXPERT ||
        down_expert_bytes > down_bytes / DS4_ROCM_N_EXPERT) return 0;

    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (pair_count > (uint64_t)UINT32_MAX) return 0;
    std::vector<int32_t> h_selected((size_t)pair_count);
    cudaError_t err = cudaMemcpy(h_selected.data(), selected->ptr,
                                 (size_t)(pair_count * sizeof(int32_t)),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    uint8_t used[DS4_ROCM_N_EXPERT] = {0};
    uint32_t experts[DS4_ROCM_N_EXPERT];
    uint32_t unique = 0;
    for (uint64_t i = 0; i < pair_count; i++) {
        const int32_t e = h_selected[(size_t)i];
        if (e < 0 || e >= (int32_t)DS4_ROCM_N_EXPERT) return 0;
        if (!used[e]) {
            used[e] = 1;
            experts[unique++] = (uint32_t)e;
        }
    }
    if (unique == 0) return 0;
    const uint32_t max_unique = cuda_parse_u32_env_alias(
            "DS4_CUDA_MOE_SPARSE_MAX_EXPERTS", "DS4_HIP_MOE_SPARSE_MAX_EXPERTS",
            64u, 1u, DS4_ROCM_N_EXPERT);
    if (unique > max_unique) return 0;

    if (!cuda_moe_sparse_buffers_prepare(gate_bytes, down_bytes)) return 0;

    uint64_t copied_bytes = 0;
    for (uint32_t i = 0; i < unique; i++) {
        const uint32_t e = experts[i];
        const uint64_t gate_delta = (uint64_t)e * gate_expert_bytes;
        const uint64_t down_delta = (uint64_t)e * down_expert_bytes;
        if (gate_delta > gate_bytes || gate_expert_bytes > gate_bytes - gate_delta ||
            down_delta > down_bytes || down_expert_bytes > down_bytes - down_delta) return 0;
        if (!cuda_model_copy_range_to_device_buffer(model_map, model_size,
                                                    g_moe_sparse_gate + gate_delta,
                                                    gate_offset + gate_delta,
                                                    gate_expert_bytes,
                                                    "sparse_moe_gate") ||
            !cuda_model_copy_range_to_device_buffer(model_map, model_size,
                                                    g_moe_sparse_up + gate_delta,
                                                    up_offset + gate_delta,
                                                    gate_expert_bytes,
                                                    "sparse_moe_up") ||
            !cuda_model_copy_range_to_device_buffer(model_map, model_size,
                                                    g_moe_sparse_down + down_delta,
                                                    down_offset + down_delta,
                                                    down_expert_bytes,
                                                    "sparse_moe_down")) {
            return 0;
        }
        copied_bytes += gate_expert_bytes + gate_expert_bytes + down_expert_bytes;
    }

    if (cuda_env_flag_any3("DS4_CUDA_MOE_SPARSE_VERBOSE", "DS4_HIP_MOE_SPARSE_VERBOSE", NULL)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "sparse MoE cache tokens=%u unique=%u copied=%.2f MiB layer=%.2f GiB\n",
                n_tokens, unique,
                (double)copied_bytes / 1048576.0,
                (double)(gate_bytes + gate_bytes + down_bytes) / 1073741824.0);
    }
    *gate_w = g_moe_sparse_gate;
    *up_w = g_moe_sparse_up;
    *down_w = g_moe_sparse_down;
    return 1;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t bytes = 1792ull * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    return (char *)dev;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        return cuda_model_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        return cuda_model_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (cuda_model_image_owned(model_map)) {
        g_model_host_base = model_map;
        g_model_device_base = cuda_model_image_ptr(model_map, 0);
        g_model_registered_size = model_size;
        g_model_device_owned = 1;
        return 1;
    }

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, DS4_GPU_LOG_PREFIX "chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) {
        (void)cudaFree(dev);
        return 0;
    }

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < model_size) {
        const uint64_t n = (model_size - copied < chunk) ? (model_size - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed: %s\n", cudaGetErrorString(err));
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staged read failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, strerror(errno));
            (void)cudaFree(dev);
            return 0;
        }
        err = cudaMemcpyAsync((char *)dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(copied, n);
        cuda_model_discard_source_pages(model_map, model_size, copied, n);
        copied += n;
        chunk_idx++;
        cuda_model_load_progress_note(copied > map_offset ? copied - map_offset : 0);
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload sync failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }
    g_model_images.push_back({model_map, model_size, (char *)dev});
    g_model_host_base = model_map;
    g_model_device_base = (const char *)dev;
    g_model_registered_size = model_size;
    g_model_device_owned = 1;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " %s failed: status %d\n", what, (int)st);
    return 0;
}


extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (!g_hipblaslt_ready) {
        if (hipblaslt_ok(hipblasLtCreate(&g_hipblaslt), "create handle")) {
            g_hipblaslt_ready = 1;
        }
    }
#endif
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    cuda_shared_gate_up_async_cleanup();
#ifdef __HIP_PLATFORM_AMD__
    hipblaslt_gemm_plan_clear();
#endif
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (g_hipblaslt_ready) {
        (void)hipblasLtDestroy(g_hipblaslt);
        g_hipblaslt_ready = 0;
        g_hipblaslt = NULL;
    }
#endif
    cuda_moe_resident_layers_release();
    cuda_moe_expert_cache_release();
    cuda_moe_sparse_buffers_release();
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    cuda_model_image_release_all();
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_end_commands(void) {
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_moe_resident_layers_release();
    cuda_moe_expert_cache_release();
    cuda_moe_sparse_buffers_release();
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    g_model_host_base = model_map;
    g_model_device_base = cuda_model_image_owned(model_map) ?
                          cuda_model_image_ptr(model_map, 0) :
                          (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_device_owned = cuda_model_image_owned(model_map);
    g_model_range_mapping_supported = 1;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    /* Strix Halo uses the staged full-copy path in ds4_gpu_set_model_map_range().
     * Avoid host-registering the mmap here: that would make the staged copier
     * believe the model is already device-resident. */
    return 1;
}

static int cuda_model_full_copy_requested(void) {
    return cuda_env_flag_any3("DS4_CUDA_COPY_MODEL", "DS4_HIP_COPY_MODEL", NULL) ||
           cuda_env_flag_any3("DS4_CUDA_COPY_MODEL_CHUNKED", "DS4_HIP_COPY_MODEL_CHUNKED", NULL) ||
           cuda_env_flag_any3("DS4_CUDA_COPY_MODEL_STAGED", "DS4_HIP_COPY_MODEL_STAGED", NULL);
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (!cuda_model_full_copy_requested()) return 1;
    return cuda_model_copy_chunked(model_map, model_size, map_offset, map_size);
}

extern "C" int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    uint64_t min_offset = offsets[0];
    uint64_t max_end = offsets[0] + sizes[0];
    for (uint32_t i = 1; i < count; i++) {
        if (offsets[i] < min_offset) min_offset = offsets[i];
        const uint64_t end = offsets[i] + sizes[i];
        if (end > max_end) max_end = end;
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (!cuda_model_full_copy_requested()) return 1;
    return cuda_model_copy_chunked(model_map, model_size, min_offset, max_end - min_offset);
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (!cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return cuda_model_range_is_cached(model_map, offset, bytes);
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    const int preload_transposed_b = !g_quality_mode &&
                                     strstr(cache_label, "attn_output_b") != NULL;
    if (preload_transposed_b) {
        const __half *f16_t = cuda_q8_f16_transpose_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16_t) {
            if (strstr(cache_label, "attn_output_b") != NULL && in_dim == 8192u && out_dim == 4096u) {
                cuda_q8_f16_warmup_attention_output_b_gemm(f16_t, in_dim, out_dim);
            }
            return 1;
        }
    } else {
        const __half *f16 = cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16) {
            if (strstr(cache_label, "attn_output_a") != NULL && in_dim == 4096u && out_dim == 8192u) {
                cuda_q8_f16_warmup_attention_output_a_gemm(f16, in_dim, 1024u, 8u);
            }
            return 1;
        }
    }
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "memory %s: query failed: %s\n",
                label ? label : "", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return;
    }
    const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
    const char *placement = cuda_model_image_bytes() ? "device_copy" : "mapped/range_cache";
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "memory %s: used=%.2f GiB free=%.2f GiB total=%.2f GiB "
            "placement=%s model_image=%.2f GiB range_cache=%.2f GiB "
            "q8_f16_cache=%.2f GiB scratch=%.2f GiB",
            label ? label : "",
            (double)used_b / 1073741824.0,
            (double)free_b / 1073741824.0,
            (double)total_b / 1073741824.0,
            placement,
            (double)cuda_model_image_bytes() / 1073741824.0,
            (double)g_model_range_bytes / 1073741824.0,
            (double)g_q8_f16_bytes / 1073741824.0,
            (double)g_cuda_tmp_bytes / 1073741824.0);
    fprintf(stderr, "\n");
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    const int new_quality_mode = quality ? 1 : 0;
    if (g_quality_mode != new_quality_mode) {
        g_rocm_cfg.initialized = 0;
    }
    g_quality_mode = new_quality_mode;
    if (g_cublas_ready) {
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}
