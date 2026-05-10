#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
#include "ds4_metal.h"
}

struct ds4_metal_tensor {
    unsigned char *ptr;
    uint64_t bytes;
    bool owner;
};

struct ds4_hip_model_range {
    const unsigned char *host_base;
    uint64_t host_size;
    unsigned char *device_base;
    uint64_t device_size;
    uint64_t map_offset;
    uint64_t map_size;
    bool copied;
};

struct ds4_hip_cached_model_tensor {
    const unsigned char *host_ptr;
    uint64_t bytes;
    unsigned char *device_ptr;
};

struct ds4_hip_repacked_q8_tensor {
    const unsigned char *host_ptr;
    uint64_t bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    int8_t *q;
    uint16_t *scales;
    uint64_t q_bytes;
    uint64_t scale_bytes;
};

struct ds4_hip_repacked_q8_split16_tensor {
    const unsigned char *host_ptr;
    uint64_t bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t n_splits;
    unsigned char *pack;
};

struct ds4_hip_repacked_q8_wmma_tensor {
    const unsigned char *host_ptr;
    uint64_t bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    half *bhalf_kn;
    uint64_t half_bytes;
};

static std::mutex g_mu;
static bool g_initialized;
static bool g_quality;
static int g_device;
static hipStream_t g_stream;
static hipDeviceProp_t g_prop;
static uint64_t g_tensor_live_bytes;
static uint64_t g_tensor_peak_bytes;
static uint64_t g_model_registered_bytes;
static uint64_t g_model_copied_bytes;
static uint64_t g_model_cached_bytes;
static uint64_t g_q8_repacked_bytes;
static uint64_t g_q8_split16_repacked_bytes;
static uint64_t g_q8_wmma_repacked_bytes;
static bool g_unsupported_warned;
static float *g_q8_partial_scratch;
static uint64_t g_q8_partial_scratch_floats;
static std::vector<ds4_hip_model_range> g_model_ranges;
static std::vector<ds4_hip_cached_model_tensor> g_model_cache;
static std::vector<ds4_hip_repacked_q8_tensor> g_q8_repack_cache;
static std::vector<ds4_hip_repacked_q8_split16_tensor> g_q8_split16_cache;
static std::vector<ds4_hip_repacked_q8_wmma_tensor> g_q8_wmma_cache;

static void ds4_hip_q8_repack_eager_from_gguf(const void *model_map, uint64_t model_size);

static const char *ds4_hip_err(hipError_t e) {
    return hipGetErrorString(e);
}

static bool ds4_hip_check(hipError_t e, const char *what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "ds4: HIP %s failed: %s\n", what, ds4_hip_err(e));
    return false;
}

static uint64_t ds4_hip_round_up_u64(uint64_t x, uint64_t a) {
    return a ? ((x + a - 1u) / a) * a : x;
}

static double ds4_hip_now_sec(void) {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

static uint64_t ds4_hip_env_mb(const char *name, uint64_t def_mb, uint64_t min_mb, uint64_t max_mb) {
    const char *s = std::getenv(name);
    if (!s || !s[0]) return def_mb;
    uint64_t v = (uint64_t)std::strtoull(s, nullptr, 10);
    if (v < min_mb) v = min_mb;
    if (v > max_mb) v = max_mb;
    return v;
}

static void ds4_hip_log_mem_info(const char *label) {
    size_t free_b = 0;
    size_t total_b = 0;
    hipError_t e = hipMemGetInfo(&free_b, &total_b);
    if (e == hipSuccess) {
        std::fprintf(stderr,
                     "ds4: HIP memory %s: free %.2f GiB / total %.2f GiB\n",
                     label ? label : "",
                     (double)free_b / 1073741824.0,
                     (double)total_b / 1073741824.0);
        std::fflush(stderr);
    } else {
        std::fprintf(stderr,
                     "ds4: HIP memory %s: hipMemGetInfo failed: %s\n",
                     label ? label : "",
                     ds4_hip_err(e));
        std::fflush(stderr);
    }
}

static unsigned ds4_hip_warp_threads(void) {
    return (g_prop.warpSize > 0 && g_prop.warpSize <= 256) ? (unsigned)g_prop.warpSize : 64u;
}

__global__ static void ds4_hip_probe_model_kernel(const unsigned char *p, uint64_t n, unsigned int *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!p || n == 0) {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        return;
    }
    out[0] = (unsigned int)p[0];
    out[1] = (unsigned int)p[n / 2u];
    out[2] = (unsigned int)p[n - 1u];
}

static bool ds4_hip_probe_model_range(const unsigned char *device_ptr,
                                      const unsigned char *host_ptr,
                                      uint64_t bytes) {
    if (!bytes) return true;
    unsigned int *out = nullptr;
    hipError_t e = hipMallocManaged(&out, 3u * sizeof(out[0]));
    if (!ds4_hip_check(e, "probe allocation")) return false;
    out[0] = out[1] = out[2] = 0;
    ds4_hip_probe_model_kernel<<<1, 1, 0, g_stream>>>(device_ptr, bytes, out);
    e = hipGetLastError();
    if (!ds4_hip_check(e, "probe launch")) {
        (void)hipFree(out);
        return false;
    }
    e = hipStreamSynchronize(g_stream);
    if (!ds4_hip_check(e, "probe synchronization")) {
        (void)hipFree(out);
        return false;
    }
    const unsigned int expect0 = (unsigned int)host_ptr[0];
    const unsigned int expect1 = (unsigned int)host_ptr[bytes / 2u];
    const unsigned int expect2 = (unsigned int)host_ptr[bytes - 1u];
    const bool ok = out[0] == expect0 && out[1] == expect1 && out[2] == expect2;
    if (!ok) {
        std::fprintf(stderr,
                     "ds4: HIP model probe mismatch: got {%u,%u,%u}, expected {%u,%u,%u}\n",
                     out[0], out[1], out[2], expect0, expect1, expect2);
    }
    (void)hipFree(out);
    return ok;
}

static int ds4_hip_unsupported(const char *fn) {
    if (!g_unsupported_warned) {
        g_unsupported_warned = true;
        std::fprintf(stderr,
                     "ds4: HIP backend loaded the model, but compute kernel %s is not ported yet\n",
                     fn ? fn : "<unknown>");
    }
    return 0;
}

extern "C" int ds4_metal_init(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_initialized) return 1;

    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (!ds4_hip_check(e, "device enumeration") || n <= 0) {
        std::fprintf(stderr, "ds4: HIP backend unavailable: no HIP GPU devices found\n");
        return 0;
    }

    const char *dev_env = std::getenv("DS4_HIP_DEVICE");
    g_device = dev_env && dev_env[0] ? std::atoi(dev_env) : 0;
    if (g_device < 0 || g_device >= n) {
        std::fprintf(stderr, "ds4: HIP device %d is outside available device range 0..%d\n", g_device, n - 1);
        return 0;
    }

    e = hipSetDevice(g_device);
    if (!ds4_hip_check(e, "set device")) return 0;
    e = hipGetDeviceProperties(&g_prop, g_device);
    if (!ds4_hip_check(e, "device properties")) return 0;
    e = hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking);
    if (!ds4_hip_check(e, "stream creation")) return 0;

    g_initialized = true;
    std::fprintf(stderr,
                 "ds4: HIP backend initialized: device %d: %s (gcnArch=%s, %.2f GiB global memory)\n",
                 g_device,
                 g_prop.name,
                 g_prop.gcnArchName,
                 (double)g_prop.totalGlobalMem / 1073741824.0);
    return 1;
}

extern "C" void ds4_metal_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_initialized) return;
    (void)hipStreamSynchronize(g_stream);

    for (auto &q8s : g_q8_split16_cache) {
        if (q8s.pack) (void)hipFree(q8s.pack);
    }
    g_q8_split16_cache.clear();
    for (auto &q8 : g_q8_repack_cache) {
        if (q8.q) (void)hipFree(q8.q);
        if (q8.scales) (void)hipFree(q8.scales);
    }
    g_q8_repack_cache.clear();
    for (auto &q8w : g_q8_wmma_cache) {
        if (q8w.bhalf_kn) (void)hipFree(q8w.bhalf_kn);
    }
    g_q8_wmma_cache.clear();
    for (auto &c : g_model_cache) {
        if (c.device_ptr) (void)hipFree(c.device_ptr);
    }
    g_model_cache.clear();
    for (auto &r : g_model_ranges) {
        if (r.copied) {
            if (r.device_base) (void)hipFree(r.device_base);
        } else if (r.host_base) {
            (void)hipHostUnregister((void *)r.host_base);
        }
    }
    g_model_ranges.clear();
    g_model_registered_bytes = 0;
    g_model_copied_bytes = 0;
    g_model_cached_bytes = 0;
    g_q8_repacked_bytes = 0;
    g_q8_split16_repacked_bytes = 0;
    g_q8_wmma_repacked_bytes = 0;
    if (g_q8_partial_scratch) (void)hipFree(g_q8_partial_scratch);
    g_q8_partial_scratch = nullptr;
    g_q8_partial_scratch_floats = 0;

    if (g_stream) (void)hipStreamDestroy(g_stream);
    g_stream = nullptr;
    g_initialized = false;
}

extern "C" ds4_metal_tensor *ds4_metal_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_metal_init()) return nullptr;
    if (bytes == 0) bytes = 1;
    void *ptr = nullptr;
    const bool device_tensors = std::getenv("DS4_HIP_MANAGED_TENSORS") == nullptr;
    hipError_t e = device_tensors ? hipMalloc(&ptr, (size_t)bytes) : hipMallocManaged(&ptr, (size_t)bytes);
    if (!ds4_hip_check(e, "tensor allocation")) return nullptr;
    if (!device_tensors) {
        (void)hipMemAdvise(ptr, (size_t)bytes, hipMemAdviseSetPreferredLocation, g_device);
        if (std::getenv("DS4_HIP_PREFETCH_TENSORS") != nullptr) {
            (void)hipMemPrefetchAsync(ptr, (size_t)bytes, g_device, g_stream);
        }
    }

    auto *t = new ds4_metal_tensor;
    t->ptr = static_cast<unsigned char *>(ptr);
    t->bytes = bytes;
    t->owner = true;
    g_tensor_live_bytes += bytes;
    g_tensor_peak_bytes = std::max(g_tensor_peak_bytes, g_tensor_live_bytes);
    return t;
}

extern "C" ds4_metal_tensor *ds4_metal_tensor_view(const ds4_metal_tensor *base,
                                                   uint64_t offset,
                                                   uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return nullptr;
    auto *t = new ds4_metal_tensor;
    t->ptr = base->ptr + offset;
    t->bytes = bytes;
    t->owner = false;
    return t;
}

extern "C" void ds4_metal_tensor_free(ds4_metal_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) {
        (void)hipStreamSynchronize(g_stream);
        (void)hipFree(tensor->ptr);
        if (g_tensor_live_bytes >= tensor->bytes) g_tensor_live_bytes -= tensor->bytes;
        else g_tensor_live_bytes = 0;
    }
    delete tensor;
}

extern "C" uint64_t ds4_metal_tensor_bytes(const ds4_metal_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_metal_tensor_contents(ds4_metal_tensor *tensor) {
    if (!tensor || std::getenv("DS4_HIP_MANAGED_TENSORS") == nullptr) return nullptr;
    (void)hipStreamSynchronize(g_stream);
    return tensor->ptr;
}

extern "C" int ds4_metal_tensor_write(ds4_metal_tensor *tensor,
                                       uint64_t offset,
                                       const void *data,
                                       uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    hipError_t e = hipMemcpyAsync(tensor->ptr + offset, data, (size_t)bytes, hipMemcpyHostToDevice, g_stream);
    if (!ds4_hip_check(e, "tensor write")) return 0;
    e = hipStreamSynchronize(g_stream);
    return ds4_hip_check(e, "tensor write synchronization") ? 1 : 0;
}

extern "C" int ds4_metal_tensor_read(const ds4_metal_tensor *tensor,
                                      uint64_t offset,
                                      void *data,
                                      uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    hipError_t e = hipMemcpyAsync(data, tensor->ptr + offset, (size_t)bytes, hipMemcpyDeviceToHost, g_stream);
    if (!ds4_hip_check(e, "tensor read")) return 0;
    e = hipStreamSynchronize(g_stream);
    return ds4_hip_check(e, "tensor read synchronization") ? 1 : 0;
}

extern "C" int ds4_metal_tensor_copy(ds4_metal_tensor *dst,
                                      uint64_t dst_offset,
                                      const ds4_metal_tensor *src,
                                      uint64_t src_offset,
                                      uint64_t bytes) {
    if (!dst || !src) return 0;
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 0;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    hipError_t e = hipMemcpyAsync(dst->ptr + dst_offset,
                                  src->ptr + src_offset,
                                  (size_t)bytes,
                                  hipMemcpyDeviceToDevice,
                                  g_stream);
    return ds4_hip_check(e, "tensor copy") ? 1 : 0;
}

extern "C" int ds4_metal_begin_commands(void) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    return 1;
}

extern "C" int ds4_metal_flush_commands(void) {
    if (!g_initialized) return 0;
    return ds4_hip_check(hipStreamSynchronize(g_stream), "flush") ? 1 : 0;
}

extern "C" int ds4_metal_end_commands(void) {
    if (!g_initialized) return 0;
    if (std::getenv("DS4_HIP_ASYNC_END_COMMANDS") != nullptr) {
        return ds4_hip_check(hipGetLastError(), "end commands") ? 1 : 0;
    }
    return ds4_hip_check(hipStreamSynchronize(g_stream), "end commands") ? 1 : 0;
}

extern "C" int ds4_metal_synchronize(void) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    return ds4_hip_check(hipStreamSynchronize(g_stream), "synchronize") ? 1 : 0;
}

static bool ds4_hip_pread_full(int fd, unsigned char *dst, uint64_t bytes, uint64_t file_offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = (size_t)std::min<uint64_t>(bytes - done, (uint64_t)SSIZE_MAX);
        const ssize_t r = pread(fd, dst + done, n, (off_t)(file_offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr,
                         "ds4: HIP staged model read failed at file offset %.2f GiB: %s\n",
                         (double)(file_offset + done) / 1073741824.0,
                         std::strerror(errno));
            std::fflush(stderr);
            return false;
        }
        if (r == 0) {
            std::fprintf(stderr,
                         "ds4: HIP staged model read hit EOF at file offset %.2f GiB\n",
                         (double)(file_offset + done) / 1073741824.0);
            std::fflush(stderr);
            return false;
        }
        done += (uint64_t)r;
    }
    return true;
}

static void ds4_hip_drop_model_source_pages(const void *model_map,
                                            uint64_t    model_size,
                                            int         model_fd,
                                            uint64_t    file_offset,
                                            uint64_t    bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (model_fd >= 0) {
        (void)posix_fadvise(model_fd, (off_t)file_offset, (off_t)bytes, POSIX_FADV_DONTNEED);
    }
#endif
#if defined(MADV_DONTNEED)
    if (model_map && bytes != 0 && file_offset < model_size) {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uint64_t end = std::min<uint64_t>(model_size, file_offset + bytes);
        const uint64_t aligned = file_offset & ~(page - 1u);
        const uint64_t aligned_end = ds4_hip_round_up_u64(end, page);
        if (aligned_end > aligned && aligned < model_size) {
            const uint64_t len = std::min<uint64_t>(aligned_end - aligned, model_size - aligned);
            (void)madvise((void *)(static_cast<const unsigned char *>(model_map) + aligned), (size_t)len, MADV_DONTNEED);
        }
    }
#else
    (void)model_map;
    (void)model_size;
    (void)model_fd;
    (void)file_offset;
    (void)bytes;
#endif
}

static bool ds4_hip_copy_model_staged(unsigned char *device_start,
                                      const void    *model_map,
                                      uint64_t       model_size,
                                      int            model_fd,
                                      uint64_t       map_offset,
                                      uint64_t       map_size,
                                      bool           quiet) {
    if (!device_start || !model_map || model_fd < 0 || map_size == 0) return false;
    const uint64_t chunk_mb = ds4_hip_env_mb("DS4_HIP_COPY_MODEL_CHUNK_MB", 256u, 16u, 4096u);
    const uint64_t chunk_bytes = chunk_mb * 1048576ull;
    double heartbeat_sec = 5.0;
    if (const char *hb_env = std::getenv("DS4_HIP_COPY_MODEL_HEARTBEAT_SEC")) {
        heartbeat_sec = std::strtod(hb_env, nullptr);
        if (heartbeat_sec < 0.0) heartbeat_sec = 0.0;
    }
    if (!quiet) {
        std::fprintf(stderr,
                     "ds4: HIP staged full model copy begin: %.2f GiB offset=%" PRIu64 " chunk=%" PRIu64 " MiB buffers=2\n",
                     (double)map_size / 1073741824.0,
                     map_offset,
                     chunk_mb);
        std::fflush(stderr);
    }

    unsigned char *stage[2] = {nullptr, nullptr};
    hipEvent_t ev[2]{};
    bool busy[2] = {false, false};
    uint64_t off[2] = {0, 0};
    uint64_t len[2] = {0, 0};
    double submit_t[2] = {0.0, 0.0};
    for (int i = 0; i < 2; i++) {
        hipError_t e = hipHostMalloc((void **)&stage[i], (size_t)chunk_bytes, hipHostMallocDefault);
        if (!ds4_hip_check(e, "staged model pinned host allocation")) {
            for (int j = 0; j <= i; j++) if (stage[j]) (void)hipHostFree(stage[j]);
            return false;
        }
        e = hipEventCreateWithFlags(&ev[i], hipEventDisableTiming);
        if (!ds4_hip_check(e, "staged model copy event allocation")) {
            for (int j = 0; j <= i; j++) if (stage[j]) (void)hipHostFree(stage[j]);
            for (int j = 0; j < i; j++) (void)hipEventDestroy(ev[j]);
            return false;
        }
    }

    const double t0 = ds4_hip_now_sec();
    double last_progress_t = t0;
    uint64_t completed = 0;
    uint64_t submitted = 0;

    auto wait_buffer = [&](int i) -> bool {
        if (!busy[i]) return true;
        double last_wait_log = ds4_hip_now_sec();
        for (;;) {
            hipError_t q = hipEventQuery(ev[i]);
            if (q == hipSuccess) break;
            if (q != hipErrorNotReady) {
                (void)ds4_hip_check(q, "staged model copy event query");
                return false;
            }
            const double now = ds4_hip_now_sec();
            if (!quiet && heartbeat_sec > 0.0 && now - last_wait_log >= heartbeat_sec) {
                std::fprintf(stderr,
                             "ds4: HIP staged model copy waiting at %.2f/%.2f GiB chunk %.2f MiB wait=%.1fs total=%.1fs\n",
                             (double)off[i] / 1073741824.0,
                             (double)map_size / 1073741824.0,
                             (double)len[i] / 1048576.0,
                             now - submit_t[i],
                             now - t0);
                std::fflush(stderr);
                last_wait_log = now;
            }
            usleep(100000);
        }
        busy[i] = false;
        completed += len[i];
        ds4_hip_drop_model_source_pages(model_map, model_size, model_fd, map_offset + off[i], len[i]);
        if (!quiet) {
            const double now = ds4_hip_now_sec();
            const double total_s = now - t0;
            const double recent_s = now - last_progress_t;
            const double avg_gibs = total_s > 0.0 ? ((double)completed / 1073741824.0) / total_s : 0.0;
            const double recent_gibs = recent_s > 0.0 ? ((double)len[i] / 1073741824.0) / recent_s : 0.0;
            const double eta_s = avg_gibs > 0.0 ? ((double)(map_size - completed) / 1073741824.0) / avg_gibs : 0.0;
            std::fprintf(stderr,
                         "ds4: HIP staged model copy %.2f/%.2f GiB (%.1f%%) recent=%.2f GiB/s avg=%.2f GiB/s elapsed=%.1fs eta=%.1fs\n",
                         (double)completed / 1073741824.0,
                         (double)map_size / 1073741824.0,
                         100.0 * (double)completed / (double)map_size,
                         recent_gibs,
                         avg_gibs,
                         total_s,
                         eta_s);
            std::fflush(stderr);
            last_progress_t = now;
        }
        return true;
    };

    int b = 0;
    while (submitted < map_size) {
        if (!wait_buffer(b)) goto fail;
        const uint64_t n = std::min<uint64_t>(chunk_bytes, map_size - submitted);
        if (!quiet) {
            std::fprintf(stderr,
                         "ds4: HIP staged model read begin %.2f/%.2f GiB size=%.2f MiB\n",
                         (double)submitted / 1073741824.0,
                         (double)map_size / 1073741824.0,
                         (double)n / 1048576.0);
            std::fflush(stderr);
        }
        const double read_t0 = ds4_hip_now_sec();
        if (!ds4_hip_pread_full(model_fd, stage[b], n, map_offset + submitted)) goto fail;
        const double read_t1 = ds4_hip_now_sec();
        if (!quiet) {
            const double read_s = read_t1 - read_t0;
            std::fprintf(stderr,
                         "ds4: HIP staged model read done %.2f/%.2f GiB read=%.2f GiB/s\n",
                         (double)(submitted + n) / 1073741824.0,
                         (double)map_size / 1073741824.0,
                         read_s > 0.0 ? ((double)n / 1073741824.0) / read_s : 0.0);
            std::fflush(stderr);
        }
        hipError_t e = hipMemcpyAsync(device_start + submitted, stage[b], (size_t)n, hipMemcpyHostToDevice, g_stream);
        if (!ds4_hip_check(e, "staged model copy to device")) goto fail;
        e = hipEventRecord(ev[b], g_stream);
        if (!ds4_hip_check(e, "staged model copy event record")) goto fail;
        busy[b] = true;
        off[b] = submitted;
        len[b] = n;
        submit_t[b] = ds4_hip_now_sec();
        submitted += n;
        b ^= 1;
    }
    if (!wait_buffer(0)) goto fail;
    if (!wait_buffer(1)) goto fail;
    for (int i = 0; i < 2; i++) {
        (void)hipEventDestroy(ev[i]);
        (void)hipHostFree(stage[i]);
    }
    if (!quiet) {
        const double total_s = ds4_hip_now_sec() - t0;
        std::fprintf(stderr,
                     "ds4: HIP staged full model copy done %.2f GiB in %.3f s (avg %.2f GiB/s)\n",
                     (double)map_size / 1073741824.0,
                     total_s,
                     total_s > 0.0 ? ((double)map_size / 1073741824.0) / total_s : 0.0);
        std::fflush(stderr);
    }
    return true;

fail:
    for (int i = 0; i < 2; i++) {
        if (ev[i]) (void)hipEventDestroy(ev[i]);
        if (stage[i]) (void)hipHostFree(stage[i]);
    }
    return false;
}

extern "C" int ds4_metal_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_metal_set_model_map_range(model_map, model_size, 0, model_size);
}

extern "C" int ds4_metal_set_model_map_range(const void *model_map,
                                              uint64_t model_size,
                                              uint64_t map_offset,
                                              uint64_t map_size) {
    return ds4_metal_set_model_map_range_fd(model_map, model_size, -1, map_offset, map_size);
}

extern "C" int ds4_metal_set_model_map_range_fd(const void *model_map,
                                                 uint64_t model_size,
                                                 int model_fd,
                                                 uint64_t map_offset,
                                                 uint64_t map_size) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!model_map || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (map_size == 0) return 1;

    const bool copy_model = std::getenv("DS4_HIP_COPY_MODEL") != nullptr;
    const unsigned char *host_start = static_cast<const unsigned char *>(model_map) + map_offset;
    unsigned char *device_start = nullptr;

    ds4_hip_model_range range{};
    range.map_offset = map_offset;
    range.map_size = map_size;
    range.copied = copy_model;

    if (copy_model) {
        const bool copy_quiet = std::getenv("DS4_HIP_COPY_MODEL_QUIET") != nullptr;
        const uint64_t chunk_mb = ds4_hip_env_mb("DS4_HIP_COPY_MODEL_CHUNK_MB", 1024u, 16u, 16384u);
        const uint64_t chunk_bytes = chunk_mb * 1048576ull;
        double heartbeat_sec = 5.0;
        if (const char *hb_env = std::getenv("DS4_HIP_COPY_MODEL_HEARTBEAT_SEC")) {
            heartbeat_sec = std::strtod(hb_env, nullptr);
            if (heartbeat_sec < 0.0) heartbeat_sec = 0.0;
        }
        const double t_begin = ds4_hip_now_sec();
        if (!copy_quiet) {
            std::fprintf(stderr,
                         "ds4: HIP full model copy begin: %.2f GiB at GGUF offset %" PRIu64
                         " chunk=%" PRIu64 " MiB\n",
                         (double)map_size / 1073741824.0,
                         map_offset,
                         chunk_mb);
            ds4_hip_log_mem_info("before model device allocation");
            std::fprintf(stderr, "ds4: HIP model device allocation begin %.2f GiB\n",
                         (double)map_size / 1073741824.0);
            std::fflush(stderr);
        }
        const double t_alloc0 = ds4_hip_now_sec();
        hipError_t e = hipMalloc(reinterpret_cast<void **>(&device_start), (size_t)map_size);
        const double t_alloc1 = ds4_hip_now_sec();
        if (!ds4_hip_check(e, "model device allocation")) {
            if (!copy_quiet) ds4_hip_log_mem_info("after failed model device allocation");
            return 0;
        }
        if (!copy_quiet) {
            std::fprintf(stderr,
                         "ds4: HIP model device allocation done %.3f s ptr=%p\n",
                         t_alloc1 - t_alloc0,
                         (void *)device_start);
            std::fflush(stderr);
            ds4_hip_log_mem_info("after model device allocation");
        }

        hipEvent_t copy_event{};
        bool have_copy_event = hipEventCreateWithFlags(&copy_event, hipEventDisableTiming) == hipSuccess;
        const bool use_staged_copy = model_fd >= 0 && std::getenv("DS4_HIP_COPY_MODEL_DIRECT") == nullptr;
        if (use_staged_copy) {
            if (!ds4_hip_copy_model_staged(device_start, model_map, model_size, model_fd, map_offset, map_size, copy_quiet)) {
                (void)hipFree(device_start);
                return 0;
            }
        } else {
        uint64_t copied = 0;
        double last_t = ds4_hip_now_sec();
        uint64_t last_copied = 0;
        while (copied < map_size) {
            const uint64_t n = std::min(chunk_bytes, map_size - copied);
            const double chunk_t0 = ds4_hip_now_sec();
            if (!copy_quiet) {
                std::fprintf(stderr,
                             "ds4: HIP model copy chunk begin %.2f/%.2f GiB size=%.2f MiB\n",
                             (double)copied / 1073741824.0,
                             (double)map_size / 1073741824.0,
                             (double)n / 1048576.0);
                std::fflush(stderr);
            }
            e = hipMemcpyAsync(device_start + copied, host_start + copied, (size_t)n, hipMemcpyHostToDevice, g_stream);
            if (!ds4_hip_check(e, "model copy to device")) {
                std::fprintf(stderr,
                             "ds4: HIP model copy failed at %.2f/%.2f GiB chunk %.2f MiB\n",
                             (double)copied / 1073741824.0,
                             (double)map_size / 1073741824.0,
                             (double)n / 1048576.0);
                std::fflush(stderr);
                if (have_copy_event) (void)hipEventDestroy(copy_event);
                (void)hipFree(device_start);
                return 0;
            }
            if (have_copy_event) {
                e = hipEventRecord(copy_event, g_stream);
                if (e != hipSuccess) have_copy_event = false;
            }
            if (have_copy_event) {
                double wait_last = chunk_t0;
                for (;;) {
                    e = hipEventQuery(copy_event);
                    if (e == hipSuccess) break;
                    if (e != hipErrorNotReady) {
                        if (!ds4_hip_check(e, "model copy event query")) {
                            if (have_copy_event) (void)hipEventDestroy(copy_event);
                            (void)hipFree(device_start);
                            return 0;
                        }
                    }
                    const double now_wait = ds4_hip_now_sec();
                    if (!copy_quiet && heartbeat_sec > 0.0 && now_wait - wait_last >= heartbeat_sec) {
                        std::fprintf(stderr,
                                     "ds4: HIP model copy waiting at %.2f/%.2f GiB chunk %.2f MiB wait=%.1fs total=%.1fs\n",
                                     (double)copied / 1073741824.0,
                                     (double)map_size / 1073741824.0,
                                     (double)n / 1048576.0,
                                     now_wait - chunk_t0,
                                     now_wait - t_begin);
                        std::fflush(stderr);
                        wait_last = now_wait;
                    }
                    usleep(100000);
                }
            } else {
                e = hipStreamSynchronize(g_stream);
                if (!ds4_hip_check(e, "model copy synchronization")) {
                    std::fprintf(stderr,
                                 "ds4: HIP model copy synchronization failed at %.2f/%.2f GiB\n",
                                 (double)copied / 1073741824.0,
                                 (double)map_size / 1073741824.0);
                    std::fflush(stderr);
                    (void)hipFree(device_start);
                    return 0;
                }
            }
            copied += n;
            const double now = ds4_hip_now_sec();
            if (!copy_quiet) {
                const double chunk_s = now - chunk_t0;
                const double total_s = now - t_begin;
                const double interval_s = now - last_t;
                const double avg_gibs = total_s > 0.0 ? ((double)copied / 1073741824.0) / total_s : 0.0;
                const double chunk_gibs = chunk_s > 0.0 ? ((double)n / 1073741824.0) / chunk_s : 0.0;
                const double interval_gibs = interval_s > 0.0 ? ((double)(copied - last_copied) / 1073741824.0) / interval_s : 0.0;
                const double eta_s = avg_gibs > 0.0 ? ((double)(map_size - copied) / 1073741824.0) / avg_gibs : 0.0;
                std::fprintf(stderr,
                             "ds4: HIP model copy %.2f/%.2f GiB (%.1f%%) chunk=%.2f GiB/s recent=%.2f GiB/s avg=%.2f GiB/s elapsed=%.1fs eta=%.1fs\n",
                             (double)copied / 1073741824.0,
                             (double)map_size / 1073741824.0,
                             100.0 * (double)copied / (double)map_size,
                             chunk_gibs,
                             interval_gibs,
                             avg_gibs,
                             total_s,
                             eta_s);
                std::fflush(stderr);
                last_t = now;
                last_copied = copied;
            }
        }
        if (have_copy_event) (void)hipEventDestroy(copy_event);
        }
        if (!copy_quiet) {
            const double total_s = ds4_hip_now_sec() - t_begin;
            std::fprintf(stderr,
                         "ds4: HIP full model copy done %.2f GiB in %.3f s (avg %.2f GiB/s)\n",
                         (double)map_size / 1073741824.0,
                         total_s,
                         total_s > 0.0 ? ((double)map_size / 1073741824.0) / total_s : 0.0);
            std::fflush(stderr);
            ds4_hip_log_mem_info("after model copy");
        }
        range.host_base = host_start;
        range.host_size = map_size;
        range.device_base = device_start;
        range.device_size = map_size;
        g_model_copied_bytes += map_size;
    } else {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uintptr_t full_base = reinterpret_cast<uintptr_t>(model_map);
        const uintptr_t wanted = reinterpret_cast<uintptr_t>(host_start);
        const uintptr_t aligned = wanted & ~(uintptr_t)(page - 1u);
        const uint64_t aligned_offset = (uint64_t)(aligned - full_base);
        const uint64_t delta = (uint64_t)(wanted - aligned);
        const uint64_t register_size = model_size - aligned_offset;
        void *registered_host = reinterpret_cast<void *>(aligned);
        hipError_t e = hipHostRegister(registered_host,
                                       (size_t)register_size,
                                       hipHostRegisterMapped | hipHostRegisterReadOnly);
        if (!ds4_hip_check(e, "model host registration")) return 0;
        void *device_base = nullptr;
        e = hipHostGetDevicePointer(&device_base, registered_host, 0);
        if (!ds4_hip_check(e, "model mapped device pointer")) {
            (void)hipHostUnregister(registered_host);
            return 0;
        }
        range.host_base = static_cast<const unsigned char *>(registered_host);
        range.host_size = register_size;
        range.device_base = static_cast<unsigned char *>(device_base);
        range.device_size = register_size;
        device_start = range.device_base + delta;
        if (std::getenv("DS4_HIP_MODEL_MEMADVISE") != nullptr) {
            hipError_t a = hipMemAdvise(device_base, (size_t)register_size, hipMemAdviseSetReadMostly, g_device);
            if (a != hipSuccess) {
                std::fprintf(stderr, "ds4: HIP model read-mostly memadvise ignored: %s\n", ds4_hip_err(a));
            }
            a = hipMemAdvise(device_base, (size_t)register_size, hipMemAdviseSetPreferredLocation, g_device);
            if (a != hipSuccess) {
                std::fprintf(stderr, "ds4: HIP model preferred-location memadvise ignored: %s\n", ds4_hip_err(a));
            }
        }
        g_model_registered_bytes += register_size;
    }

    if (!ds4_hip_probe_model_range(device_start, host_start, map_size)) {
        if (range.copied) (void)hipFree(range.device_base);
        else (void)hipHostUnregister((void *)range.host_base);
        return 0;
    }

    g_model_ranges.push_back(range);
    std::fprintf(stderr,
                 "ds4: HIP model range %s and GPU-probed: %.2f GiB at GGUF offset %" PRIu64 "\n",
                 copy_model ? "copied to device memory" : "host-registered for zero-copy GPU access",
                 (double)map_size / 1073741824.0,
                 map_offset);
    if (std::getenv("DS4_HIP_Q8_REPACK") != nullptr ||
        std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") != nullptr ||
        std::getenv("DS4_HIP_Q8_WMMA_FAST") != nullptr) {
        ds4_hip_q8_repack_eager_from_gguf(model_map, model_size);
    }
    return 1;
}

extern "C" void ds4_metal_set_quality(bool quality) {
    g_quality = quality;
}

extern "C" void ds4_metal_print_memory_report(const char *label) {
    std::fprintf(stderr,
                 "ds4: HIP memory%s%s: tensors live %.2f MiB peak %.2f MiB, model registered %.2f GiB copied %.2f GiB cached %.2f GiB q8_repacked %.2f GiB q8_split16 %.2f GiB q8_wmma %.2f GiB\n",
                 label ? " " : "",
                 label ? label : "",
                 (double)g_tensor_live_bytes / 1048576.0,
                 (double)g_tensor_peak_bytes / 1048576.0,
                 (double)g_model_registered_bytes / 1073741824.0,
                 (double)g_model_copied_bytes / 1073741824.0,
                 (double)g_model_cached_bytes / 1073741824.0,
                 (double)g_q8_repacked_bytes / 1073741824.0,
                 (double)g_q8_split16_repacked_bytes / 1073741824.0,
                 (double)g_q8_wmma_repacked_bytes / 1073741824.0);
}

static bool ds4_hip_should_cache_model_tensor(const char *what, uint64_t bytes) {
    if (std::getenv("DS4_HIP_CACHE_FINAL_Q8") != nullptr) {
        return what && std::strstr(what, "Q8_0 matmul") != nullptr && bytes >= 512ull * 1024ull * 1024ull;
    }
    if (std::getenv("DS4_HIP_CACHE_HOT_Q8") != nullptr) {
        return what && std::strstr(what, "Q8_0") != nullptr && bytes >= 32ull * 1024ull * 1024ull && bytes <= 40ull * 1024ull * 1024ull;
    }
    if (std::getenv("DS4_HIP_CACHE_Q8") == nullptr) return false;
    if (!what || std::strstr(what, "Q8_0") == nullptr) return false;
    uint64_t max_bytes = 128ull * 1024ull * 1024ull;
    const char *max_mb = std::getenv("DS4_HIP_CACHE_MODEL_MAX_MB");
    if (max_mb && max_mb[0]) max_bytes = (uint64_t)std::strtoull(max_mb, nullptr, 10) * 1024ull * 1024ull;
    return bytes != 0 && bytes <= max_bytes;
}

static float *ds4_hip_q8_partial_scratch(uint64_t floats) {
    if (floats == 0) return nullptr;
    if (g_q8_partial_scratch && g_q8_partial_scratch_floats >= floats) return g_q8_partial_scratch;
    if (g_q8_partial_scratch) {
        (void)hipStreamSynchronize(g_stream);
        (void)hipFree(g_q8_partial_scratch);
        g_q8_partial_scratch = nullptr;
        g_q8_partial_scratch_floats = 0;
    }
    float *p = nullptr;
    hipError_t e = hipMalloc(reinterpret_cast<void **>(&p), (size_t)(floats * sizeof(float)));
    if (!ds4_hip_check(e, "Q8 partial scratch allocation")) return nullptr;
    g_q8_partial_scratch = p;
    g_q8_partial_scratch_floats = floats;
    return p;
}

static const unsigned char *ds4_hip_model_ptr(const void *model_map,
                                              uint64_t model_size,
                                              uint64_t offset,
                                              uint64_t bytes,
                                              const char *what) {
    if (!model_map || offset > model_size || bytes > model_size - offset) {
        std::fprintf(stderr, "ds4: HIP %s range is outside the mapped model\n", what ? what : "weight");
        return nullptr;
    }
    const unsigned char *host = static_cast<const unsigned char *>(model_map) + offset;
    for (const auto &c : g_model_cache) {
        if (c.host_ptr == host && c.bytes == bytes) return c.device_ptr;
    }
    const unsigned char *mapped = nullptr;
    for (const auto &r : g_model_ranges) {
        if (host < r.host_base) continue;
        const uint64_t delta = (uint64_t)(host - r.host_base);
        if (delta <= r.host_size && bytes <= r.host_size - delta) {
            mapped = r.device_base + delta;
            break;
        }
    }
    if (!mapped) {
        std::fprintf(stderr, "ds4: HIP %s range has not been registered with the GPU\n", what ? what : "weight");
        return nullptr;
    }
    if (!ds4_hip_should_cache_model_tensor(what, bytes)) return mapped;

    unsigned char *cached = nullptr;
    hipError_t e = hipMalloc(reinterpret_cast<void **>(&cached), (size_t)bytes);
    if (e != hipSuccess) return mapped;
    e = hipMemcpyAsync(cached, mapped, (size_t)bytes, hipMemcpyDeviceToDevice, g_stream);
    if (e != hipSuccess || hipStreamSynchronize(g_stream) != hipSuccess) {
        (void)hipFree(cached);
        return mapped;
    }
    g_model_cache.push_back({host, bytes, cached});
    g_model_cached_bytes += bytes;
    return cached;
}

static bool ds4_hip_q8_repack_allowed(uint64_t weight_bytes, uint64_t in_dim, uint64_t out_dim) {
    if (std::getenv("DS4_HIP_Q8_REPACK") == nullptr) return false;
    if ((in_dim & 31ull) != 0 || in_dim == 0 || out_dim == 0) return false;
    /* Default to the proven ~34 MiB decode-hot Q8_0 tensors, especially
     * attn_q_b.  Smaller q_a/kv/shared shapes use different kernels and were
     * slower with the generic repacked matvec.  Override for experiments. */
    uint64_t min_bytes = 32ull * 1024ull * 1024ull;
    if (const char *min_mb = std::getenv("DS4_HIP_Q8_REPACK_MIN_MB")) {
        if (min_mb[0]) min_bytes = (uint64_t)std::strtoull(min_mb, nullptr, 10) * 1024ull * 1024ull;
    }
    uint64_t max_bytes = 40ull * 1024ull * 1024ull;
    if (const char *max_mb = std::getenv("DS4_HIP_Q8_REPACK_MAX_MB")) {
        if (max_mb[0]) max_bytes = (uint64_t)std::strtoull(max_mb, nullptr, 10) * 1024ull * 1024ull;
    }
    return weight_bytes >= min_bytes && weight_bytes <= max_bytes;
}

static const ds4_hip_repacked_q8_tensor *ds4_hip_q8_repack_get(const void *model_map,
                                                               uint64_t model_size,
                                                               uint64_t offset,
                                                               uint64_t in_dim,
                                                               uint64_t out_dim,
                                                               const char *what) {
    const uint64_t n_blocks = in_dim >> 5;
    if (n_blocks == 0 || out_dim == 0 || out_dim > UINT32_MAX) return nullptr;
    const uint64_t row_bytes = n_blocks * 34ull;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (!ds4_hip_q8_repack_allowed(weight_bytes, in_dim, out_dim)) return nullptr;
    if (!model_map || offset > model_size || weight_bytes > model_size - offset) return nullptr;
    const unsigned char *host = static_cast<const unsigned char *>(model_map) + offset;
    for (const auto &c : g_q8_repack_cache) {
        if (c.host_ptr == host && c.bytes == weight_bytes && c.in_dim == in_dim && c.out_dim == out_dim) return &c;
    }

    const uint64_t q_bytes = out_dim * in_dim;
    const uint64_t scale_elems = out_dim * n_blocks;
    const uint64_t scale_bytes = scale_elems * sizeof(uint16_t);
    std::vector<int8_t> hq;
    std::vector<uint16_t> hs;
    try {
        hq.resize((size_t)q_bytes);
        hs.resize((size_t)scale_elems);
    } catch (...) {
        std::fprintf(stderr, "ds4: HIP Q8 repack host allocation failed for %s %.2f MiB\n",
                     what ? what : "Q8_0", (double)(q_bytes + scale_bytes) / 1048576.0);
        return nullptr;
    }

    const double t0 = ds4_hip_now_sec();
    for (uint64_t row = 0; row < out_dim; row++) {
        const unsigned char *src = host + row * row_bytes;
        int8_t *qrow = hq.data() + row * in_dim;
        uint16_t *srow = hs.data() + row * n_blocks;
        for (uint64_t b = 0; b < n_blocks; b++) {
            const unsigned char *blk = src + b * 34ull;
            srow[b] = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            std::memcpy(qrow + b * 32ull, blk + 2u, 32u);
        }
    }

    int8_t *dq = nullptr;
    uint16_t *ds = nullptr;
    hipError_t e = hipMalloc(reinterpret_cast<void **>(&dq), (size_t)q_bytes);
    if (e != hipSuccess) {
        if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP Q8 repack skipped for %s: q allocation %.2f MiB failed: %s\n",
                         what ? what : "Q8_0", (double)q_bytes / 1048576.0, ds4_hip_err(e));
        }
        return nullptr;
    }
    e = hipMalloc(reinterpret_cast<void **>(&ds), (size_t)scale_bytes);
    if (e != hipSuccess) {
        if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP Q8 repack skipped for %s: scale allocation %.2f MiB failed: %s\n",
                         what ? what : "Q8_0", (double)scale_bytes / 1048576.0, ds4_hip_err(e));
        }
        (void)hipFree(dq);
        return nullptr;
    }
    e = hipMemcpyAsync(dq, hq.data(), (size_t)q_bytes, hipMemcpyHostToDevice, g_stream);
    if (e == hipSuccess) e = hipMemcpyAsync(ds, hs.data(), (size_t)scale_bytes, hipMemcpyHostToDevice, g_stream);
    if (e == hipSuccess) e = hipStreamSynchronize(g_stream);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP Q8 repack copy failed for %s: %s\n", what ? what : "Q8_0", ds4_hip_err(e));
        (void)hipFree(dq);
        (void)hipFree(ds);
        return nullptr;
    }

    g_q8_repack_cache.push_back({host, weight_bytes, in_dim, out_dim, dq, ds, q_bytes, scale_bytes});
    g_q8_repacked_bytes += q_bytes + scale_bytes;
    if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr) {
        const double dt = std::max(1e-9, ds4_hip_now_sec() - t0);
        std::fprintf(stderr,
                     "ds4: HIP Q8 repacked %s in=%.0f out=%.0f raw %.2f MiB -> q %.2f MiB + scales %.2f MiB in %.3f s\n",
                     what ? what : "Q8_0",
                     (double)in_dim,
                     (double)out_dim,
                     (double)weight_bytes / 1048576.0,
                     (double)q_bytes / 1048576.0,
                     (double)scale_bytes / 1048576.0,
                     dt);
    }
    return &g_q8_repack_cache.back();
}

static const ds4_hip_repacked_q8_split16_tensor *ds4_hip_q8_split16_repack_get(const void *model_map,
                                                                                uint64_t model_size,
                                                                                uint64_t offset,
                                                                                uint64_t in_dim,
                                                                                uint64_t out_dim,
                                                                                const char *what) {
    if (std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") == nullptr) return nullptr;
    if ((in_dim & 31ull) != 0 || in_dim == 0 || out_dim == 0 || out_dim > UINT32_MAX) return nullptr;
    const uint64_t n_blocks = in_dim >> 5;
    if ((n_blocks & 15ull) != 0) return nullptr;
    const uint32_t n_splits = (uint32_t)(n_blocks >> 4);
    const uint64_t row_bytes = n_blocks * 34ull;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (!model_map || offset > model_size || weight_bytes > model_size - offset) return nullptr;
    const unsigned char *host = static_cast<const unsigned char *>(model_map) + offset;
    for (const auto &c : g_q8_split16_cache) {
        if (c.host_ptr == host && c.bytes == weight_bytes && c.in_dim == in_dim && c.out_dim == out_dim) return &c;
    }

    const uint64_t pack_bytes = (uint64_t)n_splits * out_dim * 544ull;
    std::vector<unsigned char> hp;
    try {
        hp.resize((size_t)pack_bytes);
    } catch (...) {
        std::fprintf(stderr, "ds4: HIP Q8 split16 host allocation failed for %s %.2f MiB\n",
                     what ? what : "Q8_0", (double)pack_bytes / 1048576.0);
        return nullptr;
    }

    const double t0 = ds4_hip_now_sec();
    for (uint32_t s = 0; s < n_splits; s++) {
        for (uint64_t row = 0; row < out_dim; row++) {
            unsigned char *rec = hp.data() + ((uint64_t)s * out_dim + row) * 544ull;
            uint16_t *sc = reinterpret_cast<uint16_t *>(rec);
            unsigned char *qw = rec + 32u;
            const unsigned char *src = host + row * row_bytes + (uint64_t)s * 16ull * 34ull;
            for (uint32_t bb = 0; bb < 16u; bb++) {
                const unsigned char *blk = src + (uint64_t)bb * 34ull;
                sc[bb] = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
                std::memcpy(qw + (uint64_t)bb * 32ull, blk + 2u, 32u);
            }
        }
    }

    unsigned char *dp = nullptr;
    hipError_t e = hipMalloc(reinterpret_cast<void **>(&dp), (size_t)pack_bytes);
    if (e != hipSuccess) {
        if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP Q8 split16 repack skipped for %s: allocation %.2f MiB failed: %s\n",
                         what ? what : "Q8_0", (double)pack_bytes / 1048576.0, ds4_hip_err(e));
        }
        return nullptr;
    }
    e = hipMemcpyAsync(dp, hp.data(), (size_t)pack_bytes, hipMemcpyHostToDevice, g_stream);
    if (e == hipSuccess) e = hipStreamSynchronize(g_stream);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP Q8 split16 repack copy failed for %s: %s\n", what ? what : "Q8_0", ds4_hip_err(e));
        (void)hipFree(dp);
        return nullptr;
    }

    g_q8_split16_cache.push_back({host, weight_bytes, in_dim, out_dim, n_splits, dp});
    g_q8_split16_repacked_bytes += pack_bytes;
    if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr) {
        const double dt = std::max(1e-9, ds4_hip_now_sec() - t0);
        std::fprintf(stderr,
                     "ds4: HIP Q8 split16 repacked %s in=%.0f out=%.0f splits=%u bytes %.2f MiB in %.3f s\n",
                     what ? what : "Q8_0", (double)in_dim, (double)out_dim, n_splits,
                     (double)pack_bytes / 1048576.0, dt);
    }
    return &g_q8_split16_cache.back();
}

static uint32_t ds4_hip_gguf_u32(const unsigned char *p, uint64_t &pos) {
    uint32_t v;
    std::memcpy(&v, p + pos, sizeof(v));
    pos += sizeof(v);
    return v;
}

static uint64_t ds4_hip_gguf_u64(const unsigned char *p, uint64_t &pos) {
    uint64_t v;
    std::memcpy(&v, p + pos, sizeof(v));
    pos += sizeof(v);
    return v;
}

static std::string ds4_hip_gguf_string(const unsigned char *p, uint64_t &pos) {
    const uint64_t n = ds4_hip_gguf_u64(p, pos);
    std::string s((const char *)p + pos, (size_t)n);
    pos += n;
    return s;
}

__global__ static void ds4_hip_q8_wmma_repack_half_kn_kernel(half *__restrict__ bhalf_kn,
                                                              const unsigned char *__restrict__ w,
                                                              uint32_t in_dim,
                                                              uint32_t out_dim,
                                                              uint64_t row_bytes) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)in_dim * out_dim;
    if (idx >= total) return;
    const uint32_t k = (uint32_t)(idx / out_dim);
    const uint32_t row = (uint32_t)(idx - (uint64_t)k * out_dim);
    const unsigned char *blk = w + (uint64_t)row * row_bytes + (uint64_t)(k >> 5) * 34ull;
    const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)d_bits));
    const int8_t q = reinterpret_cast<const int8_t *>(blk + 2u)[k & 31u];
    bhalf_kn[idx] = __float2half(d * (float)q);
}

static const ds4_hip_repacked_q8_wmma_tensor *ds4_hip_q8_wmma_lookup(const void *model_map,
                                                                      uint64_t model_size,
                                                                      uint64_t offset,
                                                                      uint64_t in_dim,
                                                                      uint64_t out_dim) {
    if (!model_map || offset > model_size) return nullptr;
    const uint64_t n_blocks = (in_dim + 31u) >> 5;
    const uint64_t weight_bytes = out_dim * n_blocks * 34ull;
    if (weight_bytes > model_size - offset) return nullptr;
    const unsigned char *host = static_cast<const unsigned char *>(model_map) + offset;
    for (const auto &c : g_q8_wmma_cache) {
        if (c.host_ptr == host && c.bytes == weight_bytes && c.in_dim == in_dim && c.out_dim == out_dim) return &c;
    }
    return nullptr;
}

static const ds4_hip_repacked_q8_wmma_tensor *ds4_hip_q8_wmma_repack_create(const void *model_map,
                                                                             uint64_t model_size,
                                                                             uint64_t offset,
                                                                             uint64_t in_dim,
                                                                             uint64_t out_dim,
                                                                             const char *what) {
    if (std::getenv("DS4_HIP_Q8_WMMA_FAST") == nullptr) return nullptr;
    if (!model_map || in_dim == 0 || out_dim == 0 || (in_dim & 15ull) != 0 || out_dim > UINT32_MAX) return nullptr;
    const uint64_t n_blocks = (in_dim + 31u) >> 5;
    if ((in_dim & 31ull) != 0) return nullptr; // Q8_0 tensors in DS4 hot path are block-aligned.
    const uint64_t row_bytes = n_blocks * 34ull;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (offset > model_size || weight_bytes > model_size - offset) return nullptr;
    if (const auto *hit = ds4_hip_q8_wmma_lookup(model_map, model_size, offset, in_dim, out_dim)) return hit;

    const uint64_t half_elems = in_dim * out_dim;
    const uint64_t half_bytes = half_elems * sizeof(half);
    if (half_elems == 0 || half_elems > (uint64_t)SIZE_MAX / sizeof(half)) return nullptr;

    const double t0 = ds4_hip_now_sec();
    const unsigned char *host = static_cast<const unsigned char *>(model_map) + offset;
    const unsigned char *src_dev = ds4_hip_model_ptr(model_map, model_size, offset, weight_bytes, "Q8_0 WMMA repack source");
    if (!src_dev) return nullptr;

    half *db = nullptr;
    hipError_t e = hipMalloc(reinterpret_cast<void **>(&db), (size_t)half_bytes);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP Q8 WMMA repack skipped for %s: allocation %.2f MiB failed: %s\n",
                     what ? what : "Q8_0", (double)half_bytes / 1048576.0, ds4_hip_err(e));
        return nullptr;
    }
    ds4_hip_q8_wmma_repack_half_kn_kernel<<<(unsigned)((half_elems + 255ull) / 256ull), 256, 0, g_stream>>>(
            db, src_dev, (uint32_t)in_dim, (uint32_t)out_dim, row_bytes);
    e = hipGetLastError();
    if (e == hipSuccess) e = hipStreamSynchronize(g_stream);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP Q8 WMMA repack kernel failed for %s: %s\n", what ? what : "Q8_0", ds4_hip_err(e));
        (void)hipFree(db);
        return nullptr;
    }

    g_q8_wmma_cache.push_back({host, weight_bytes, in_dim, out_dim, db, half_bytes});
    g_q8_wmma_repacked_bytes += half_bytes;
    if (std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr || std::getenv("DS4_HIP_Q8_WMMA_LOG") != nullptr) {
        const double dt = std::max(1e-9, ds4_hip_now_sec() - t0);
        std::fprintf(stderr,
                     "ds4: HIP Q8 WMMA repacked %s in=%.0f out=%.0f raw %.2f MiB -> half %.2f MiB in %.3f s\n",
                     what ? what : "Q8_0", (double)in_dim, (double)out_dim,
                     (double)weight_bytes / 1048576.0, (double)half_bytes / 1048576.0, dt);
    }
    return &g_q8_wmma_cache.back();
}

static void ds4_hip_gguf_skip_value(const unsigned char *p, uint64_t &pos, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: pos += 1; break;
        case 2: case 3: pos += 2; break;
        case 4: case 5: case 6: pos += 4; break;
        case 10: case 11: case 12: pos += 8; break;
        case 8: {
            const uint64_t n = ds4_hip_gguf_u64(p, pos);
            pos += n;
            break;
        }
        case 9: {
            const uint32_t elem_type = ds4_hip_gguf_u32(p, pos);
            const uint64_t n = ds4_hip_gguf_u64(p, pos);
            for (uint64_t i = 0; i < n; i++) ds4_hip_gguf_skip_value(p, pos, elem_type);
            break;
        }
        default:
            pos = UINT64_MAX;
            break;
    }
}

static void ds4_hip_q8_repack_eager_from_gguf(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size < 32u) return;
    if (std::getenv("DS4_HIP_Q8_REPACK_EAGER") != nullptr && std::getenv("DS4_HIP_Q8_REPACK") == nullptr) return;
    const unsigned char *p = static_cast<const unsigned char *>(model_map);
    if (std::memcmp(p, "GGUF", 4) != 0) return;
    uint64_t pos = 4;
    (void)ds4_hip_gguf_u32(p, pos);
    const uint64_t n_tensors = ds4_hip_gguf_u64(p, pos);
    const uint64_t n_kv = ds4_hip_gguf_u64(p, pos);
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && pos < model_size; i++) {
        const std::string key = ds4_hip_gguf_string(p, pos);
        const uint32_t type = ds4_hip_gguf_u32(p, pos);
        if (key == "general.alignment" && type == 4) alignment = ds4_hip_gguf_u32(p, pos);
        else ds4_hip_gguf_skip_value(p, pos, type);
        if (pos == UINT64_MAX || pos > model_size) return;
    }

    struct q8_info { std::string name; uint64_t in_dim; uint64_t out_dim; uint64_t rel; };
    std::vector<q8_info> hot;
    std::vector<q8_info> split16_hot;
    std::vector<q8_info> wmma_hot;
    hot.reserve(64);
    split16_hot.reserve(160);
    wmma_hot.reserve(256);
    for (uint64_t i = 0; i < n_tensors && pos < model_size; i++) {
        const std::string name = ds4_hip_gguf_string(p, pos);
        const uint32_t nd = ds4_hip_gguf_u32(p, pos);
        uint64_t dims[4] = {0, 0, 0, 0};
        for (uint32_t d = 0; d < nd && d < 4u; d++) dims[d] = ds4_hip_gguf_u64(p, pos);
        for (uint32_t d = 4u; d < nd; d++) (void)ds4_hip_gguf_u64(p, pos);
        const uint32_t type = ds4_hip_gguf_u32(p, pos);
        const uint64_t rel = ds4_hip_gguf_u64(p, pos);
        if (type == 8u && nd == 2u && dims[0] != 0 && dims[1] != 0) {
            const bool is_q_b = name.find(".attn_q_b.weight") != std::string::npos;
            const bool allow_all = std::getenv("DS4_HIP_Q8_REPACK_EAGER_ALL") != nullptr;
            if ((std::getenv("DS4_HIP_Q8_REPACK") != nullptr) && (is_q_b || allow_all)) hot.push_back({name, dims[0], dims[1], rel});
            const bool is_out_a = name.find(".attn_output_a.weight") != std::string::npos;
            const bool is_out_b = name.find(".attn_output_b.weight") != std::string::npos;
            if (std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") != nullptr) {
                const bool is_shared_down = name.find(".ffn_down_shexp.weight") != std::string::npos;
                if (is_out_a || is_out_b || is_shared_down) split16_hot.push_back({name, dims[0], dims[1], rel});
            }
            if (std::getenv("DS4_HIP_Q8_WMMA_FAST") != nullptr) {
                const bool is_q_a = name.find(".attn_q_a.weight") != std::string::npos;
                /* WMMA remains opt-in and is restricted to attention/indexer Q-side
                 * projections.  Output projections write directly to the residual
                 * stream and showed worse greedy drift with little useful speedup. */
                if (is_q_a || is_q_b) wmma_hot.push_back({name, dims[0], dims[1], rel});
            }
        }
    }
    if (pos == UINT64_MAX || pos > model_size) return;
    const uint64_t data_pos = ds4_hip_round_up_u64(pos, alignment ? alignment : 32u);

    const double t0 = ds4_hip_now_sec();
    uint32_t done = 0;
    uint64_t before = g_q8_repacked_bytes;
    for (const auto &t : hot) {
        const uint64_t abs = data_pos + t.rel;
        if (abs >= model_size) continue;
        if (ds4_hip_q8_repack_get(model_map, model_size, abs, t.in_dim, t.out_dim, t.name.c_str())) done++;
    }
    uint32_t split16_done = 0;
    uint64_t split16_before = g_q8_split16_repacked_bytes;
    for (const auto &t : split16_hot) {
        const uint64_t abs = data_pos + t.rel;
        if (abs >= model_size) continue;
        if (ds4_hip_q8_split16_repack_get(model_map, model_size, abs, t.in_dim, t.out_dim, t.name.c_str())) split16_done++;
    }
    uint32_t wmma_done = 0;
    uint64_t wmma_before = g_q8_wmma_repacked_bytes;
    for (const auto &t : wmma_hot) {
        const uint64_t abs = data_pos + t.rel;
        if (abs >= model_size) continue;
        if (ds4_hip_q8_wmma_repack_create(model_map, model_size, abs, t.in_dim, t.out_dim, t.name.c_str())) wmma_done++;
    }
    if (done != 0 || split16_done != 0 || wmma_done != 0 || std::getenv("DS4_HIP_Q8_REPACK_LOG") != nullptr || std::getenv("DS4_HIP_Q8_WMMA_LOG") != nullptr) {
        const double dt = ds4_hip_now_sec() - t0;
        std::fprintf(stderr,
                     "ds4: HIP Q8 eager repack q_b=%u/%zu bytes=%.2f GiB split16=%u/%zu bytes=%.2f GiB wmma=%u/%zu bytes=%.2f GiB in %.3f s\n",
                     done, hot.size(), (double)(g_q8_repacked_bytes - before) / 1073741824.0,
                     split16_done, split16_hot.size(), (double)(g_q8_split16_repacked_bytes - split16_before) / 1073741824.0,
                     wmma_done, wmma_hot.size(), (double)(g_q8_wmma_repacked_bytes - wmma_before) / 1073741824.0,
                     dt);
    }
}

__host__ __device__ static float ds4_hip_f16_to_f32(uint16_t h) {
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    return __half2float(__ushort_as_half((unsigned short)h));
#else
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            exp = exp + (127u - 15u);
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp + (127u - 15u);
        bits = sign | (exp << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v;
    v.u = bits;
    return v.f;
#endif
}

__device__ static float ds4_hip_warp_reduce_sum(float v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, warpSize);
    }
    return v;
}

__device__ static float ds4_hip_warp_reduce_max(float v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down(v, offset, warpSize));
    }
    return v;
}

__device__ static float ds4_hip_block_reduce_sum(float v) {
    __shared__ float sh[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (warpSize - 1u);
    const unsigned int wid = tid / warpSize;
    const unsigned int nwarp = (blockDim.x + warpSize - 1u) / warpSize;
    v = ds4_hip_warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : 0.0f;
    if (wid == 0) v = ds4_hip_warp_reduce_sum(v);
    if (tid == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ static float ds4_hip_block_reduce_max(float v) {
    __shared__ float sh[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (warpSize - 1u);
    const unsigned int wid = tid / warpSize;
    const unsigned int nwarp = (blockDim.x + warpSize - 1u) / warpSize;
    v = ds4_hip_warp_reduce_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : -3.4e38f;
    if (wid == 0) v = ds4_hip_warp_reduce_max(v);
    if (tid == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ static void ds4_hip_block_reduce_store(float *out, float v, uint64_t idx) {
    v = ds4_hip_block_reduce_sum(v);
    if (threadIdx.x == 0) out[idx] = v;
}

__global__ static void ds4_hip_matmul_f16_kernel(float *out,
                                                 const uint16_t *w,
                                                 const float *x,
                                                 uint64_t in_dim,
                                                 uint64_t out_dim) {
    const uint64_t o = (uint64_t)blockIdx.x;
    const uint64_t t = (uint64_t)blockIdx.y;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const uint16_t *row = w + o * in_dim;
    const float *xr = x + t * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += ds4_hip_f16_to_f32(row[i]) * xr[i];
    }
    ds4_hip_block_reduce_store(out, acc, t * out_dim + o);
}

__global__ static void ds4_hip_matmul_f16_pair_kernel(float *out_a,
                                                      float *out_b,
                                                      const uint16_t *wa,
                                                      const uint16_t *wb,
                                                      const float *x,
                                                      uint64_t in_dim,
                                                      uint64_t out_dim) {
    const uint64_t o = (uint64_t)blockIdx.x;
    if (o >= out_dim) return;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    const uint16_t *row_a = wa + o * in_dim;
    const uint16_t *row_b = wb + o * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float xv = x[i];
        acc_a += ds4_hip_f16_to_f32(row_a[i]) * xv;
        acc_b += ds4_hip_f16_to_f32(row_b[i]) * xv;
    }
    __shared__ float sha[256];
    __shared__ float shb[256];
    const unsigned int tid = threadIdx.x;
    sha[tid] = acc_a;
    shb[tid] = acc_b;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) {
            sha[tid] += sha[tid + stride];
            shb[tid] += shb[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_a[o] = sha[0];
        out_b[o] = shb[0];
    }
}

__global__ static void ds4_hip_matmul_f32_kernel(float *out,
                                                 const float *w,
                                                 const float *x,
                                                 uint64_t in_dim,
                                                 uint64_t out_dim) {
    const uint64_t o = (uint64_t)blockIdx.x;
    const uint64_t t = (uint64_t)blockIdx.y;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const float *row = w + o * in_dim;
    const float *xr = x + t * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) acc += row[i] * xr[i];
    ds4_hip_block_reduce_store(out, acc, t * out_dim + o);
}

__device__ static inline float ds4_hip_q8_0_scale_broadcast(const unsigned char *blk) {
    const uint32_t lane = (uint32_t)threadIdx.x & ((uint32_t)warpSize - 1u);
    const uint32_t sublane = lane & 31u;
    float d = 0.0f;
    if (sublane == 0) {
        const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        d = ds4_hip_f16_to_f32(d_bits);
    }
    return __shfl(d, (int)(lane - sublane), warpSize);
}

__device__ static inline float ds4_hip_q8_0_scale_broadcast_w32(const unsigned char *blk) {
    float d = 0.0f;
    if ((threadIdx.x & 31u) == 0u) {
        const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        d = ds4_hip_f16_to_f32(d_bits);
    }
    return __shfl(d, 0, 32);
}

__device__ static inline float ds4_hip_q8_repack_scale_broadcast_w32(const uint16_t *__restrict__ scales,
                                                                     uint64_t idx) {
    float d = 0.0f;
    if ((threadIdx.x & 31u) == 0u) d = ds4_hip_f16_to_f32(scales[idx]);
    return __shfl(d, 0, 32);
}

__global__ static void ds4_hip_matmul_q8_repack_warp_rows_w32_kernel(float *__restrict__ out,
                                                                     const int8_t *__restrict__ q,
                                                                     const uint16_t *__restrict__ scales,
                                                                     const float *__restrict__ x,
                                                                     uint32_t n_blocks,
                                                                     uint64_t out_dim) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    const int8_t *qrow = q + o * ((uint64_t)n_blocks << 5);
    const uint16_t *srow = scales + o * (uint64_t)n_blocks;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(srow, b);
        const int8_t qv = qrow[((uint64_t)b << 5) + lane];
        acc += d * (float)qv * x[((uint64_t)b << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[o] = acc;
}

__global__ static void ds4_hip_matmul_q8_repack_sharedx_rows_w32_kernel(float *__restrict__ out,
                                                                        const int8_t *__restrict__ q,
                                                                        const uint16_t *__restrict__ scales,
                                                                        const float *__restrict__ x,
                                                                        uint32_t n_blocks,
                                                                        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t in_dim = n_blocks << 5;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    const int8_t *qrow = q + o * (uint64_t)in_dim;
    const uint16_t *srow = scales + o * (uint64_t)n_blocks;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(srow, b);
        const uint32_t i = (b << 5) + lane;
        const int8_t qv = qrow[i];
        acc += d * (float)qv * shx[i];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[o] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_kernel(float *out,
                                                  const unsigned char *w,
                                                  const float *x,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  uint64_t row_bytes) {
    const uint64_t o = (uint64_t)blockIdx.x;
    const uint64_t t = (uint64_t)blockIdx.y;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const unsigned char *row = w + o * row_bytes;
    const float *xr = x + t * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const uint64_t b = i >> 5;
        const uint64_t lane = i & 31u;
        const unsigned char *blk = row + b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * xr[i];
    }
    if (blockDim.x == warpSize) {
        acc = ds4_hip_warp_reduce_sum(acc);
        if (threadIdx.x == 0) out[t * out_dim + o] = acc;
    } else {
        ds4_hip_block_reduce_store(out, acc, t * out_dim + o);
    }
}

__global__ static void ds4_hip_matmul_q8_0_kernel_w32(float *__restrict__ out,
                                                      const unsigned char *__restrict__ w,
                                                      const float *__restrict__ x,
                                                      uint32_t n_blocks,
                                                      uint64_t out_dim,
                                                      uint64_t row_bytes) {
    const uint64_t o = (uint64_t)blockIdx.x;
    const uint64_t t = (uint64_t)blockIdx.y;
    if (o >= out_dim) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t waves_per_block = blockDim.x >> 5;
    float acc = 0.0f;
    const unsigned char *row = w + o * row_bytes;
    const float *xr = x + t * ((uint64_t)n_blocks << 5);
    for (uint32_t b = wave; b < n_blocks; b += waves_per_block) {
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * xr[((uint64_t)b << 5) + lane];
    }
    ds4_hip_block_reduce_store(out, acc, t * out_dim + o);
}

__global__ static void ds4_hip_matmul_q8_0_warp_rows_w32_kernel(float *__restrict__ out,
                                                                 const unsigned char *__restrict__ w,
                                                                 const float *__restrict__ x,
                                                                 uint32_t n_blocks,
                                                                 uint64_t out_dim,
                                                                 uint64_t row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    const unsigned char *row = w + o * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * x[((uint64_t)b << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[o] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_warp_rows_w32_2row_kernel(float *__restrict__ out,
                                                                     const unsigned char *__restrict__ w,
                                                                     const float *__restrict__ x,
                                                                     uint32_t n_blocks,
                                                                     uint64_t out_dim,
                                                                     uint64_t row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = (blockDim.x >> 5) << 1;
    const uint64_t o0 = (uint64_t)blockIdx.x * rows_per_block + ((uint64_t)wave << 1);
    if (o0 >= out_dim) return;
    const uint64_t o1 = o0 + 1u;
    const unsigned char *row0 = w + o0 * row_bytes;
    const unsigned char *row1 = w + o1 * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float xv = x[((uint64_t)b << 5) + lane];
        const unsigned char *blk0 = row0 + (uint64_t)b * 34u;
        const float d0 = ds4_hip_q8_0_scale_broadcast_w32(blk0);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        acc0 += d0 * (float)q0 * xv;
        if (o1 < out_dim) {
            const unsigned char *blk1 = row1 + (uint64_t)b * 34u;
            const float d1 = ds4_hip_q8_0_scale_broadcast_w32(blk1);
            const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
            acc1 += d1 * (float)q1 * xv;
        }
    }
    acc0 = ds4_hip_warp_reduce_sum(acc0);
    acc1 = ds4_hip_warp_reduce_sum(acc1);
    if (lane == 0) {
        out[o0] = acc0;
        if (o1 < out_dim) out[o1] = acc1;
    }
}

/* Batched Q8_0 matmul for prefill.  One wave computes one output row for a
 * small tile of tokens, reusing each zero-copy weight row load across multiple
 * tokens.  This is the same FlashInfer/AITER-style idea as tiled prefill: move
 * the expensive invariant operand once, consume several query rows. */
template <uint32_t TOK_TILE>
__global__ static void ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel(float *__restrict__ out,
                                                                        const unsigned char *__restrict__ w,
                                                                        const float *__restrict__ x,
                                                                        uint32_t n_blocks,
                                                                        uint32_t out_dim,
                                                                        uint32_t n_tok,
                                                                        uint64_t row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t o = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (o >= out_dim || t0 >= n_tok) return;
    const unsigned char *row = w + (uint64_t)o * row_bytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        const float wv = d * (float)q;
        const uint64_t xoff = ((uint64_t)b << 5) + lane;
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) acc[u] += wv * x[(uint64_t)t * ((uint64_t)n_blocks << 5) + xoff];
        }
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = ds4_hip_warp_reduce_sum(acc[u]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) out[(uint64_t)t * out_dim + o] = acc[u];
        }
    }
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel(float *__restrict__ out,
                                                                                const unsigned char *__restrict__ w,
                                                                                const float *__restrict__ x,
                                                                                uint32_t n_blocks,
                                                                                uint32_t out_dim,
                                                                                uint32_t n_tok,
                                                                                uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t o = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (t0 >= n_tok) return;
    const bool row_valid = o < out_dim;
    const unsigned char *row = w + (uint64_t)(row_valid ? o : 0u) * row_bytes;
    const uint32_t in_dim = n_blocks << 5;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;
    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            shx[j] = (t < n_tok && bb < b_count) ? x[(uint64_t)t * in_dim + ((uint64_t)(b0 + bb) << 5) + k] : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *blk = row + (uint64_t)(b0 + bb) * 34u;
                const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                const float wv = d * (float)q;
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = ds4_hip_warp_reduce_sum(acc[u]);
    if (lane == 0 && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) out[(uint64_t)t * out_dim + o] = acc[u];
        }
    }
}

template <uint32_t BLOCKS_TILE>
static inline void ds4_hip_launch_q8_0_batch_sharedx(float *out,
                                                     const unsigned char *w,
                                                     const float *x,
                                                     uint32_t n_blocks,
                                                     uint32_t out_dim,
                                                     uint32_t n_tok,
                                                     uint64_t row_bytes,
                                                     dim3 grid,
                                                     unsigned threads,
                                                     unsigned tile) {
    const size_t shmem = (size_t)tile * BLOCKS_TILE * 32u * sizeof(float);
    if (tile == 16u) {
        ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<16, BLOCKS_TILE><<<grid, threads, shmem, g_stream>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 8u) {
        ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<8, BLOCKS_TILE><<<grid, threads, shmem, g_stream>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 2u) {
        ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<2, BLOCKS_TILE><<<grid, threads, shmem, g_stream>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 32u) {
        ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<32, BLOCKS_TILE><<<grid, threads, shmem, g_stream>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        ds4_hip_matmul_q8_0_warp_rows_w32_toktile_sharedx_kernel<4, BLOCKS_TILE><<<grid, threads, shmem, g_stream>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    }
}

template <int TILES_N, int BM = 16, int BN = 16, int BK = 16>
__global__ static void ds4_hip_matmul_q8_wmma_packed_multin_kernel(float *__restrict__ out,
                                                                   const half *__restrict__ bhalf_kn,
                                                                   const float *__restrict__ x,
                                                                   uint32_t n_tok,
                                                                   uint32_t in_dim,
                                                                   uint32_t out_dim) {
    extern __shared__ unsigned char shmem_raw[];
    half *shA = reinterpret_cast<half *>(shmem_raw);
    half *shB = shA + BM * BK;
    float *shC = reinterpret_cast<float *>(shB + TILES_N * BK * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t n0 = blockIdx.x * (TILES_N * BN);
    const uint32_t n_wave = n0 + wave * BN;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);

    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t mm = j / BK;
            const uint32_t kk = j - mm * BK;
            const uint32_t m = m0 + mm;
            shA[j] = (m < n_tok) ? __float2half(x[(uint64_t)m * in_dim + k0 + kk]) : __float2half(0.0f);
        }
        for (uint32_t j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const uint32_t tile = j / (BK * BN);
            const uint32_t r = j - tile * (BK * BN);
            const uint32_t kk = r / BN;
            const uint32_t nn = r - kk * BN;
            const uint32_t n = n0 + tile * BN + nn;
            shB[j] = (n < out_dim) ? bhalf_kn[(uint64_t)(k0 + kk) * out_dim + n] : __float2half(0.0f);
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < TILES_N) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < TILES_N * BM * BN; j += blockDim.x) {
        const uint32_t tile = j / (BM * BN);
        const uint32_t r = j - tile * (BM * BN);
        const uint32_t mm = r / BN;
        const uint32_t nn = r - mm * BN;
        const uint32_t m = m0 + mm;
        const uint32_t n = n0 + tile * BN + nn;
        if (m < n_tok && n < out_dim) out[(uint64_t)m * out_dim + n] = shC[j];
    }
}

template <int TILES_N, int BM = 16, int BN = 16, int BK = 16>
__global__ static void ds4_hip_matmul_q8_wmma_packed_multin_xsplit_kernel(float *__restrict__ out,
                                                                          const half *__restrict__ bhalf_kn,
                                                                          const float *__restrict__ x,
                                                                          uint32_t n_tok,
                                                                          uint32_t in_dim,
                                                                          uint32_t out_dim) {
    extern __shared__ unsigned char shmem_raw[];
    half *shA = reinterpret_cast<half *>(shmem_raw);
    half *shB = shA + BM * BK;
    float *shC = reinterpret_cast<float *>(shB + TILES_N * BK * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t n0 = blockIdx.x * (TILES_N * BN);

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);

    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t mm = j / BK;
            const uint32_t kk = j - mm * BK;
            const uint32_t m = m0 + mm;
            const float xv = (m < n_tok) ? x[(uint64_t)m * in_dim + k0 + kk] : 0.0f;
            shA[j] = __float2half(xv);
        }
        for (uint32_t j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const uint32_t tile = j / (BK * BN);
            const uint32_t r = j - tile * (BK * BN);
            const uint32_t kk = r / BN;
            const uint32_t nn = r - kk * BN;
            const uint32_t n = n0 + tile * BN + nn;
            shB[j] = (n < out_dim) ? bhalf_kn[(uint64_t)(k0 + kk) * out_dim + n] : __float2half(0.0f);
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t mm = j / BK;
            const uint32_t kk = j - mm * BK;
            const uint32_t m = m0 + mm;
            const float xv = (m < n_tok) ? x[(uint64_t)m * in_dim + k0 + kk] : 0.0f;
            const half xh = __float2half(xv);
            shA[j] = __float2half(xv - __half2float(xh));
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if ((n_tok % BM) == 0u && (out_dim % (TILES_N * BN)) == 0u) {
        if (wave < TILES_N) {
            rocwmma::store_matrix_sync(out + (uint64_t)m0 * out_dim + n0 + wave * BN,
                                       acc,
                                       out_dim,
                                       rocwmma::mem_row_major);
        }
        return;
    }

    if (wave < TILES_N) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < TILES_N * BM * BN; j += blockDim.x) {
        const uint32_t tile = j / (BM * BN);
        const uint32_t r = j - tile * (BM * BN);
        const uint32_t mm = r / BN;
        const uint32_t nn = r - mm * BN;
        const uint32_t m = m0 + mm;
        const uint32_t n = n0 + tile * BN + nn;
        if (m < n_tok && n < out_dim) out[(uint64_t)m * out_dim + n] = shC[j];
    }
}

template <uint32_t TOK_TILE>
__global__ static void ds4_hip_matmul_q8_0_warp_rows_w32_toktile_2row_kernel(float *__restrict__ out,
                                                                             const unsigned char *__restrict__ w,
                                                                             const float *__restrict__ x,
                                                                             uint32_t n_blocks,
                                                                             uint32_t out_dim,
                                                                             uint32_t n_tok,
                                                                             uint64_t row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = (blockDim.x >> 5) << 1;
    const uint32_t o0 = blockIdx.x * rows_per_block + (wave << 1);
    const uint32_t o1 = o0 + 1u;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (o0 >= out_dim || t0 >= n_tok) return;
    const unsigned char *row0 = w + (uint64_t)o0 * row_bytes;
    const unsigned char *row1 = row0 + row_bytes;
    float acc0[TOK_TILE];
    float acc1[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) { acc0[u] = 0.0f; acc1[u] = 0.0f; }
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint64_t xoff = ((uint64_t)b << 5) + lane;
        const unsigned char *blk0 = row0 + (uint64_t)b * 34u;
        const float d0 = ds4_hip_q8_0_scale_broadcast_w32(blk0);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        const float wv0 = d0 * (float)q0;
        float wv1 = 0.0f;
        if (o1 < out_dim) {
            const unsigned char *blk1 = row1 + (uint64_t)b * 34u;
            const float d1 = ds4_hip_q8_0_scale_broadcast_w32(blk1);
            const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
            wv1 = d1 * (float)q1;
        }
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) {
                const float xv = x[(uint64_t)t * ((uint64_t)n_blocks << 5) + xoff];
                acc0[u] += wv0 * xv;
                acc1[u] += wv1 * xv;
            }
        }
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) { acc0[u] = ds4_hip_warp_reduce_sum(acc0[u]); acc1[u] = ds4_hip_warp_reduce_sum(acc1[u]); }
    if (lane == 0) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) {
                out[(uint64_t)t * out_dim + o0] = acc0[u];
                if (o1 < out_dim) out[(uint64_t)t * out_dim + o1] = acc1[u];
            }
        }
    }
}

__global__ static void ds4_hip_matmul_q8_0_warp_rows_kernel(float *out,
                                                            const unsigned char *w,
                                                            const float *x,
                                                            uint64_t in_dim,
                                                            uint64_t out_dim,
                                                            uint64_t row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    const uint64_t t = (uint64_t)blockIdx.y;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const unsigned char *row = w + o * row_bytes;
    const float *xr = x + t * in_dim;
    for (uint64_t i = lane; i < in_dim; i += (uint32_t)warpSize) {
        const uint64_t b = i >> 5;
        const uint64_t qlane = i & 31u;
        const unsigned char *blk = row + b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[qlane];
        acc += d * (float)q * xr[i];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[t * out_dim + o] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_sharedx_rows_kernel(float *out,
                                                               const unsigned char *w,
                                                               const float *x,
                                                               uint64_t in_dim,
                                                               uint64_t out_dim,
                                                               uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    const uint64_t t = (uint64_t)blockIdx.y;
    const float *xr = x + t * in_dim;
    for (uint64_t i = tid; i < in_dim; i += blockDim.x) shx[i] = xr[i];
    __syncthreads();

    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const unsigned char *row = w + o * row_bytes;
    for (uint64_t i = lane; i < in_dim; i += (uint32_t)warpSize) {
        const uint64_t b = i >> 5;
        const uint64_t qlane = i & 31u;
        const unsigned char *blk = row + b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[qlane];
        acc += d * (float)q * shx[i];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[t * out_dim + o] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_sharedx_rows_w32_kernel(float *__restrict__ out,
                                                                   const unsigned char *__restrict__ w,
                                                                   const float *__restrict__ x,
                                                                   uint32_t n_blocks,
                                                                   uint64_t out_dim,
                                                                   uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t in_dim = n_blocks << 5;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    float acc = 0.0f;
    const unsigned char *row = w + o * row_bytes;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(b << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[o] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_hc_partial16_w32_kernel(float *__restrict__ partial,
                                                                   const unsigned char *__restrict__ w,
                                                                   const float *__restrict__ x,
                                                                   uint32_t out_dim,
                                                                   uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t b0 = split << 4;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * row_bytes;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void ds4_hip_matmul_q8_repack_hc_partial16_w32_kernel(float *__restrict__ partial,
                                                                       const int8_t *__restrict__ q,
                                                                       const uint16_t *__restrict__ scales,
                                                                       const float *__restrict__ x,
                                                                       uint32_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t b0 = split << 4;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const int8_t *qrow = q + (uint64_t)row * ((uint64_t)gridDim.y << 9);
    const uint16_t *srow = scales + (uint64_t)row * ((uint64_t)gridDim.y << 4);
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(srow, b);
        const int8_t qv = qrow[((uint64_t)b << 5) + lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void ds4_hip_matmul_q8_0_hc_partial_w32_kernel(float *__restrict__ partial,
                                                                 const unsigned char *__restrict__ w,
                                                                 const float *__restrict__ x,
                                                                 uint32_t n_blocks,
                                                                 uint32_t out_dim,
                                                                 uint64_t row_bytes,
                                                                 uint32_t n_splits) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t chunk = (n_blocks + n_splits - 1u) / n_splits;
    const uint32_t b0 = split * chunk;
    const uint32_t b1 = min(n_blocks, b0 + chunk);
    const uint32_t chunk_blocks = b1 > b0 ? b1 - b0 : 0;
    for (uint32_t i = tid; i < (chunk_blocks << 5); i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t bb = 0; bb < chunk_blocks; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void ds4_hip_q8_partial_sum8_kernel(float *__restrict__ out,
                                                      const float *__restrict__ partial,
                                                      uint32_t out_dim) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 8u; s++) acc += partial[(uint64_t)s * out_dim + row];
    out[row] = acc;
}

__global__ static void ds4_hip_q8_partial_sum_kernel(float *__restrict__ out,
                                                     const float *__restrict__ partial,
                                                     uint32_t out_dim,
                                                     uint32_t n_splits) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) acc += partial[(uint64_t)s * out_dim + row];
    out[row] = acc;
}

__global__ static void ds4_hip_hc_expand_add_partial4_kernel(float *__restrict__ out_hc,
                                                             float *__restrict__ block_out,
                                                             const float *__restrict__ partial,
                                                             const float *__restrict__ block_add,
                                                             const float *__restrict__ residual_hc,
                                                             const float *__restrict__ split,
                                                             uint32_t out_dim,
                                                             uint32_t n_hc,
                                                             bool store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 4u; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float block = acc + block_add[row];
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = block * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void ds4_hip_hc_expand_partial16_kernel(float *__restrict__ out_hc,
                                                          float *__restrict__ block_out,
                                                          const float *__restrict__ partial,
                                                          const float *__restrict__ residual_hc,
                                                          const float *__restrict__ split,
                                                          uint32_t out_dim,
                                                          uint32_t n_hc,
                                                          bool store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 16u; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = acc * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void ds4_hip_hc_expand_partial_kernel(float *__restrict__ out_hc,
                                                        float *__restrict__ block_out,
                                                        const float *__restrict__ partial,
                                                        const float *__restrict__ residual_hc,
                                                        const float *__restrict__ split,
                                                        uint32_t out_dim,
                                                        uint32_t n_hc,
                                                        uint32_t n_splits,
                                                        bool store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = acc * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void ds4_hip_matmul_q8_0_hc_expand_w32_kernel(float *__restrict__ out_hc,
                                                                float *__restrict__ block_out,
                                                                const unsigned char *__restrict__ w,
                                                                const float *__restrict__ x,
                                                                const float *__restrict__ block_add,
                                                                const float *__restrict__ residual_hc,
                                                                const float *__restrict__ split,
                                                                uint32_t n_blocks,
                                                                uint64_t out_dim,
                                                                uint64_t row_bytes,
                                                                uint32_t n_hc,
                                                                bool store_block_out) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t in_dim = n_blocks << 5;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(b << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) {
        if (store_block_out) block_out[row] = acc;
        const float block = acc + (block_add ? block_add[row] : 0.0f);
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float v = block * post[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                v += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
            }
            out_hc[(uint64_t)dst * out_dim + row] = v;
        }
    }
}

__device__ static float ds4_hip_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ static float ds4_hip_silu(float x) {
    return x * ds4_hip_sigmoid(x);
}

__global__ static void ds4_hip_shared_gate_up_swiglu_q8_0_kernel(float *gate,
                                                                 float *up,
                                                                 float *mid,
                                                                 const unsigned char *wg,
                                                                 const unsigned char *wu,
                                                                 const float *x,
                                                                 uint64_t in_dim,
                                                                 uint64_t out_dim,
                                                                 uint64_t row_bytes,
                                                                 bool store_gate_up) {
    const uint64_t o = (uint64_t)blockIdx.x;
    if (o >= out_dim) return;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    const unsigned char *row_g = wg + o * row_bytes;
    const unsigned char *row_u = wu + o * row_bytes;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const uint64_t b = i >> 5;
        const uint64_t lane = i & 31u;
        const unsigned char *bg = row_g + b * 34u;
        const unsigned char *bu = row_u + b * 34u;
        const uint16_t dg_bits = (uint16_t)bg[0] | ((uint16_t)bg[1] << 8);
        const uint16_t du_bits = (uint16_t)bu[0] | ((uint16_t)bu[1] << 8);
        const int8_t qg = ((const int8_t *)(bg + 2u))[lane];
        const int8_t qu = ((const int8_t *)(bu + 2u))[lane];
        const float xv = x[i];
        acc_g += ds4_hip_f16_to_f32(dg_bits) * (float)qg * xv;
        acc_u += ds4_hip_f16_to_f32(du_bits) * (float)qu * xv;
    }
    const unsigned int tid = threadIdx.x;
    float g = acc_g;
    float u = acc_u;
    if (blockDim.x == warpSize) {
        g = ds4_hip_warp_reduce_sum(g);
        u = ds4_hip_warp_reduce_sum(u);
    } else {
        __shared__ float shg[256];
        __shared__ float shu[256];
        shg[tid] = g;
        shu[tid] = u;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) {
                shg[tid] += shg[tid + stride];
                shu[tid] += shu[tid + stride];
            }
            __syncthreads();
        }
        g = shg[0];
        u = shu[0];
    }
    if (tid == 0) {
        if (store_gate_up) {
            gate[o] = g;
            up[o] = u;
        }
        mid[o] = ds4_hip_silu(g) * u;
    }
}

__global__ static void ds4_hip_shared_gate_up_swiglu_q8_0_w32_kernel(float *gate,
                                                                     float *up,
                                                                     float *mid,
                                                                     const unsigned char *wg,
                                                                     const unsigned char *wu,
                                                                     const float *x,
                                                                     uint32_t n_blocks,
                                                                     uint64_t out_dim,
                                                                     uint64_t row_bytes,
                                                                     bool store_gate_up) {
    const uint64_t o = (uint64_t)blockIdx.x;
    if (o >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    const unsigned char *row_g = wg + o * row_bytes;
    const unsigned char *row_u = wu + o * row_bytes;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *bg = row_g + (uint64_t)b * 34u;
        const unsigned char *bu = row_u + (uint64_t)b * 34u;
        const uint16_t dg_bits = (uint16_t)bg[0] | ((uint16_t)bg[1] << 8);
        const uint16_t du_bits = (uint16_t)bu[0] | ((uint16_t)bu[1] << 8);
        const int8_t qg = ((const int8_t *)(bg + 2u))[lane];
        const int8_t qu = ((const int8_t *)(bu + 2u))[lane];
        const float xv = x[((uint64_t)b << 5) + lane];
        acc_g += ds4_hip_f16_to_f32(dg_bits) * (float)qg * xv;
        acc_u += ds4_hip_f16_to_f32(du_bits) * (float)qu * xv;
    }
    float g = ds4_hip_warp_reduce_sum(acc_g);
    float u = ds4_hip_warp_reduce_sum(acc_u);
    if (lane == 0) {
        if (store_gate_up) {
            gate[o] = g;
            up[o] = u;
        }
        mid[o] = ds4_hip_silu(g) * u;
    }
}

__global__ static void ds4_hip_shared_gate_up_partial16_w32_kernel(float *__restrict__ partial_g,
                                                                   float *__restrict__ partial_u,
                                                                   const unsigned char *__restrict__ wg,
                                                                   const unsigned char *__restrict__ wu,
                                                                   const float *__restrict__ x,
                                                                   uint64_t out_dim,
                                                                   uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t b0 = split << 4;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    const unsigned char *row_g = wg + o * row_bytes;
    const unsigned char *row_u = wu + o * row_bytes;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *bg = row_g + (uint64_t)b * 34u;
        const unsigned char *bu = row_u + (uint64_t)b * 34u;
        const float dg = ds4_hip_q8_0_scale_broadcast_w32(bg);
        const float du = ds4_hip_q8_0_scale_broadcast_w32(bu);
        const int8_t qg = ((const int8_t *)(bg + 2u))[lane];
        const int8_t qu = ((const int8_t *)(bu + 2u))[lane];
        const float xv = shx[(bb << 5) + lane];
        acc_g += dg * (float)qg * xv;
        acc_u += du * (float)qu * xv;
    }
    acc_g = ds4_hip_warp_reduce_sum(acc_g);
    acc_u = ds4_hip_warp_reduce_sum(acc_u);
    if (lane == 0) {
        partial_g[(uint64_t)split * out_dim + o] = acc_g;
        partial_u[(uint64_t)split * out_dim + o] = acc_u;
    }
}

__global__ static void ds4_hip_shared_gate_up_partial_sum8_kernel(float *__restrict__ gate,
                                                                  float *__restrict__ up,
                                                                  float *__restrict__ mid,
                                                                  const float *__restrict__ partial_g,
                                                                  const float *__restrict__ partial_u,
                                                                  uint32_t out_dim,
                                                                  bool store_gate_up) {
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float g = 0.0f;
    float u = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 8u; s++) {
        g += partial_g[(uint64_t)s * out_dim + o];
        u += partial_u[(uint64_t)s * out_dim + o];
    }
    if (store_gate_up) {
        gate[o] = g;
        up[o] = u;
    }
    mid[o] = ds4_hip_silu(g) * u;
}

__global__ static void ds4_hip_shared_gate_up_swiglu_q8_0_rows_w32_kernel(float *__restrict__ gate,
                                                                          float *__restrict__ up,
                                                                          float *__restrict__ mid,
                                                                          const unsigned char *__restrict__ wg,
                                                                          const unsigned char *__restrict__ wu,
                                                                          const float *__restrict__ x,
                                                                          uint32_t n_blocks,
                                                                          uint64_t out_dim,
                                                                          uint64_t row_bytes,
                                                                          bool store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint64_t o = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (o >= out_dim) return;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    const unsigned char *row_g = wg + o * row_bytes;
    const unsigned char *row_u = wu + o * row_bytes;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *bg = row_g + (uint64_t)b * 34u;
        const unsigned char *bu = row_u + (uint64_t)b * 34u;
        const float dg = ds4_hip_q8_0_scale_broadcast_w32(bg);
        const float du = ds4_hip_q8_0_scale_broadcast_w32(bu);
        const int8_t qg = ((const int8_t *)(bg + 2u))[lane];
        const int8_t qu = ((const int8_t *)(bu + 2u))[lane];
        const float xv = x[((uint64_t)b << 5) + lane];
        acc_g += dg * (float)qg * xv;
        acc_u += du * (float)qu * xv;
    }
    const float g = ds4_hip_warp_reduce_sum(acc_g);
    const float u = ds4_hip_warp_reduce_sum(acc_u);
    if (lane == 0) {
        if (store_gate_up) {
            gate[o] = g;
            up[o] = u;
        }
        mid[o] = ds4_hip_silu(g) * u;
    }
}

__global__ static void ds4_hip_swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    out[i] = ds4_hip_silu(g) * u * weight;
}

__global__ static void ds4_hip_add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ static void ds4_hip_repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    if (i < n) out[i] = row[i % n_embd];
}

__device__ static float ds4_hip_q8_0_row_value(const unsigned char *row, uint32_t i) {
    const uint32_t b = i >> 5;
    const uint32_t lane = i & 31u;
    const unsigned char *blk = row + (uint64_t)b * 34u;
    const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const int8_t q = ((const int8_t *)(blk + 2u))[lane];
    return ds4_hip_f16_to_f32(d_bits) * (float)q;
}

__global__ static void ds4_hip_embed_token_hc_q8_kernel(float *out, const unsigned char *w,
                                                        uint32_t token, uint32_t n_embd,
                                                        uint32_t n_hc, uint64_t row_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    const uint32_t d = (uint32_t)(i % n_embd);
    const unsigned char *row = w + (uint64_t)token * row_bytes;
    out[i] = ds4_hip_q8_0_row_value(row, d);
}

__global__ static void ds4_hip_embed_tokens_hc_q8_kernel(float *out, const int *tokens, const unsigned char *w,
                                                         uint32_t n_vocab, uint32_t n_tokens,
                                                         uint32_t n_embd, uint32_t n_hc,
                                                         uint64_t row_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;
    const uint32_t d = (uint32_t)(i % n_embd);
    const uint32_t t = (uint32_t)(i / ((uint64_t)n_hc * n_embd));
    int tok = tokens[t];
    if (tok < 0 || (uint32_t)tok >= n_vocab) tok = 0;
    const unsigned char *row = w + (uint64_t)(uint32_t)tok * row_bytes;
    out[i] = ds4_hip_q8_0_row_value(row, d);
}

__global__ static void ds4_hip_rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    const uint64_t r = (uint64_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float sh[256];
    const unsigned int tid = threadIdx.x;
    sh[tid] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)n + eps);
    float *yr = out + r * n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv;
}

__global__ static void ds4_hip_rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    const uint64_t r = (uint64_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float sh[256];
    const unsigned int tid = threadIdx.x;
    sh[tid] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)n + eps);
    float *yr = out + r * n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

__global__ static void ds4_hip_qkv_rms_norm_kernel(float *q_out, const float *q, const float *qw, uint32_t q_n,
                                                   float *kv_out, const float *kv, const float *kvw, uint32_t kv_n,
                                                   uint32_t rows, float eps) {
    const uint64_t r = (uint64_t)blockIdx.x;
    const bool do_q = blockIdx.y == 0;
    if (r >= rows) return;
    const uint32_t n = do_q ? q_n : kv_n;
    const float *xr = (do_q ? q : kv) + r * n;
    const float *wr = do_q ? qw : kvw;
    float *yr = (do_q ? q_out : kv_out) + r * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float sh[256];
    const unsigned int tid = threadIdx.x;
    sh[tid] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * wr[i];
}

__global__ static void ds4_hip_head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t rows = (uint64_t)n_tok * n_head;
    if (row >= rows) return;
    float *xr = x + row * head_dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float sh[256];
    const unsigned int tid = threadIdx.x;
    sh[tid] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= inv;
}

__device__ static void ds4_hip_hc_split_one(float *out, const float *mix, const float *scale, const float *base,
                                            uint32_t n_hc, uint32_t iters, float eps) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (uint32_t i = 0; i < n_hc; i++) {
        out[i] = ds4_hip_sigmoid(mix[i] * pre_scale + base[i]) + eps;
    }
    for (uint32_t i = 0; i < n_hc; i++) {
        const uint32_t off = n_hc + i;
        out[off] = 2.0f * ds4_hip_sigmoid(mix[off] * post_scale + base[off]);
    }

    float c[256];
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float row_max = -1.0e30f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const uint32_t off = 2u * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (uint32_t src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }
    for (uint32_t iter = 1; iter < iters; iter++) {
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (uint32_t src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (uint32_t src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }
    for (uint32_t i = 0; i < n_hc * n_hc; i++) out[2u * n_hc + i] = c[i];
}

__global__ static void ds4_hip_hc_split_kernel(float *out, const float *mix, const float *scale, const float *base,
                                               uint32_t n_hc, uint32_t iters, uint32_t rows, uint64_t mix_hc, float eps) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    ds4_hip_hc_split_one(out + (uint64_t)r * mix_hc, mix + (uint64_t)r * mix_hc, scale, base, n_hc, iters, eps);
}

__global__ static void ds4_hip_hc_split4_kernel(float *out, const float *mix, const float *scale, const float *base,
                                                uint32_t iters, uint32_t rows, uint64_t mix_hc, float eps) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float *o = out + (uint64_t)r * mix_hc;
    const float *m = mix + (uint64_t)r * mix_hc;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (uint32_t i = 0; i < 4u; i++) o[i] = ds4_hip_sigmoid(m[i] * pre_scale + base[i]) + eps;
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t off = 4u + i;
        o[off] = 2.0f * ds4_hip_sigmoid(m[off] * post_scale + base[off]);
    }

    float c[16];
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float row_max = -1.0e30f;
        for (uint32_t src = 0; src < 4u; src++) {
            const uint32_t idx = src + dst * 4u;
            const uint32_t off = 8u + idx;
            const float v = m[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t src = 0; src < 4u; src++) {
            const uint32_t idx = src + dst * 4u;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (uint32_t src = 0; src < 4u; src++) {
            const uint32_t idx = src + dst * 4u;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (uint32_t src = 0; src < 4u; src++) {
        float sum = 0.0f;
        for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
        const float inv = 1.0f / (sum + eps);
        for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
    }
    for (uint32_t iter = 1; iter < iters; iter++) {
        for (uint32_t dst = 0; dst < 4u; dst++) {
            float sum = 0.0f;
            for (uint32_t src = 0; src < 4u; src++) sum += c[src + dst * 4u];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t src = 0; src < 4u; src++) c[src + dst * 4u] *= inv;
        }
        for (uint32_t src = 0; src < 4u; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
        }
    }
    for (uint32_t i = 0; i < 16u; i++) o[8u + i] = c[i];
}

__global__ static void ds4_hip_hc_split4_weighted_sum_norm_kernel(float *out, float *norm_out, float *split,
                                                                  const float *mix, const float *residual_hc,
                                                                  const float *scale, const float *base,
                                                                  const float *norm_w, uint32_t n_embd,
                                                                  uint32_t iters, float eps, float norm_eps) {
    __shared__ float s[24];
    const uint32_t tid = threadIdx.x;
    if (tid == 0) {
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        for (uint32_t i = 0; i < 4u; i++) s[i] = ds4_hip_sigmoid(mix[i] * pre_scale + base[i]) + eps;
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t off = 4u + i;
            s[off] = 2.0f * ds4_hip_sigmoid(mix[off] * post_scale + base[off]);
        }
        float c[16];
        for (uint32_t dst = 0; dst < 4u; dst++) {
            float row_max = -1.0e30f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const uint32_t off = 8u + idx;
                const float v = mix[off] * comb_scale + base[off];
                c[idx] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }
            const float inv = 1.0f / row_sum;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                c[idx] = c[idx] * inv + eps;
            }
        }
        for (uint32_t src = 0; src < 4u; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
        }
        for (uint32_t iter = 1; iter < iters; iter++) {
            for (uint32_t dst = 0; dst < 4u; dst++) {
                float sum = 0.0f;
                for (uint32_t src = 0; src < 4u; src++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + eps);
                for (uint32_t src = 0; src < 4u; src++) c[src + dst * 4u] *= inv;
            }
            for (uint32_t src = 0; src < 4u; src++) {
                float sum = 0.0f;
                for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + eps);
                for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
            }
        }
        for (uint32_t i = 0; i < 16u; i++) s[8u + i] = c[i];
        for (uint32_t i = 0; i < 24u; i++) split[i] = s[i];
    }
    __syncthreads();

    float ss = 0.0f;
    for (uint32_t d = tid; d < n_embd; d += blockDim.x) {
        const float v = residual_hc[d] * s[0] +
                        residual_hc[(uint64_t)n_embd + d] * s[1] +
                        residual_hc[(uint64_t)2u * n_embd + d] * s[2] +
                        residual_hc[(uint64_t)3u * n_embd + d] * s[3];
        out[d] = v;
        ss += v * v;
    }
    const float total = ds4_hip_block_reduce_sum(ss);
    const float inv = rsqrtf(total / (float)n_embd + norm_eps);
    for (uint32_t d = tid; d < n_embd; d += blockDim.x) norm_out[d] = out[d] * inv * norm_w[d];
}

__global__ static void ds4_hip_hc_weighted_sum_kernel(float *out, const float *residual_hc, const float *weights,
                                                      uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                                                      uint64_t weight_stride_floats) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (i >= n) return;
    const uint32_t t = (uint32_t)(i / n_embd);
    const uint32_t d = (uint32_t)(i % n_embd);
    const float *x = residual_hc + (uint64_t)t * n_hc * n_embd;
    const float *w = weights + (uint64_t)t * weight_stride_floats;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) acc += x[(uint64_t)h * n_embd + d] * w[h];
    out[i] = acc;
}

__global__ static void ds4_hip_output_hc_weights_kernel(float *out, const float *pre, const float *scale,
                                                        const float *base, uint32_t n_hc, float eps) {
    const uint32_t i = threadIdx.x;
    if (i < n_hc) out[i] = ds4_hip_sigmoid(pre[i] * scale[0] + base[i]) + eps;
}

__global__ static void ds4_hip_hc_expand_kernel(float *out_hc, const float *block_out, const float *block_add,
                                                const float *residual_hc, const float *split_or_post,
                                                const float *comb_arg, uint32_t n_embd, uint32_t n_hc,
                                                uint32_t n_tokens, bool split_layout,
                                                uint64_t post_stride, uint64_t comb_stride) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t row_elems = (uint64_t)n_embd * n_hc;
    const uint64_t n = (uint64_t)n_tokens * row_elems;
    if (i >= n) return;
    const uint32_t t = (uint32_t)(i / row_elems);
    const uint64_t local = i - (uint64_t)t * row_elems;
    const uint32_t dst = (uint32_t)(local / n_embd);
    const uint32_t d = (uint32_t)(local % n_embd);
    const float *post = split_layout ? split_or_post + (uint64_t)t * post_stride + n_hc
                                     : split_or_post + (uint64_t)t * post_stride;
    const float *comb = split_layout ? split_or_post + (uint64_t)t * post_stride + 2u * n_hc
                                     : comb_arg + (uint64_t)t * comb_stride;
    float block = block_out[(uint64_t)t * n_embd + d];
    if (block_add) block += block_add[(uint64_t)t * n_embd + d];
    float acc = block * post[dst];
    const float *res = residual_hc + (uint64_t)t * row_elems;
    for (uint32_t src = 0; src < n_hc; src++) {
        acc += comb[dst + src * n_hc] * res[(uint64_t)src * n_embd + d];
    }
    out_hc[i] = acc;
}

__host__ __device__ static uint16_t ds4_hip_f32_to_f16_bits(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

__device__ static float ds4_hip_rope_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static float ds4_hip_rope_corr_dim(int n_dims, uint32_t n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(base));
}

__global__ static void ds4_hip_rope_tail_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                                                uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
                                                float freq_base, float freq_scale, float ext_factor, float attn_factor,
                                                float beta_fast, float beta_slow) {
    const uint64_t pair_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs_per_head = n_rot / 2u;
    const uint64_t total = (uint64_t)n_tok * n_head * pairs_per_head;
    if (pair_id >= total) return;
    const uint32_t pair = (uint32_t)(pair_id % pairs_per_head);
    const uint64_t htmp = pair_id / pairs_per_head;
    const uint32_t h = (uint32_t)(htmp % n_head);
    const uint32_t t = (uint32_t)(htmp / n_head);
    const uint32_t i = pair * 2u;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t pos = pos0 + t;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float theta_extrap = (float)pos * powf(theta_scale, (float)pair);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float start = floorf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_fast, freq_base));
        const float end = ceilf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_slow, freq_base));
        const float low = fmaxf(0.0f, start);
        const float high = fminf((float)(n_rot - 1u), end);
        const float ramp_mix = ds4_hip_rope_ramp(low, high, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale;
    const float s = sin_sign * sinf(theta) * mscale;
    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

/* Token-major RoPE variant inspired by AITER's DSV4 fused-RoPE kernels: compute
 * the 32 RoPE sin/cos pairs once per token, not once per head.  This keeps the
 * same math as ds4_hip_rope_tail_kernel but removes a large amount of duplicate
 * pow/cos/sin work for Q and attention-output tensors with 64 heads. */
__global__ static void ds4_hip_rope_tail_token_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                                                      uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
                                                      float freq_base, float freq_scale, float ext_factor, float attn_factor,
                                                      float beta_fast, float beta_slow) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tok || n_head == 0 || n_rot == 0 || n_rot > head_dim) return;
    const uint32_t pairs_per_head = n_rot >> 1;
    extern __shared__ float sh[];
    float *cs = sh;
    float *ss = sh + pairs_per_head;
    const uint32_t pos = pos0 + t;
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float low = 0.0f;
    float high = 0.0f;
    float ext_mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float start = floorf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_fast, freq_base));
        const float end = ceilf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_slow, freq_base));
        low = fmaxf(0.0f, start);
        high = fminf((float)(n_rot - 1u), end);
        ext_mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    for (uint32_t pair = tid; pair < pairs_per_head; pair += blockDim.x) {
        const uint32_t i = pair << 1;
        float theta_extrap = (float)pos * powf(theta_scale, (float)pair);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix = ds4_hip_rope_ramp(low, high, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale = ext_mscale;
        }
        cs[pair] = cosf(theta) * mscale;
        ss[pair] = sin_sign * sinf(theta) * mscale;
    }
    __syncthreads();
    const uint32_t pairs_total = n_head * pairs_per_head;
    for (uint32_t idx = tid; idx < pairs_total; idx += blockDim.x) {
        const uint32_t pair = idx % pairs_per_head;
        const uint32_t h = idx / pairs_per_head;
        const uint32_t i = pair << 1;
        float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
        const float c = cs[pair];
        const float s = ss[pair];
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float ds4_hip_e4m3_value(int q) {
    const int exp = q >> 3;
    const int mant = q & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)(exp - 7));
}

__device__ static float ds4_hip_e4m3_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_hip_e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ds4_hip_e4m3_value(best));
        const float next_diff = fabsf(ax - ds4_hip_e4m3_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * ds4_hip_e4m3_value(best);
}

__global__ static void ds4_hip_fp8_kv_quant_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t off = grp * 64u;
    if (row >= n_tok || off >= n_nope) return;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float sh[64];
    float av = 0.0f;
    if (threadIdx.x < 64u && off + threadIdx.x < n_nope) av = fabsf(xr[off + threadIdx.x]);
    sh[threadIdx.x] = av;
    __syncthreads();
    for (uint32_t stride = 32; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride && sh[threadIdx.x + stride] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + stride];
        __syncthreads();
    }
    float amax = sh[0];
    if (amax < 1.0e-4f) amax = 1.0e-4f;
    const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
    if (threadIdx.x < 64u && off + threadIdx.x < n_nope) {
        float v = xr[off + threadIdx.x] / scale;
        v = fminf(448.0f, fmaxf(-448.0f, v));
        xr[off + threadIdx.x] = ds4_hip_e4m3_dequant(v) * scale;
    }
}

__global__ static void ds4_hip_fill_f32_kernel(float *x, uint64_t n, float v) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__device__ static float ds4_hip_ape_value(const unsigned char *ape, uint32_t ape_type,
                                          uint32_t width, uint32_t pos_mod, uint32_t x) {
    if (ape_type == 1u) {
        const uint16_t *p = (const uint16_t *)ape;
        return ds4_hip_f16_to_f32(p[(uint64_t)pos_mod * width + x]);
    }
    if (ape_type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *row = ape + (uint64_t)pos_mod * row_bytes;
        const uint64_t b = x >> 5;
        const uint64_t lane = x & 31u;
        const unsigned char *blk = row + b * 34u;
        const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        return ds4_hip_f16_to_f32(d_bits) * (float)q;
    }
    return ((const float *)ape)[(uint64_t)pos_mod * width + x];
}

__global__ static void ds4_hip_compressor_store_rows_kernel(float *state_kv, float *state_score,
                                                            const float *kv, const float *sc,
                                                            const unsigned char *ape,
                                                            uint32_t ape_type, uint32_t width,
                                                            uint32_t ratio, uint32_t pos0,
                                                            uint32_t n_rows, uint32_t dst_row0) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_rows * width;
    if (idx >= total || ratio == 0) return;
    const uint32_t t = (uint32_t)(idx / width);
    const uint32_t j = (uint32_t)(idx % width);
    const uint32_t pos_mod = (pos0 + t) % ratio;
    const uint64_t dst = (uint64_t)(dst_row0 + t) * width + j;
    state_kv[dst] = kv[idx];
    state_score[dst] = sc[idx] + ds4_hip_ape_value(ape, ape_type, width, pos_mod, j);
}

__global__ static void ds4_hip_compressor_store_batch_kernel(float *state_kv, float *state_score,
                                                             const float *kv, const float *sc,
                                                             const unsigned char *ape,
                                                             uint32_t ape_type, uint32_t width,
                                                             uint32_t ratio, uint32_t pos0,
                                                             uint32_t n_tokens) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * width;
    if (idx >= total || ratio == 0) return;
    const uint32_t t = (uint32_t)(idx / width);
    const uint32_t j = (uint32_t)(idx % width);
    const uint32_t pos_mod = (pos0 + t) % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const uint64_t dst = (uint64_t)dst_row * width + j;
    state_kv[dst] = kv[idx];
    state_score[dst] = sc[idx] + ds4_hip_ape_value(ape, ape_type, width, pos_mod, j);
}

__global__ static void ds4_hip_compressor_pool_prefill_kernel(float *out,
                                                              const float *state_kv,
                                                              const float *state_score,
                                                              const float *kv,
                                                              const float *sc,
                                                              const unsigned char *ape,
                                                              uint32_t ape_type, uint32_t head_dim,
                                                              uint32_t ratio, uint32_t pos0,
                                                              uint32_t n_comp, bool replay_prev) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_comp * head_dim;
    if (idx >= total || ratio == 0) return;
    const uint32_t j = (uint32_t)(idx % head_dim);
    const uint32_t c = (uint32_t)(idx / head_dim);
    const uint32_t width = ratio == 4u ? 2u * head_dim : head_dim;
    float max_score = -1.0e30f;

    if (ratio == 4u) {
        for (uint32_t r = 0; r < 8u; r++) {
            float s = -1.0e30f;
            if (r < 4u) {
                if (c > 0u) {
                    const uint32_t tok = (c - 1u) * 4u + r;
                    const uint32_t lane = j;
                    s = sc[(uint64_t)tok * width + lane] +
                        ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) & 3u, lane);
                } else if (replay_prev && state_score) {
                    s = state_score[(uint64_t)r * width + j];
                }
            } else {
                const uint32_t tok = c * 4u + (r - 4u);
                const uint32_t lane = head_dim + j;
                s = sc[(uint64_t)tok * width + lane] +
                    ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) & 3u, lane);
            }
            if (s > max_score) max_score = s;
        }
        if (max_score <= -5.0e29f) {
            out[idx] = 0.0f;
            return;
        }
        float denom = 0.0f;
        float sum = 0.0f;
        for (uint32_t r = 0; r < 8u; r++) {
            float s = -1.0e30f;
            float v = 0.0f;
            if (r < 4u) {
                if (c > 0u) {
                    const uint32_t tok = (c - 1u) * 4u + r;
                    const uint32_t lane = j;
                    s = sc[(uint64_t)tok * width + lane] +
                        ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) & 3u, lane);
                    v = kv[(uint64_t)tok * width + lane];
                } else if (replay_prev && state_score && state_kv) {
                    s = state_score[(uint64_t)r * width + j];
                    v = state_kv[(uint64_t)r * width + j];
                }
            } else {
                const uint32_t tok = c * 4u + (r - 4u);
                const uint32_t lane = head_dim + j;
                s = sc[(uint64_t)tok * width + lane] +
                    ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) & 3u, lane);
                v = kv[(uint64_t)tok * width + lane];
            }
            if (s > -5.0e29f) {
                const float w = expf(s - max_score);
                denom += w;
                sum += w * v;
            }
        }
        out[idx] = denom > 0.0f ? sum / denom : 0.0f;
        return;
    }

    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t tok = c * ratio + r;
        const float s = sc[(uint64_t)tok * width + j] +
            ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) % ratio, j);
        if (s > max_score) max_score = s;
    }
    float denom = 0.0f;
    float sum = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t tok = c * ratio + r;
        const float s = sc[(uint64_t)tok * width + j] +
            ds4_hip_ape_value(ape, ape_type, width, (pos0 + tok) % ratio, j);
        const float w = expf(s - max_score);
        denom += w;
        sum += w * kv[(uint64_t)tok * width + j];
    }
    out[idx] = denom > 0.0f ? sum / denom : 0.0f;
}

__global__ static void ds4_hip_compressor_pool_state_kernel(float *out,
                                                            const float *state_kv,
                                                            const float *state_score,
                                                            uint32_t head_dim, uint32_t ratio,
                                                            uint32_t comp_row) {
    const uint32_t j = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= head_dim || ratio == 0) return;
    const uint32_t width = ratio == 4u ? 2u * head_dim : head_dim;
    const uint32_t rows = ratio == 4u ? 8u : ratio;
    float max_score = -1.0e30f;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t lane = (ratio == 4u && r >= 4u) ? head_dim + j : j;
        const float s = state_score[(uint64_t)r * width + lane];
        if (s > max_score) max_score = s;
    }
    if (max_score <= -5.0e29f) {
        out[(uint64_t)comp_row * head_dim + j] = 0.0f;
        return;
    }
    float denom = 0.0f;
    float sum = 0.0f;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t lane = (ratio == 4u && r >= 4u) ? head_dim + j : j;
        const float s = state_score[(uint64_t)r * width + lane];
        if (s <= -5.0e29f) continue;
        const float w = expf(s - max_score);
        denom += w;
        sum += w * state_kv[(uint64_t)r * width + lane];
    }
    out[(uint64_t)comp_row * head_dim + j] = denom > 0.0f ? sum / denom : 0.0f;
}

__global__ static void ds4_hip_ratio4_shift_kernel(float *state_kv, float *state_score, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = 4ull * width;
    if (i < n) {
        state_kv[i] = state_kv[n + i];
        state_score[i] = state_score[n + i];
    }
}

__global__ static void ds4_hip_rope_tail_stride_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                                                       uint32_t n_rot, uint32_t pos0, uint32_t pos_stride,
                                                       uint32_t n_ctx_orig, bool inverse,
                                                       float freq_base, float freq_scale, float ext_factor,
                                                       float attn_factor, float beta_fast, float beta_slow) {
    const uint64_t pair_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs_per_head = n_rot / 2u;
    const uint64_t total = (uint64_t)n_tok * n_head * pairs_per_head;
    if (pair_id >= total) return;
    const uint32_t pair = (uint32_t)(pair_id % pairs_per_head);
    const uint64_t htmp = pair_id / pairs_per_head;
    const uint32_t h = (uint32_t)(htmp % n_head);
    const uint32_t t = (uint32_t)(htmp / n_head);
    const uint32_t i = pair * 2u;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t pos = pos0 + t * pos_stride;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float theta_extrap = (float)pos * powf(theta_scale, (float)pair);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float start = floorf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_fast, freq_base));
        const float end = ceilf(ds4_hip_rope_corr_dim((int)n_rot, n_ctx_orig, beta_slow, freq_base));
        const float low = fmaxf(0.0f, start);
        const float high = fminf((float)(n_rot - 1u), end);
        const float ramp_mix = ds4_hip_rope_ramp(low, high, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale;
    const float s = sin_sign * sinf(theta) * mscale;
    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void ds4_hip_store_raw_kernel(float *raw, const float *kv, uint32_t raw_cap,
                                                uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                                                bool fp8_first, uint32_t n_rot) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    if (i >= n) return;
    const uint32_t t = (uint32_t)(i / head_dim);
    const uint32_t d = (uint32_t)(i % head_dim);
    float v = kv[i];
    if (fp8_first && d < head_dim - n_rot) {
        /* Caller already ran the in-place FP8 kernel for batch paths; decode fused paths use this after it. */
    }
    v = ds4_hip_f16_to_f32(ds4_hip_f32_to_f16_bits(v));
    raw[(uint64_t)((pos0 + t) % raw_cap) * head_dim + d] = v;
}

__global__ static void ds4_hip_attention_prefill_raw_kernel(float *heads, const float *q, const float *raw_kv,
                                                            const float *sinks, uint32_t n_tokens, uint32_t window,
                                                            uint32_t n_head, uint32_t head_dim) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_head * head_dim;
    if (idx >= total) return;
    const uint32_t d = (uint32_t)(idx % head_dim);
    const uint64_t th = idx / head_dim;
    const uint32_t h = (uint32_t)(th % n_head);
    const uint32_t t = (uint32_t)(th / n_head);
    const uint32_t raw_count = (t + 1u) < window ? (t + 1u) : window;
    const uint32_t raw_start = t + 1u - raw_count;
    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    float max_score = sinks[h];
    for (uint32_t r = 0; r < raw_count; r++) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }
    float denom = expf(sinks[h] - max_score);
    float acc = 0.0f;
    for (uint32_t r = 0; r < raw_count; r++) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        const float w = expf(s * scale - max_score);
        denom += w;
        acc += w * kv[d];
    }
    heads[idx] = acc / denom;
}

/* FlashAttention-style fixed-shape raw SWA prefill for DeepSeek-V4 Flash.
 * One block computes one (token, head) row.  The old naive kernel computed the
 * same Q.K dot product once for every output dimension; this computes each
 * score once, normalizes the <=128-token SWA window in shared memory, and then
 * reuses the weights for all value dimensions. */
__global__ static void ds4_hip_attention_prefill_raw_scores_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head) return;

    extern __shared__ float sh[];
    float *scores = sh;                 /* window floats */
    float *red = scores + window;       /* blockDim.x floats */
    float *meta = red + blockDim.x;     /* [max_score, denom] */

    const uint32_t raw_count = (t + 1u) < window ? (t + 1u) : window;
    const uint32_t raw_start = t + 1u - raw_count;
    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;

    float local_max = sinks[h];
    for (uint32_t r = 0; r < raw_count; r++) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float acc = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += blockDim.x) acc += qh[i] * kv[i];
        red[tid] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float s = red[0] * scale;
            scores[r] = s;
            if (s > local_max) local_max = s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = expf(sinks[h] - local_max);
        for (uint32_t r = 0; r < raw_count; r++) {
            const float w = expf(scores[r] - local_max);
            scores[r] = w;
            denom += w;
        }
        meta[0] = local_max;
        meta[1] = denom;
    }
    __syncthreads();

    const float denom = meta[1];
    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += scores[r] * raw_kv[(uint64_t)(raw_start + r) * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_indexer_scores_kernel(float *scores, const float *q, const float *weights,
                                                     const float *index_comp, uint32_t n_comp,
                                                     uint32_t n_tokens, uint32_t n_head,
                                                     uint32_t head_dim, float scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    float acc = 0.0f;
    const uint32_t total = n_head * head_dim;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t h = i / head_dim;
        const uint32_t d = i - h * head_dim;
        acc += q[((uint64_t)t * n_head + h) * head_dim + d] *
               index_comp[(uint64_t)c * head_dim + d] *
               weights[(uint64_t)t * n_head + h];
    }
    ds4_hip_block_reduce_store(scores, acc * scale, (uint64_t)t * n_comp + c);
}

__global__ static void ds4_hip_indexer_select_all_kernel(int *selected, uint32_t n_comp,
                                                         uint32_t n_tokens, uint32_t top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_comp;
    if (i >= n) return;
    const uint32_t t = (uint32_t)(i / n_comp);
    const uint32_t c = (uint32_t)(i - (uint64_t)t * n_comp);
    selected[(uint64_t)t * top_k + c] = (int)c;
}

__global__ static void ds4_hip_indexer_top1_parallel_kernel(int *selected, const float *scores,
                                                            uint32_t n_comp, uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    float best = -3.4e38f;
    int best_i = 0;
    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
        const float s = row[c];
        if (s > best || (s == best && (int)c < best_i)) {
            best = s;
            best_i = (int)c;
        }
    }
    __shared__ float shv[256];
    __shared__ int shi[256];
    shv[tid] = best;
    shi[tid] = best_i;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (tid < stride) {
            const float ov = shv[tid + stride];
            const int oi = shi[tid + stride];
            if (ov > shv[tid] || (ov == shv[tid] && oi < shi[tid])) {
                shv[tid] = ov;
                shi[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) selected[t] = shi[0];
}

__global__ static void ds4_hip_indexer_topk_kernel(int *selected, const float *scores,
                                                   uint32_t n_comp, uint32_t n_tokens,
                                                   uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    for (uint32_t k = 0; k < top_k; k++) {
        float best = -3.4e38f;
        int best_i = 0;
        for (uint32_t c = 0; c < n_comp; c++) {
            const float s = scores[(uint64_t)t * n_comp + c];
            bool used = false;
            for (uint32_t p = 0; p < k; p++) {
                if (selected[(uint64_t)t * top_k + p] == (int)c) { used = true; break; }
            }
            if (!used && (s > best || (s == best && (int)c < best_i))) {
                best = s;
                best_i = (int)c;
            }
        }
        selected[(uint64_t)t * top_k + k] = best_i;
    }
}

__global__ static void ds4_hip_indexer_topk_iter_parallel_kernel(int *__restrict__ selected,
                                                                 const float *__restrict__ scores,
                                                                 uint32_t n_comp,
                                                                 uint32_t n_tokens,
                                                                 uint32_t top_k) {
    extern __shared__ unsigned char smem[];
    float *vals = reinterpret_cast<float *>(smem);
    float *best_vals = vals + n_comp;
    int *best_idx = reinterpret_cast<int *>(best_vals + blockDim.x);
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t c = tid; c < n_comp; c += blockDim.x) vals[c] = row[c];
    __syncthreads();

    for (uint32_t k = 0; k < top_k; k++) {
        float best = -3.4e38f;
        int best_i = 0;
        for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
            const float s = vals[c];
            if (s > best || (s == best && (int)c < best_i)) {
                best = s;
                best_i = (int)c;
            }
        }
        best_vals[tid] = best;
        best_idx[tid] = best_i;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) {
                const float ov = best_vals[tid + stride];
                const int oi = best_idx[tid + stride];
                if (ov > best_vals[tid] || (ov == best_vals[tid] && oi < best_idx[tid])) {
                    best_vals[tid] = ov;
                    best_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int bi = best_idx[0];
            selected[(uint64_t)t * top_k + k] = bi;
            if (bi >= 0 && (uint32_t)bi < n_comp) vals[(uint32_t)bi] = -3.4e38f;
        }
        __syncthreads();
    }
}

__global__ static void ds4_hip_topk_mask_kernel(float *mask, const int *topk,
                                                uint32_t n_comp, uint32_t n_tokens,
                                                uint32_t top_k) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_comp;
    if (i >= n) return;
    const uint32_t c = (uint32_t)(i % n_comp);
    const uint32_t t = (uint32_t)(i / n_comp);
    float v = -1.0e30f;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == (int)c) { v = 0.0f; break; }
    }
    mask[i] = v;
}

__global__ static void ds4_hip_attention_decode_mixed_one_kernel(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const float *sinks,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t use_mask, uint32_t n_head, uint32_t head_dim) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_head * head_dim;
    if (idx >= total) return;
    const uint32_t d = (uint32_t)(idx % head_dim);
    const uint32_t h = (uint32_t)(idx / head_dim);
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = rsqrtf((float)head_dim);
    float max_score = sinks[h];
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (use_mask && comp_mask && comp_mask[c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (use_mask && comp_mask) s += comp_mask[c];
        if (s > max_score) max_score = s;
    }
    float denom = expf(sinks[h] - max_score);
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        const float w = expf(s * scale - max_score);
        denom += w;
        acc += w * kv[d];
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (use_mask && comp_mask && comp_mask[c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (use_mask && comp_mask) s += comp_mask[c];
        const float w = expf(s - max_score);
        denom += w;
        acc += w * kv[d];
    }
    heads[idx] = acc / denom;
}

__global__ static void ds4_hip_attention_decode_mixed_one_fast_kernel(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const float *sinks,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t use_mask, uint32_t n_head, uint32_t head_dim) {
    const uint32_t h = (uint32_t)blockIdx.x;
    if (h >= n_head) return;
    extern __shared__ float scores[];
    const uint32_t tid = threadIdx.x;
    const uint32_t n_rows = n_raw + n_comp;
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float local_max = sinks[h];
    for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
        float s = -3.4e38f;
        if (!(use_mask && comp_mask && comp_mask[c] <= -5.0e29f)) {
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * kv[i];
            s = dot * scale;
            if (use_mask && comp_mask) s += comp_mask[c];
        }
        scores[n_raw + c] = s;
        local_max = fmaxf(local_max, s);
    }
    float max_score = ds4_hip_block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
        float w = expf(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0) local_sum += expf(sinks[h] - max_score);
    float denom = ds4_hip_block_reduce_sum(local_sum);
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
    }
}

__global__ static void ds4_hip_attention_decode_raw_batch_scores_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head || n_raw == 0 || raw_cap == 0) return;

    extern __shared__ float sh[];
    float *scores = sh;
    float *red = scores + window;
    float *meta = red + blockDim.x;

    const uint32_t qpos = pos0 + t;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t min_kpos = (window != 0 && qpos + 1u > window) ? qpos + 1u - window : 0u;
    const uint32_t vis_first = first_raw_pos > min_kpos ? first_raw_pos : min_kpos;
    const uint32_t vis_last = qpos;
    const uint32_t raw_offset = vis_first - first_raw_pos;
    const uint32_t raw_count = vis_last >= vis_first ? (vis_last - vis_first + 1u) : 0u;
    if (raw_count == 0 || raw_count > window) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    float local_max = sinks[h];
    for (uint32_t r = 0; r < raw_count; r++) {
        const uint32_t row = (raw_start + raw_offset + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float acc = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += blockDim.x) acc += qh[i] * kv[i];
        red[tid] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float s = red[0] * scale;
            scores[r] = s;
            if (s > local_max) local_max = s;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float denom = expf(sinks[h] - local_max);
        for (uint32_t r = 0; r < raw_count; r++) {
            const float w = expf(scores[r] - local_max);
            scores[r] = w;
            denom += w;
        }
        meta[0] = denom;
    }
    __syncthreads();

    const float denom = meta[0];
    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t row = (raw_start + raw_offset + r) % raw_cap;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_attention_decode_mixed_batch_warprows_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ comp_mask,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        bool use_comp_mask) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    if (t >= n_tokens || h >= n_head || raw_cap == 0 || ratio == 0 || rows_per_block == 0) return;

    extern __shared__ float sh[];
    float *scores = sh;
    const uint32_t score_cap = (window ? window : n_raw) + n_comp;
    float *qsh = scores + score_cap;
    const uint32_t qpos = pos0 + t;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t min_kpos = (window != 0 && qpos + 1u > window) ? qpos + 1u - window : 0u;
    const uint32_t vis_first = first_raw_pos > min_kpos ? first_raw_pos : min_kpos;
    const uint32_t raw_offset = vis_first - first_raw_pos;
    const uint32_t raw_count = qpos >= vis_first ? (qpos - vis_first + 1u) : 0u;
    uint32_t comp_visible = (qpos + 1u) / ratio;
    if (comp_visible > n_comp) comp_visible = n_comp;
    const uint32_t n_scores = raw_count + comp_visible;
    if (n_scores == 0 || n_scores > score_cap) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) qsh[i] = qh[i];
    __syncthreads();

    for (uint32_t base = 0; base < n_scores; base += rows_per_block) {
        const uint32_t row = base + wave;
        if (row < n_scores) {
            const bool is_comp = row >= raw_count;
            const uint32_t k = is_comp ? (row - raw_count) : row;
            float mask = 0.0f;
            bool skip = false;
            const float *kv = nullptr;
            if (is_comp) {
                if (use_comp_mask && comp_mask) {
                    mask = comp_mask[(uint64_t)t * n_comp + k];
                    skip = mask <= -5.0e29f;
                }
                kv = comp_kv + (uint64_t)k * head_dim;
            } else {
                const uint32_t phys = (raw_start + raw_offset + k) % raw_cap;
                kv = raw_kv + (uint64_t)phys * head_dim;
            }
            float acc = 0.0f;
            if (!skip) {
                for (uint32_t i = lane; i < head_dim; i += (uint32_t)warpSize) acc += qsh[i] * kv[i];
            }
            acc = ds4_hip_warp_reduce_sum(acc);
            if (lane == 0) scores[row] = skip ? -1.0e30f : (acc * scale + mask);
        }
        __syncthreads();
    }

    float lmax = tid == 0 ? sinks[h] : -3.4e38f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) lmax = fmaxf(lmax, scores[i]);
    const float max_score = ds4_hip_block_reduce_max(lmax);

    float lsum = tid == 0 ? expf(sinks[h] - max_score) : 0.0f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) {
        const float w = expf(scores[i] - max_score);
        scores[i] = w;
        lsum += w;
    }
    const float denom = ds4_hip_block_reduce_sum(lsum);

    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t phys = (raw_start + raw_offset + r) % raw_cap;
            acc += scores[r] * raw_kv[(uint64_t)phys * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_visible; c++) {
            acc += scores[raw_count + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_attention_indexed_mixed_batch_warprows_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const int *__restrict__ topk,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    if (t >= n_tokens || h >= n_head || raw_cap == 0 || ratio == 0 || rows_per_block == 0 || top_k == 0) return;

    extern __shared__ float sh[];
    float *scores = sh;
    const uint32_t raw_score_cap = window ? window : n_raw;
    const uint32_t score_cap = raw_score_cap + top_k;
    float *qsh = scores + score_cap;

    const uint32_t qpos = pos0 + t;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t min_kpos = (window != 0 && qpos + 1u > window) ? qpos + 1u - window : 0u;
    const uint32_t vis_first = first_raw_pos > min_kpos ? first_raw_pos : min_kpos;
    const uint32_t raw_offset = vis_first - first_raw_pos;
    const uint32_t raw_count = qpos >= vis_first ? (qpos - vis_first + 1u) : 0u;
    uint32_t comp_visible = (qpos + 1u) / ratio;
    if (comp_visible > n_comp) comp_visible = n_comp;
    const uint32_t n_scores = raw_count + top_k;
    if (raw_count > raw_score_cap || n_scores == 0 || n_scores > score_cap) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) qsh[i] = qh[i];
    __syncthreads();

    for (uint32_t base = 0; base < n_scores; base += rows_per_block) {
        const uint32_t row = base + wave;
        if (row < n_scores) {
            const bool is_comp = row >= raw_count;
            const uint32_t k = is_comp ? (row - raw_count) : row;
            const float *kv = nullptr;
            bool skip = false;
            if (is_comp) {
                const int ci = topk[(uint64_t)t * top_k + k];
                skip = ci < 0 || (uint32_t)ci >= n_comp || (uint32_t)ci >= comp_visible;
                kv = skip ? nullptr : (comp_kv + (uint64_t)(uint32_t)ci * head_dim);
            } else {
                const uint32_t phys = (raw_start + raw_offset + k) % raw_cap;
                kv = raw_kv + (uint64_t)phys * head_dim;
            }
            float acc = 0.0f;
            if (!skip) {
                for (uint32_t i = lane; i < head_dim; i += (uint32_t)warpSize) acc += qsh[i] * kv[i];
            }
            acc = ds4_hip_warp_reduce_sum(acc);
            if (lane == 0) scores[row] = skip ? -1.0e30f : (acc * scale);
        }
        __syncthreads();
    }

    float lmax = tid == 0 ? sinks[h] : -3.4e38f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) lmax = fmaxf(lmax, scores[i]);
    const float max_score = ds4_hip_block_reduce_max(lmax);

    float lsum = tid == 0 ? expf(sinks[h] - max_score) : 0.0f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) {
        const float w = expf(scores[i] - max_score);
        scores[i] = w;
        lsum += w;
    }
    const float denom = ds4_hip_block_reduce_sum(lsum);

    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t phys = (raw_start + raw_offset + r) % raw_cap;
            acc += scores[r] * raw_kv[(uint64_t)phys * head_dim + d];
        }
        for (uint32_t u = 0; u < top_k; u++) {
            const int ci = topk[(uint64_t)t * top_k + u];
            if (ci < 0 || (uint32_t)ci >= n_comp || (uint32_t)ci >= comp_visible) continue;
            acc += scores[raw_count + u] * comp_kv[(uint64_t)(uint32_t)ci * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_attention_decode_batch_mixed_kernel(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const int *topk, const float *sinks,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim, uint32_t mode) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_head * head_dim;
    if (idx >= total) return;
    const uint32_t d = (uint32_t)(idx % head_dim);
    const uint64_t th = idx / head_dim;
    const uint32_t h = (uint32_t)(th % n_head);
    const uint32_t t = (uint32_t)(th / n_head);
    const uint32_t qpos = pos0 + t;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);
    float max_score = sinks[h];

    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t kpos = first_raw_pos + r;
        if (kpos > qpos) continue;
        if (window != 0 && qpos - kpos >= window) continue;
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }
    const uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
    const uint32_t scan_comp = mode == 2u ? top_k : n_comp;
    for (uint32_t u = 0; u < scan_comp; u++) {
        uint32_t c = u;
        if (mode == 2u) {
            if (!topk) continue;
            const int ci = topk[(uint64_t)t * top_k + u];
            if (ci < 0) continue;
            c = (uint32_t)ci;
        }
        if (c >= n_comp || c >= visible) continue;
        if (mode == 1u && comp_mask && comp_mask[(uint64_t)t * n_comp + c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (mode == 1u && comp_mask) s += comp_mask[(uint64_t)t * n_comp + c];
        if (s > max_score) max_score = s;
    }

    float denom = expf(sinks[h] - max_score);
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t kpos = first_raw_pos + r;
        if (kpos > qpos) continue;
        if (window != 0 && qpos - kpos >= window) continue;
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        const float w = expf(s * scale - max_score);
        denom += w;
        acc += w * kv[d];
    }
    for (uint32_t u = 0; u < scan_comp; u++) {
        uint32_t c = u;
        if (mode == 2u) {
            if (!topk) continue;
            const int ci = topk[(uint64_t)t * top_k + u];
            if (ci < 0) continue;
            c = (uint32_t)ci;
        }
        if (c >= n_comp || c >= visible) continue;
        if (mode == 1u && comp_mask && comp_mask[(uint64_t)t * n_comp + c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (mode == 1u && comp_mask) s += comp_mask[(uint64_t)t * n_comp + c];
        const float w = expf(s - max_score);
        denom += w;
        acc += w * kv[d];
    }
    heads[idx] = acc / denom;
}

__global__ static void ds4_hip_attention_prefill_static_mixed_kernel(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const float *sinks,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim, bool use_comp_mask) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_head * head_dim;
    if (idx >= total) return;
    const uint32_t d = (uint32_t)(idx % head_dim);
    const uint64_t th = idx / head_dim;
    const uint32_t h = (uint32_t)(th % n_head);
    const uint32_t t = (uint32_t)(th / n_head);
    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    float max_score = sinks[h];

    for (uint32_t k = 0; k < n_tokens; k++) {
        if (k > t) continue;
        if (window != 0 && t - k >= window) continue;
        const float *kv = raw_kv + (uint64_t)k * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }
    const uint32_t comp_visible = ratio ? (t + 1u) / ratio : 0u;
    for (uint32_t c = 0; c < n_comp && c < comp_visible; c++) {
        if (use_comp_mask && comp_mask && comp_mask[(uint64_t)t * n_comp + c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (use_comp_mask && comp_mask) s += comp_mask[(uint64_t)t * n_comp + c];
        if (s > max_score) max_score = s;
    }

    float denom = expf(sinks[h] - max_score);
    float acc = 0.0f;
    for (uint32_t k = 0; k < n_tokens; k++) {
        if (k > t) continue;
        if (window != 0 && t - k >= window) continue;
        const float *kv = raw_kv + (uint64_t)k * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        const float w = expf(s * scale - max_score);
        denom += w;
        acc += w * kv[d];
    }
    for (uint32_t c = 0; c < n_comp && c < comp_visible; c++) {
        if (use_comp_mask && comp_mask && comp_mask[(uint64_t)t * n_comp + c] <= -5.0e29f) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (use_comp_mask && comp_mask) s += comp_mask[(uint64_t)t * n_comp + c];
        const float w = expf(s - max_score);
        denom += w;
        acc += w * kv[d];
    }
    heads[idx] = acc / denom;
}

__global__ static void ds4_hip_attention_prefill_static_mixed_warprows_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ comp_mask,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        bool use_comp_mask) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    if (t >= n_tokens || h >= n_head || ratio == 0 || rows_per_block == 0) return;

    extern __shared__ float sh[];
    float *scores = sh;
    const uint32_t score_cap = (window ? window : n_tokens) + n_comp;
    float *qsh = scores + score_cap;
    const uint32_t raw_first = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    const uint32_t raw_count = t + 1u - raw_first;
    uint32_t comp_visible = (t + 1u) / ratio;
    if (comp_visible > n_comp) comp_visible = n_comp;
    const uint32_t n_scores = raw_count + comp_visible;
    if (n_scores == 0 || n_scores > score_cap) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) qsh[i] = qh[i];
    __syncthreads();

    for (uint32_t base = 0; base < n_scores; base += rows_per_block) {
        const uint32_t row = base + wave;
        if (row < n_scores) {
            const bool is_comp = row >= raw_count;
            const uint32_t k = is_comp ? (row - raw_count) : (raw_first + row);
            float mask = 0.0f;
            bool skip = false;
            if (is_comp && use_comp_mask && comp_mask) {
                mask = comp_mask[(uint64_t)t * n_comp + k];
                skip = mask <= -5.0e29f;
            }
            float acc = 0.0f;
            if (!skip) {
                const float *kv = is_comp ? (comp_kv + (uint64_t)k * head_dim)
                                          : (raw_kv + (uint64_t)k * head_dim);
                for (uint32_t i = lane; i < head_dim; i += (uint32_t)warpSize) acc += qsh[i] * kv[i];
            }
            acc = ds4_hip_warp_reduce_sum(acc);
            if (lane == 0) scores[row] = skip ? -1.0e30f : (acc * scale + mask);
        }
        __syncthreads();
    }

    float lmax = tid == 0 ? sinks[h] : -3.4e38f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) lmax = fmaxf(lmax, scores[i]);
    const float max_score = ds4_hip_block_reduce_max(lmax);

    float lsum = tid == 0 ? expf(sinks[h] - max_score) : 0.0f;
    for (uint32_t i = tid; i < n_scores; i += blockDim.x) {
        const float w = expf(scores[i] - max_score);
        scores[i] = w;
        lsum += w;
    }
    const float denom = ds4_hip_block_reduce_sum(lsum);

    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t k = raw_first + r;
            acc += scores[r] * raw_kv[(uint64_t)k * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_visible; c++) {
            acc += scores[raw_count + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_attention_prefill_static_mixed_scores_kernel(
        float *__restrict__ heads,
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ comp_mask,
        const float *__restrict__ sinks,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        bool use_comp_mask) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head || ratio == 0) return;

    extern __shared__ float sh[];
    float *scores = sh;
    const uint32_t score_cap = window + n_comp;
    float *red = scores + score_cap;
    float *meta = red + blockDim.x;

    const uint32_t raw_first = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    const uint32_t raw_count = t + 1u - raw_first;
    uint32_t comp_visible = (t + 1u) / ratio;
    if (comp_visible > n_comp) comp_visible = n_comp;
    if (raw_count + comp_visible > score_cap) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    float local_max = sinks[h];

    for (uint32_t r = 0; r < raw_count; r++) {
        const uint32_t k = raw_first + r;
        const float *kv = raw_kv + (uint64_t)k * head_dim;
        float acc = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += blockDim.x) acc += qh[i] * kv[i];
        red[tid] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float s = red[0] * scale;
            scores[r] = s;
            if (s > local_max) local_max = s;
        }
        __syncthreads();
    }

    for (uint32_t c = 0; c < comp_visible; c++) {
        float mask = 0.0f;
        if (use_comp_mask && comp_mask) mask = comp_mask[(uint64_t)t * n_comp + c];
        if (use_comp_mask && mask <= -5.0e29f) {
            if (tid == 0) scores[raw_count + c] = -1.0e30f;
            __syncthreads();
            continue;
        }
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float acc = 0.0f;
        for (uint32_t i = tid; i < head_dim; i += blockDim.x) acc += qh[i] * kv[i];
        red[tid] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float s = red[0] * scale + mask;
            scores[raw_count + c] = s;
            if (s > local_max) local_max = s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = expf(sinks[h] - local_max);
        const uint32_t n_scores = raw_count + comp_visible;
        for (uint32_t i = 0; i < n_scores; i++) {
            const float w = expf(scores[i] - local_max);
            scores[i] = w;
            denom += w;
        }
        meta[0] = denom;
    }
    __syncthreads();

    const float denom = meta[0];
    float *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t k = raw_first + r;
            acc += scores[r] * raw_kv[(uint64_t)k * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_visible; c++) {
            acc += scores[raw_count + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        out[d] = acc / denom;
    }
}

__global__ static void ds4_hip_q8_grouped_kernel(float *low, const unsigned char *w, const float *heads,
                                                 uint32_t n_tokens, uint32_t n_groups, uint64_t group_dim,
                                                 uint64_t rank, uint64_t row_bytes) {
    const uint64_t idx = (uint64_t)blockIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_groups * rank;
    if (idx >= total) return;
    const uint64_t row = idx % rank;
    const uint64_t gtmp = idx / rank;
    const uint32_t g = (uint32_t)(gtmp % n_groups);
    const uint32_t t = (uint32_t)(gtmp / n_groups);
    const uint64_t tensor_row = (uint64_t)g * rank + row;
    const unsigned char *wr = w + tensor_row * row_bytes;
    const float *x = heads + ((uint64_t)t * n_groups + g) * group_dim;
    float acc = 0.0f;
    for (uint64_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
        const uint64_t b = i >> 5;
        const uint64_t lane = i & 31u;
        const unsigned char *blk = wr + b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)qv * x[i];
    }
    if (blockDim.x == warpSize) {
        acc = ds4_hip_warp_reduce_sum(acc);
        if (threadIdx.x == 0) low[idx] = acc;
    } else {
        ds4_hip_block_reduce_store(low, acc, idx);
    }
}

__global__ static void ds4_hip_q8_grouped_sharedx_rows_kernel(float *low, const unsigned char *w, const float *heads,
                                                              uint32_t n_tokens, uint32_t n_groups, uint64_t group_dim,
                                                              uint64_t rank, uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    const uint64_t total = (uint64_t)n_tokens * n_groups * rank;
    const uint64_t base_idx = (uint64_t)blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint64_t base_gtmp = base_idx / rank;
    const uint32_t g = (uint32_t)(base_gtmp % n_groups);
    const uint32_t t = (uint32_t)(base_gtmp / n_groups);
    const float *x = heads + ((uint64_t)t * n_groups + g) * group_dim;
    for (uint64_t i = tid; i < group_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint64_t row = idx % rank;
    const uint64_t tensor_row = (uint64_t)g * rank + row;
    const unsigned char *wr = w + tensor_row * row_bytes;
    float acc = 0.0f;
    for (uint64_t i = lane; i < group_dim; i += (uint32_t)warpSize) {
        const uint64_t b = i >> 5;
        const uint64_t qlane = i & 31u;
        const unsigned char *blk = wr + b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[qlane];
        acc += d * (float)qv * shx[i];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) low[idx] = acc;
}

__global__ static void ds4_hip_q8_grouped_sharedx_rows_w32_kernel(float *__restrict__ low, const unsigned char *__restrict__ w, const float *__restrict__ heads,
                                                                  uint32_t n_tokens, uint32_t n_groups, uint32_t n_blocks,
                                                                  uint64_t rank, uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t group_dim = n_blocks << 5;
    const uint64_t total = (uint64_t)n_tokens * n_groups * rank;
    const uint64_t base_idx = (uint64_t)blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint64_t base_gtmp = base_idx / rank;
    const uint32_t g = (uint32_t)(base_gtmp % n_groups);
    const uint32_t t = (uint32_t)(base_gtmp / n_groups);
    const float *x = heads + ((uint64_t)t * n_groups + g) * group_dim;
    for (uint32_t i = tid; i < group_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint64_t row = idx % rank;
    const uint64_t tensor_row = (uint64_t)g * rank + row;
    const unsigned char *wr = w + tensor_row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)qv * shx[(b << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) low[idx] = acc;
}

template <uint32_t TOK_TILE>
__global__ static void ds4_hip_q8_grouped_sharedx_rows_w32_toktile_kernel(float *__restrict__ low,
                                                                          const unsigned char *__restrict__ w,
                                                                          const float *__restrict__ heads,
                                                                          uint32_t n_tokens,
                                                                          uint32_t n_groups,
                                                                          uint32_t n_blocks,
                                                                          uint32_t rank,
                                                                          uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const uint32_t g = blockIdx.x / row_blocks;
    const uint32_t row0 = (blockIdx.x - g * row_blocks) * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    const uint32_t group_dim = n_blocks << 5;
    if (g >= n_groups || t0 >= n_tokens) return;

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        const uint32_t t = t0 + u;
        float *dst = shx + (uint64_t)u * group_dim;
        if (t < n_tokens) {
            const float *src = heads + ((uint64_t)t * n_groups + g) * group_dim;
            for (uint32_t i = tid; i < group_dim; i += blockDim.x) dst[i] = src[i];
        } else {
            for (uint32_t i = tid; i < group_dim; i += blockDim.x) dst[i] = 0.0f;
        }
    }
    __syncthreads();

    if (row0 >= rank) return;
    const unsigned char *wr = w + ((uint64_t)g * rank + row0) * row_bytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[lane];
        const float wv = d * (float)qv;
        const uint32_t xoff = (b << 5) + lane;
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] += wv * shx[(uint64_t)u * group_dim + xoff];
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = ds4_hip_warp_reduce_sum(acc[u]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tokens) low[((uint64_t)t * n_groups + g) * rank + row0] = acc[u];
        }
    }
}

__global__ static void ds4_hip_q8_grouped_partial16_w32_kernel(float *__restrict__ partial, const unsigned char *__restrict__ w, const float *__restrict__ heads,
                                                               uint32_t n_groups, uint32_t rank, uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t total = n_groups * rank;
    const uint32_t base_idx = blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint32_t g = (base_idx / rank) % n_groups;
    const uint32_t b0 = split << 4;
    const float *x = heads + (uint64_t)g * 4096u;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint32_t row = idx % rank;
    const unsigned char *wr = w + (uint64_t)((uint64_t)g * rank + row) * row_bytes;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * total + idx] = acc;
}

__global__ static void ds4_hip_q8_grouped_repack_partial16_w32_kernel(float *__restrict__ partial,
                                                                      const int8_t *__restrict__ q,
                                                                      const uint16_t *__restrict__ scales,
                                                                      const float *__restrict__ heads,
                                                                      uint32_t n_groups,
                                                                      uint32_t rank) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t total = n_groups * rank;
    const uint32_t base_idx = blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint32_t g = (base_idx / rank) % n_groups;
    const uint32_t b0 = split << 4;
    const float *x = heads + (uint64_t)g * 4096u;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint32_t row = idx % rank;
    const uint64_t tensor_row = (uint64_t)g * rank + row;
    const int8_t *qrow = q + tensor_row * 4096u;
    const uint16_t *srow = scales + tensor_row * 128u;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(srow, b);
        const int8_t qv = qrow[((uint64_t)b << 5) + lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * total + idx] = acc;
}

__global__ static void ds4_hip_q8_grouped_split16_partial_w32_kernel(float *__restrict__ partial,
                                                                    const unsigned char *__restrict__ pack,
                                                                    const float *__restrict__ heads,
                                                                    uint32_t n_groups,
                                                                    uint32_t rank) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t total = n_groups * rank;
    const uint32_t base_idx = blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint32_t g = (base_idx / rank) % n_groups;
    const uint32_t b0 = split << 4;
    const float *x = heads + (uint64_t)g * 4096u;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t idx = base_idx + wave;
    if (idx >= total) return;
    const unsigned char *rec = pack + ((uint64_t)split * total + idx) * 544ull;
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(rec);
    const int8_t *q = reinterpret_cast<const int8_t *>(rec + 32u);
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(sc, bb);
        const int8_t qv = q[(bb << 5) + lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * total + idx] = acc;
}

__global__ static void ds4_hip_q8_split16_partial_w32_kernel(float *__restrict__ partial,
                                                            const unsigned char *__restrict__ pack,
                                                            const float *__restrict__ x,
                                                            uint32_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t b0 = split << 4;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *rec = pack + ((uint64_t)split * out_dim + row) * 544ull;
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(rec);
    const int8_t *q = reinterpret_cast<const int8_t *>(rec + 32u);
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const float d = ds4_hip_q8_repack_scale_broadcast_w32(sc, bb);
        const int8_t qv = q[(bb << 5) + lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void ds4_hip_q8_grouped_partial_w32_kernel(float *__restrict__ partial, const unsigned char *__restrict__ w, const float *__restrict__ heads,
                                                             uint32_t n_groups, uint32_t n_blocks, uint32_t rank,
                                                             uint64_t row_bytes, uint32_t n_splits) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t total = n_groups * rank;
    const uint32_t base_idx = blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint32_t g = (base_idx / rank) % n_groups;
    const uint32_t chunk = (n_blocks + n_splits - 1u) / n_splits;
    const uint32_t b0 = split * chunk;
    const uint32_t b1 = min(n_blocks, b0 + chunk);
    const uint32_t chunk_blocks = b1 > b0 ? b1 - b0 : 0;
    const float *x = heads + (uint64_t)g * ((uint64_t)n_blocks << 5);
    for (uint32_t i = tid; i < (chunk_blocks << 5); i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint32_t row = idx % rank;
    const unsigned char *wr = w + (uint64_t)((uint64_t)g * rank + row) * row_bytes;
    float acc = 0.0f;
    for (uint32_t bb = 0; bb < chunk_blocks; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = ds4_hip_q8_0_scale_broadcast_w32(blk);
        const int8_t qv = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)qv * shx[(bb << 5) + lane];
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) partial[(uint64_t)split * total + idx] = acc;
}

__global__ static void ds4_hip_q8_grouped_partial_sum_kernel(float *__restrict__ low, const float *__restrict__ partial,
                                                             uint32_t total, uint32_t n_splits) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) acc += partial[(uint64_t)s * total + idx];
    low[idx] = acc;
}

__global__ static void ds4_hip_q8_grouped_sharedx_rows_w32_2row_kernel(float *__restrict__ low, const unsigned char *__restrict__ w, const float *__restrict__ heads,
                                                                       uint32_t n_tokens, uint32_t n_groups, uint32_t n_blocks,
                                                                       uint64_t rank, uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = (blockDim.x >> 5) << 1;
    const uint32_t group_dim = n_blocks << 5;
    const uint64_t total = (uint64_t)n_tokens * n_groups * rank;
    const uint64_t base_idx = (uint64_t)blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint64_t base_gtmp = base_idx / rank;
    const uint32_t g = (uint32_t)(base_gtmp % n_groups);
    const uint32_t t = (uint32_t)(base_gtmp / n_groups);
    const float *x = heads + ((uint64_t)t * n_groups + g) * group_dim;
    for (uint32_t i = tid; i < group_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t idx0 = base_idx + ((uint64_t)wave << 1);
    if (idx0 >= total) return;
    const uint64_t row0 = idx0 % rank;
    const uint64_t idx1 = idx0 + 1u;
    const uint64_t tensor_row0 = (uint64_t)g * rank + row0;
    const unsigned char *wr0 = w + tensor_row0 * row_bytes;
    const unsigned char *wr1 = wr0 + row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float xv = shx[(b << 5) + lane];
        const unsigned char *blk0 = wr0 + (uint64_t)b * 34u;
        const float d0 = ds4_hip_q8_0_scale_broadcast_w32(blk0);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        acc0 += d0 * (float)q0 * xv;
        if (row0 + 1u < rank && idx1 < total) {
            const unsigned char *blk1 = wr1 + (uint64_t)b * 34u;
            const float d1 = ds4_hip_q8_0_scale_broadcast_w32(blk1);
            const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
            acc1 += d1 * (float)q1 * xv;
        }
    }
    acc0 = ds4_hip_warp_reduce_sum(acc0);
    acc1 = ds4_hip_warp_reduce_sum(acc1);
    if (lane == 0) {
        low[idx0] = acc0;
        if (row0 + 1u < rank && idx1 < total) low[idx1] = acc1;
    }
}

__device__ static float ds4_hip_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ static inline void ds4_hip_q2_k_scale_broadcast(const unsigned char *blk, float *d, float *dmin) {
    const uint32_t lane = (uint32_t)threadIdx.x & ((uint32_t)warpSize - 1u);
    float vd = 0.0f;
    float vm = 0.0f;
    if (lane == 0) {
        const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
        const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
        vd = ds4_hip_f16_to_f32(d_bits);
        vm = ds4_hip_f16_to_f32(dmin_bits);
    }
    *d = __shfl(vd, 0, warpSize);
    *dmin = __shfl(vm, 0, warpSize);
}

__device__ static inline void ds4_hip_q2_k_scale_broadcast_w32(const unsigned char *blk, float *d, float *dmin) {
    float vd = 0.0f;
    float vm = 0.0f;
    if ((threadIdx.x & 31u) == 0u) {
        const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
        const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
        vd = ds4_hip_f16_to_f32(d_bits);
        vm = ds4_hip_f16_to_f32(dmin_bits);
    }
    *d = __shfl(vd, 0, 32);
    *dmin = __shfl(vm, 0, 32);
}

__device__ static inline float ds4_hip_q2_k_dequant_256_scaled(const unsigned char *blk, uint32_t i,
                                                               float d, float dmin) {
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = i >> 4;
    const uint32_t within = g & 7u;
    const uint32_t chunk = g >> 3;
    const uint32_t shift = (within >> 1) * 2u;
    const uint32_t half = within & 1u;
    const uint32_t lane = i & 15u;
    const uint32_t qi = chunk * 32u + half * 16u + lane;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4);
    return d * scale * q - dmin * mn;
}

__device__ static inline float ds4_hip_q2_k_dequant_256_scaled_w32(const unsigned char *blk, uint32_t lane,
                                                                   uint32_t kk, float d, float dmin) {
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = (lane >> 4) + (kk << 1);
    const uint32_t within = g & 7u;
    const uint32_t qi = (g >> 3) * 32u + (within & 1u) * 16u + (lane & 15u);
    const uint32_t shift = (within >> 1) * 2u;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4);
    return d * scale * q - dmin * mn;
}

__global__ static void ds4_hip_moe_q2_gate_up_kernel(float *__restrict__ gate, float *__restrict__ up, float *__restrict__ mid,
                                                     const unsigned char *__restrict__ gate_w, const unsigned char *__restrict__ up_w,
                                                     const float *__restrict__ x, const int *__restrict__ selected, const float *__restrict__ weights,
                                                     uint32_t n_tokens, uint32_t in_dim, uint32_t mid_dim,
                                                     uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                                                     uint64_t up_expert_bytes, uint64_t up_row_bytes,
                                                     float clamp, bool store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + wave;
    const uint32_t ts = (uint32_t)blockIdx.y;
    if (row >= mid_dim || ts >= n_tokens * 6u) return;
    const uint32_t slot = ts % 6u;
    const uint32_t t = ts / 6u;
    int expert = selected[(uint64_t)t * 6u + slot];
    if (expert < 0) expert = 0;
    const float *xt = x + (uint64_t)t * in_dim;
    const unsigned char *grow = gate_w + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *urow = up_w + (uint64_t)expert * up_expert_bytes + (uint64_t)row * up_row_bytes;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint32_t nb = in_dim >> 8;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gblk = grow + (uint64_t)b * 84u;
        const unsigned char *ublk = urow + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        if (warpSize == 32) {
            ds4_hip_q2_k_scale_broadcast_w32(gblk, &gd, &gdmin);
            ds4_hip_q2_k_scale_broadcast_w32(ublk, &ud, &udmin);
        } else {
            ds4_hip_q2_k_scale_broadcast(gblk, &gd, &gdmin);
            ds4_hip_q2_k_scale_broadcast(ublk, &ud, &udmin);
        }
        const uint64_t xbase = (uint64_t)b * 256u;
        if (warpSize == 32) {
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5);
                const float xv = xt[xbase + i];
                g_acc += ds4_hip_q2_k_dequant_256_scaled_w32(gblk, lane, kk, gd, gdmin) * xv;
                u_acc += ds4_hip_q2_k_dequant_256_scaled_w32(ublk, lane, kk, ud, udmin) * xv;
            }
        } else {
            for (uint32_t i = lane; i < 256u; i += (uint32_t)warpSize) {
                const float xv = xt[xbase + i];
                g_acc += ds4_hip_q2_k_dequant_256_scaled(gblk, i, gd, gdmin) * xv;
                u_acc += ds4_hip_q2_k_dequant_256_scaled(ublk, i, ud, udmin) * xv;
            }
        }
    }

    g_acc = ds4_hip_warp_reduce_sum(g_acc);
    u_acc = ds4_hip_warp_reduce_sum(u_acc);
    if (lane == 0) {
        float g = g_acc;
        float u = u_acc;
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        const uint64_t idx = ((uint64_t)t * 6u + slot) * mid_dim + row;
        const float m = ds4_hip_silu(g) * u * weights[(uint64_t)t * 6u + slot];
        if (store_gate_up) {
            gate[idx] = g;
            up[idx] = u;
        }
        mid[idx] = m;
    }
}

__global__ static void ds4_hip_moe_q2_down_kernel(float *__restrict__ out, const unsigned char *__restrict__ down_w,
                                                  const float *__restrict__ mid, const int *__restrict__ selected,
                                                  uint32_t n_tokens, uint32_t mid_dim, uint32_t out_dim,
                                                  uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & ((uint32_t)warpSize - 1u);
    const uint32_t wave = tid / (uint32_t)warpSize;
    const uint32_t rows_per_block = blockDim.x / (uint32_t)warpSize;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + wave;
    const uint32_t t = (uint32_t)blockIdx.y;
    if (row >= out_dim || t >= n_tokens) return;
    float acc = 0.0f;
    const uint32_t nb = mid_dim >> 8;
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int expert = selected[(uint64_t)t * 6u + slot];
        if (expert < 0) expert = 0;
        const unsigned char *drow = down_w + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes;
        const float *m = mid + ((uint64_t)t * 6u + slot) * mid_dim;
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *dblk = drow + (uint64_t)b * 84u;
            float d, dmin;
            if (warpSize == 32) ds4_hip_q2_k_scale_broadcast_w32(dblk, &d, &dmin);
            else ds4_hip_q2_k_scale_broadcast(dblk, &d, &dmin);
            const uint64_t mbase = (uint64_t)b * 256u;
            if (warpSize == 32) {
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5);
                    acc += ds4_hip_q2_k_dequant_256_scaled_w32(dblk, lane, kk, d, dmin) * m[mbase + i];
                }
            } else {
                for (uint32_t i = lane; i < 256u; i += (uint32_t)warpSize) {
                    acc += ds4_hip_q2_k_dequant_256_scaled(dblk, i, d, dmin) * m[mbase + i];
                }
            }
        }
    }
    acc = ds4_hip_warp_reduce_sum(acc);
    if (lane == 0) out[(uint64_t)t * out_dim + row] = acc;
}

__global__ static void ds4_hip_moe_bucket_pairs_kernel(int *__restrict__ counts,
                                                       int *__restrict__ buckets,
                                                       const int *__restrict__ selected,
                                                       uint32_t n_tokens,
                                                       uint32_t stride) {
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_pairs = n_tokens * 6u;
    if (pair >= n_pairs) return;
    int expert = selected[pair];
    if (expert < 0 || expert >= 256) expert = 0;
    const uint32_t idx = (uint32_t)atomicAdd(&counts[expert], 1);
    if (idx < stride) buckets[(uint64_t)(uint32_t)expert * stride + idx] = (int)pair;
}

template <uint32_t PAIR_TILE>
__global__ static void ds4_hip_moe_q2_gate_up_expert_batch_kernel(float *__restrict__ mid,
                                                                  const unsigned char *__restrict__ gate_w,
                                                                  const unsigned char *__restrict__ up_w,
                                                                  const float *__restrict__ x,
                                                                  const float *__restrict__ weights,
                                                                  const int *__restrict__ counts,
                                                                  const int *__restrict__ buckets,
                                                                  uint32_t stride,
                                                                  uint32_t min_count,
                                                                  uint32_t max_count,
                                                                  uint32_t in_dim,
                                                                  uint32_t mid_dim,
                                                                  uint64_t gate_expert_bytes,
                                                                  uint64_t gate_row_bytes,
                                                                  uint64_t up_expert_bytes,
                                                                  uint64_t up_row_bytes,
                                                                  float clamp) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    if (row >= mid_dim || expert >= 256u) return;
    const uint32_t count = (uint32_t)counts[expert];
    if (count == 0 || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char *grow = gate_w + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *urow = up_w + (uint64_t)expert * up_expert_bytes + (uint64_t)row * up_row_bytes;
    const uint32_t nb = in_dim >> 8;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *gblk = grow + (uint64_t)b * 84u;
            const unsigned char *ublk = urow + (uint64_t)b * 84u;
            float gd, gdmin, ud, udmin;
            ds4_hip_q2_k_scale_broadcast_w32(gblk, &gd, &gdmin);
            ds4_hip_q2_k_scale_broadcast_w32(ublk, &ud, &udmin);
            const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5);
                const float gwv = ds4_hip_q2_k_dequant_256_scaled_w32(gblk, lane, kk, gd, gdmin);
                const float uwv = ds4_hip_q2_k_dequant_256_scaled_w32(ublk, lane, kk, ud, udmin);
#pragma unroll
                for (uint32_t u = 0; u < PAIR_TILE; u++) {
                    if (pair[u] >= 0) {
                        const uint32_t t = (uint32_t)pair[u] / 6u;
                        const float xv = x[(uint64_t)t * in_dim + xbase + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            g_acc[u] = ds4_hip_warp_reduce_sum(g_acc[u]);
            u_acc[u] = ds4_hip_warp_reduce_sum(u_acc[u]);
        }
        if (lane == 0) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] >= 0) {
                    float g = g_acc[u];
                    float upv = u_acc[u];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (upv > clamp) upv = clamp;
                        if (upv < -clamp) upv = -clamp;
                    }
                    const float m = ds4_hip_silu(g) * upv * weights[(uint32_t)pair[u]];
                    mid[(uint64_t)(uint32_t)pair[u] * mid_dim + row] = m;
                }
            }
        }
    }
}

template <uint32_t PAIR_TILE>
__global__ static void ds4_hip_moe_q2_gate_up_expert_batch_sharedx_kernel(float *__restrict__ mid,
                                                                          const unsigned char *__restrict__ gate_w,
                                                                          const unsigned char *__restrict__ up_w,
                                                                          const float *__restrict__ x,
                                                                          const float *__restrict__ weights,
                                                                          const int *__restrict__ counts,
                                                                          const int *__restrict__ buckets,
                                                                          uint32_t stride,
                                                                          uint32_t min_count,
                                                                          uint32_t max_count,
                                                                          uint32_t in_dim,
                                                                          uint32_t mid_dim,
                                                                          uint64_t gate_expert_bytes,
                                                                          uint64_t gate_row_bytes,
                                                                          uint64_t up_expert_bytes,
                                                                          uint64_t up_row_bytes,
                                                                          float clamp) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < mid_dim;
    const uint32_t count = (uint32_t)counts[expert];
    if (count == 0 || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char *grow = gate_w + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const unsigned char *urow = up_w + (uint64_t)expert * up_expert_bytes + (uint64_t)(row_valid ? row : 0u) * up_row_bytes;
    const uint32_t nb = in_dim >> 8;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t xbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8;
                const uint32_t k = j & 255u;
                if (pair[u] >= 0) {
                    const uint32_t t = (uint32_t)pair[u] / 6u;
                    shx[j] = x[(uint64_t)t * in_dim + xbase + k];
                } else {
                    shx[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *gblk = grow + (uint64_t)b * 84u;
                const unsigned char *ublk = urow + (uint64_t)b * 84u;
                float gd, gdmin, ud, udmin;
                ds4_hip_q2_k_scale_broadcast_w32(gblk, &gd, &gdmin);
                ds4_hip_q2_k_scale_broadcast_w32(ublk, &ud, &udmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5);
                    const float gwv = ds4_hip_q2_k_dequant_256_scaled_w32(gblk, lane, kk, gd, gdmin);
                    const float uwv = ds4_hip_q2_k_dequant_256_scaled_w32(ublk, lane, kk, ud, udmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) {
                        const float xv = shx[(u << 8) + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            g_acc[u] = ds4_hip_warp_reduce_sum(g_acc[u]);
            u_acc[u] = ds4_hip_warp_reduce_sum(u_acc[u]);
        }
        if (lane == 0 && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] >= 0) {
                    float g = g_acc[u];
                    float upv = u_acc[u];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (upv > clamp) upv = clamp;
                        if (upv < -clamp) upv = -clamp;
                    }
                    mid[(uint64_t)(uint32_t)pair[u] * mid_dim + row] = ds4_hip_silu(g) * upv * weights[(uint32_t)pair[u]];
                }
            }
        }
    }
}

template <uint32_t PAIR_TILE>
__global__ static void ds4_hip_moe_q2_down_expert_batch_kernel(float *__restrict__ experts,
                                                               const unsigned char *__restrict__ down_w,
                                                               const float *__restrict__ mid,
                                                               const int *__restrict__ counts,
                                                               const int *__restrict__ buckets,
                                                               uint32_t stride,
                                                               uint32_t min_count,
                                                               uint32_t max_count,
                                                               uint32_t mid_dim,
                                                               uint32_t out_dim,
                                                               uint64_t down_expert_bytes,
                                                               uint64_t down_row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    if (row >= out_dim || expert >= 256u) return;
    const uint32_t count = (uint32_t)counts[expert];
    if (count == 0 || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char *drow = down_w + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes;
    const uint32_t nb = mid_dim >> 8;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *dblk = drow + (uint64_t)b * 84u;
            float d, dmin;
            ds4_hip_q2_k_scale_broadcast_w32(dblk, &d, &dmin);
            const uint64_t mbase = (uint64_t)b * 256u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5);
                const float wv = ds4_hip_q2_k_dequant_256_scaled_w32(dblk, lane, kk, d, dmin);
#pragma unroll
                for (uint32_t u = 0; u < PAIR_TILE; u++) {
                    if (pair[u] >= 0) acc[u] += wv * mid[(uint64_t)(uint32_t)pair[u] * mid_dim + mbase + i];
                }
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] = ds4_hip_warp_reduce_sum(acc[u]);
        if (lane == 0) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] >= 0) experts[(uint64_t)(uint32_t)pair[u] * out_dim + row] = acc[u];
            }
        }
    }
}

template <uint32_t PAIR_TILE>
__global__ static void ds4_hip_moe_q2_down_expert_batch_sharedmid_kernel(float *__restrict__ experts,
                                                                         const unsigned char *__restrict__ down_w,
                                                                         const float *__restrict__ mid,
                                                                         const int *__restrict__ counts,
                                                                         const int *__restrict__ buckets,
                                                                         uint32_t stride,
                                                                         uint32_t min_count,
                                                                         uint32_t max_count,
                                                                         uint32_t mid_dim,
                                                                         uint32_t out_dim,
                                                                         uint64_t down_expert_bytes,
                                                                         uint64_t down_row_bytes) {
    extern __shared__ float shmid[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < out_dim;
    const uint32_t count = (uint32_t)counts[expert];
    if (count == 0 || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char *drow = down_w + (uint64_t)expert * down_expert_bytes + (uint64_t)(row_valid ? row : 0u) * down_row_bytes;
    const uint32_t nb = mid_dim >> 8;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t mbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8;
                const uint32_t k = j & 255u;
                shmid[j] = (pair[u] >= 0) ? mid[(uint64_t)(uint32_t)pair[u] * mid_dim + mbase + k] : 0.0f;
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *dblk = drow + (uint64_t)b * 84u;
                float d, dmin;
                ds4_hip_q2_k_scale_broadcast_w32(dblk, &d, &dmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5);
                    const float wv = ds4_hip_q2_k_dequant_256_scaled_w32(dblk, lane, kk, d, dmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] += wv * shmid[(u << 8) + i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] = ds4_hip_warp_reduce_sum(acc[u]);
        if (lane == 0 && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] >= 0) experts[(uint64_t)(uint32_t)pair[u] * out_dim + row] = acc[u];
            }
        }
    }
}

__global__ static void ds4_hip_moe_experts_reduce_kernel(float *__restrict__ out,
                                                         const float *__restrict__ experts,
                                                         uint32_t n_tokens,
                                                         uint32_t out_dim) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * out_dim;
    if (idx >= total) return;
    const uint32_t t = idx / out_dim;
    const uint32_t row = idx - t * out_dim;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) acc += experts[((uint64_t)t * 6u + slot) * out_dim + row];
    out[idx] = acc;
}

__global__ static void ds4_hip_router_select_kernel(int *selected, float *weights, float *probs,
                                                    const float *logits, const int *tokens,
                                                    const float *bias, const int *hash,
                                                    uint32_t hash_rows, uint32_t token_single,
                                                    uint32_t n_tokens, bool has_bias, bool hash_mode) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    float local_probs[256];
    for (uint32_t i = 0; i < 256u; i++) {
        const float p = sqrtf(ds4_hip_softplus(logits[(uint64_t)t * 256u + i]));
        local_probs[i] = p;
        if (probs) probs[(uint64_t)t * 256u + i] = p;
    }
    int sel[6];
    if (hash_mode) {
        int tok = tokens ? tokens[t] : (int)token_single;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int *row = hash + (uint64_t)(uint32_t)tok * 6u;
        for (uint32_t k = 0; k < 6u; k++) sel[k] = row[k];
    } else {
        float best[6];
        for (uint32_t k = 0; k < 6u; k++) { sel[k] = -1; best[k] = -3.4e38f; }
        for (uint32_t i = 0; i < 256u; i++) {
            float s = local_probs[i] + (has_bias ? bias[i] : 0.0f);
            for (uint32_t k = 0; k < 6u; k++) {
                if (s > best[k]) {
                    for (int m = 5; m > (int)k; m--) { best[m] = best[m - 1]; sel[m] = sel[m - 1]; }
                    best[k] = s;
                    sel[k] = (int)i;
                    break;
                }
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < 6u; k++) {
        int e = sel[k];
        if (e < 0 || e >= 256) e = 0;
        selected[(uint64_t)t * 6u + k] = e;
        weights[(uint64_t)t * 6u + k] = local_probs[e];
        sum += local_probs[e];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t k = 0; k < 6u; k++) weights[(uint64_t)t * 6u + k] = weights[(uint64_t)t * 6u + k] / sum * 1.5f;
}

__global__ static void ds4_hip_router_select_parallel_kernel(int *selected, float *weights, float *probs,
                                                             const float *logits, const int *tokens,
                                                             const float *bias, const int *hash,
                                                             uint32_t hash_rows, uint32_t token_single,
                                                             uint32_t n_tokens, bool has_bias, bool hash_mode) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    __shared__ float sh_probs[256];
    __shared__ float sh_scores[256];
    const float p = sqrtf(ds4_hip_softplus(logits[(uint64_t)t * 256u + i]));
    sh_probs[i] = p;
    sh_scores[i] = p + (has_bias ? bias[i] : 0.0f);
    if (probs) probs[(uint64_t)t * 256u + i] = p;
    __syncthreads();
    if (i == 0) {
        int sel[6];
        if (hash_mode) {
            int tok = tokens ? tokens[t] : (int)token_single;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int *row = hash + (uint64_t)(uint32_t)tok * 6u;
            for (uint32_t k = 0; k < 6u; k++) sel[k] = row[k];
        } else {
            float best[6];
            for (uint32_t k = 0; k < 6u; k++) { sel[k] = -1; best[k] = -3.4e38f; }
            for (uint32_t j = 0; j < 256u; j++) {
                const float s = sh_scores[j];
                for (uint32_t k = 0; k < 6u; k++) {
                    if (s > best[k]) {
                        for (int m = 5; m > (int)k; m--) { best[m] = best[m - 1]; sel[m] = sel[m - 1]; }
                        best[k] = s;
                        sel[k] = (int)j;
                        break;
                    }
                }
            }
        }
        float sum = 0.0f;
        for (uint32_t k = 0; k < 6u; k++) {
            int e = sel[k];
            if (e < 0 || e >= 256) e = 0;
            selected[(uint64_t)t * 6u + k] = e;
            weights[(uint64_t)t * 6u + k] = sh_probs[e];
            sum += sh_probs[e];
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint32_t k = 0; k < 6u; k++) weights[(uint64_t)t * 6u + k] = weights[(uint64_t)t * 6u + k] / sum * 1.5f;
    }
}

static bool ds4_hip_launch_ok(const char *what) {
    hipError_t e = hipGetLastError();
    return ds4_hip_check(e, what ? what : "kernel launch");
}

static bool ds4_hip_profile_begin(const char *env, hipEvent_t *start, hipEvent_t *stop) {
    if (!env || std::getenv(env) == nullptr) return false;
    if (hipEventCreate(start) != hipSuccess) return false;
    if (hipEventCreate(stop) != hipSuccess) {
        (void)hipEventDestroy(*start);
        return false;
    }
    (void)hipEventRecord(*start, g_stream);
    return true;
}

static void ds4_hip_profile_end(bool enabled, hipEvent_t start, hipEvent_t stop,
                                const char *label, uint64_t in_dim, uint64_t out_dim,
                                uint64_t n_tok) {
    if (!enabled) return;
    (void)hipEventRecord(stop, g_stream);
    (void)hipEventSynchronize(stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    std::fprintf(stderr, "ds4: HIP profile %s in=%" PRIu64 " out=%" PRIu64 " tokens=%" PRIu64 " %.3f ms\n",
                 label ? label : "kernel", in_dim, out_dim, n_tok, ms);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

static void ds4_hip_moe_maybe_dump_routing_counts(const int *counts_dev,
                                                   uint32_t n_tokens,
                                                   uint32_t n_expert) {
    const char *env = std::getenv("DS4_HIP_MOE_ROUTING_DUMP");
    if (!env || !counts_dev || n_tokens == 0 || n_expert == 0 || n_expert > 256u) return;
    uint32_t limit = 1u;
    if (env[0] != '\0') {
        if (std::strcmp(env, "all") == 0) {
            limit = 0xffffffffu;
        } else {
            const unsigned long v = std::strtoul(env, nullptr, 10);
            if (v > 0ul) limit = (v > 0xfffffffful) ? 0xffffffffu : (uint32_t)v;
        }
    }
    uint32_t min_tokens = 1u;
    if (const char *min_env = std::getenv("DS4_HIP_MOE_ROUTING_DUMP_MIN_TOKENS")) {
        const unsigned long v = std::strtoul(min_env, nullptr, 10);
        if (v > 0ul) min_tokens = (v > 0xfffffffful) ? 0xffffffffu : (uint32_t)v;
    }
    static uint32_t emitted = 0;
    if (emitted >= limit || n_tokens < min_tokens) return;

    int h_counts[256];
    std::memset(h_counts, 0, sizeof(h_counts));
    hipError_t e = hipMemcpyAsync(h_counts, counts_dev, n_expert * sizeof(int), hipMemcpyDeviceToHost, g_stream);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP MoE routing dump copy failed: %s\n", hipGetErrorString(e));
        return;
    }
    e = hipStreamSynchronize(g_stream);
    if (e != hipSuccess) {
        std::fprintf(stderr, "ds4: HIP MoE routing dump sync failed: %s\n", hipGetErrorString(e));
        return;
    }
    const uint32_t dump_idx = ++emitted;

    uint64_t total = 0;
    uint32_t active = 0, max_expert = 0;
    int min_nz = INT_MAX, max_count = 0;
    uint32_t ge8 = 0, ge16 = 0, ge32 = 0, ge64 = 0, ge128 = 0;
    uint64_t work_ge8 = 0, work_ge16 = 0, work_ge32 = 0, work_ge64 = 0, work_ge128 = 0;
    std::vector<int> nz;
    nz.reserve(n_expert);
    for (uint32_t i = 0; i < n_expert; i++) {
        const int c = h_counts[i];
        if (c <= 0) continue;
        total += (uint32_t)c;
        active++;
        nz.push_back(c);
        if (c < min_nz) min_nz = c;
        if (c > max_count) { max_count = c; max_expert = i; }
        if (c >= 8) { ge8++; work_ge8 += (uint32_t)c; }
        if (c >= 16) { ge16++; work_ge16 += (uint32_t)c; }
        if (c >= 32) { ge32++; work_ge32 += (uint32_t)c; }
        if (c >= 64) { ge64++; work_ge64 += (uint32_t)c; }
        if (c >= 128) { ge128++; work_ge128 += (uint32_t)c; }
    }
    std::sort(nz.begin(), nz.end());
    auto pct = [&](uint32_t num, uint32_t den) -> int {
        if (nz.empty()) return 0;
        size_t idx = (nz.size() * (size_t)num + (size_t)den - 1u) / (size_t)den;
        if (idx == 0) idx = 1;
        if (idx > nz.size()) idx = nz.size();
        return nz[idx - 1u];
    };
    auto top_sum = [&](uint32_t k) -> uint64_t {
        uint64_t s = 0;
        uint32_t n = 0;
        for (auto it = nz.rbegin(); it != nz.rend() && n < k; ++it, ++n) s += (uint32_t)*it;
        return s;
    };
    auto work_pct = [&](uint64_t work) -> double {
        return total ? (100.0 * (double)work / (double)total) : 0.0;
    };
    if (min_nz == INT_MAX) min_nz = 0;
    const double mean_all = n_expert ? (double)total / (double)n_expert : 0.0;
    const double mean_active = active ? (double)total / (double)active : 0.0;
    const uint64_t top7 = top_sum(7u);
    const uint64_t top22 = top_sum(22u);
    const uint64_t top54 = top_sum(54u);
    std::fprintf(stderr,
                 "ds4: HIP MoE routing dump #%u tokens=%u assignments=%" PRIu64 " experts=%u "
                 "active=%u empty=%u mean_all=%.2f mean_active=%.2f min_nz=%d "
                 "p50=%d p75=%d p90=%d p95=%d p99=%d max=%d max_expert=%u "
                 "ge8=%u ge16=%u ge32=%u ge64=%u ge128=%u "
                 "work_ge8=%.1f%% work_ge16=%.1f%% work_ge32=%.1f%% work_ge64=%.1f%% work_ge128=%.1f%% "
                 "top7=%.1f%% top22=%.1f%% top54=%.1f%%\n",
                 dump_idx, n_tokens, total, n_expert, active, n_expert - active,
                 mean_all, mean_active, min_nz, pct(50, 100), pct(75, 100), pct(90, 100),
                 pct(95, 100), pct(99, 100), max_count, max_expert,
                 ge8, ge16, ge32, ge64, ge128,
                 work_pct(work_ge8), work_pct(work_ge16), work_pct(work_ge32),
                 work_pct(work_ge64), work_pct(work_ge128),
                 work_pct(top7), work_pct(top22), work_pct(top54));
    if (std::getenv("DS4_HIP_MOE_ROUTING_DUMP_COUNTS") != nullptr) {
        std::fprintf(stderr, "ds4: HIP MoE routing counts #%u", dump_idx);
        for (uint32_t i = 0; i < n_expert; i++) {
            if ((i & 15u) == 0u) std::fprintf(stderr, "\nds4:   e%03u:", i);
            std::fprintf(stderr, " %d", h_counts[i]);
        }
        std::fprintf(stderr, "\n");
    }
}

extern "C" int ds4_metal_embed_token_hc_tensor(
        ds4_metal_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    if (out_hc->bytes < out_bytes) return 0;
    const uint64_t row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "Q8_0 token embedding");
    if (!w) return 0;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    ds4_hip_embed_token_hc_q8_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out_hc->ptr, w, token, n_embd, n_hc, row_bytes);
    return ds4_hip_launch_ok("Q8_0 token embedding launch") ? 1 : 0;
}

extern "C" int ds4_metal_embed_tokens_hc_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int);
    if (out_hc->bytes < out_bytes || tokens->bytes < token_bytes) return 0;
    const uint64_t row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "Q8_0 token embeddings");
    if (!w) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    ds4_hip_embed_tokens_hc_q8_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out_hc->ptr, (const int *)tokens->ptr, w, n_vocab, n_tokens, n_embd, n_hc, row_bytes);
    return ds4_hip_launch_ok("Q8_0 token embeddings launch") ? 1 : 0;
}

extern "C" int ds4_metal_indexer_scores_prefill_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

extern "C" int ds4_metal_indexer_score_one_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return ds4_metal_indexer_scores_prefill_tensor(scores, q, weights, index_comp, n_comp, 1, n_head, head_dim, 4, scale);
}

extern "C" int ds4_metal_indexer_scores_prefill_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    (void)ratio;
    if (!scores || !q || !weights || !index_comp || n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t score_bytes = (uint64_t)n_tokens * n_comp * sizeof(float);
    if (q->bytes < q_bytes || weights->bytes < weight_bytes || index_comp->bytes < comp_bytes || scores->bytes < score_bytes) return 0;
    ds4_hip_indexer_scores_kernel<<<dim3(n_comp, n_tokens), 256, 0, g_stream>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)index_comp->ptr, n_comp, n_tokens, n_head, head_dim, scale);
    return ds4_hip_launch_ok("indexer scores launch") ? 1 : 0;
}

extern "C" int ds4_metal_indexer_scores_decode_batch_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    (void)pos0;
    return ds4_metal_indexer_scores_prefill_tensor(scores, q, weights, index_comp, n_comp, n_tokens, n_head, head_dim, ratio, scale);
}

extern "C" int ds4_metal_indexer_topk_tensor(
        ds4_metal_tensor       *selected,
        const ds4_metal_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;
    if (scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(int)) return 0;
    if (top_k >= n_comp) {
        const uint64_t n = (uint64_t)n_tokens * n_comp;
        ds4_hip_indexer_select_all_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (int *)selected->ptr, n_comp, n_tokens, top_k);
    } else if (top_k == 1u && n_comp >= 1024u) {
        ds4_hip_indexer_top1_parallel_kernel<<<n_tokens, 256, 0, g_stream>>>(
                (int *)selected->ptr, (const float *)scores->ptr, n_comp, n_tokens);
    } else if (n_comp <= 8192u && top_k <= 1024u) {
        const unsigned threads = 256u;
        const size_t shmem = (size_t)n_comp * sizeof(float) + threads * (sizeof(float) + sizeof(int));
        ds4_hip_indexer_topk_iter_parallel_kernel<<<n_tokens, threads, shmem, g_stream>>>(
                (int *)selected->ptr, (const float *)scores->ptr, n_comp, n_tokens, top_k);
    } else {
        ds4_hip_indexer_topk_kernel<<<n_tokens, 1, 0, g_stream>>>((int *)selected->ptr, (const float *)scores->ptr,
                                                                  n_comp, n_tokens, top_k);
    }
    return ds4_hip_launch_ok("indexer top-k launch") ? 1 : 0;
}

extern "C" int ds4_metal_dsv4_topk_mask_tensor(
        ds4_metal_tensor       *mask,
        const ds4_metal_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;
    if (mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int)) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_comp;
    ds4_hip_topk_mask_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)mask->ptr, (const int *)topk->ptr, n_comp, n_tokens, top_k);
    return ds4_hip_launch_ok("top-k mask launch") ? 1 : 0;
}

extern "C" int ds4_metal_matmul_q8_0_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !x || in_dim == 0 || out_dim == 0 || n_tok == 0 || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    const unsigned warp_threads_top = ds4_hip_warp_threads();
    if (warp_threads_top == 32u && n_tok == 1u && (in_dim & 31u) == 0u) {
        const ds4_hip_repacked_q8_tensor *rw = ds4_hip_q8_repack_get(model_map, model_size, weight_offset, in_dim, out_dim, "Q8_0 matmul");
        if (rw) {
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_Q8_MATMUL_PROFILE", &prof_start, &prof_stop);
            unsigned rows_per_block = std::max(1u, 1024u / warp_threads_top);
            if (const char *rpb_env = std::getenv("DS4_HIP_Q8_REPACK_RPB")) {
                const unsigned v = (unsigned)std::strtoul(rpb_env, nullptr, 10);
                if (v >= 1u && v <= 32u) rows_per_block = v;
            }
            const unsigned threads = rows_per_block * 32u;
            const bool use_repack_sharedx =
                    (in_dim <= 2048u && std::getenv("DS4_HIP_Q8_REPACK_NO_SHAREDX") == nullptr) ||
                    (in_dim <= 8192u && std::getenv("DS4_HIP_Q8_REPACK_SHAREDX") != nullptr);
            if (use_repack_sharedx) {
                ds4_hip_matmul_q8_repack_sharedx_rows_w32_kernel<<<
                        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                        threads, (size_t)(in_dim * sizeof(float)), g_stream>>>(
                        (float *)out->ptr, rw->q, rw->scales, (const float *)x->ptr,
                        (uint32_t)(in_dim >> 5), out_dim);
            } else {
                ds4_hip_matmul_q8_repack_warp_rows_w32_kernel<<<
                        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                        threads, 0, g_stream>>>(
                        (float *)out->ptr, rw->q, rw->scales, (const float *)x->ptr,
                        (uint32_t)(in_dim >> 5), out_dim);
            }
            const bool ok = ds4_hip_launch_ok("Q8_0 repacked matmul launch");
            ds4_hip_profile_end(prof, prof_start, prof_stop, "q8_matmul_repack", in_dim, out_dim, n_tok);
            return ok ? 1 : 0;
        }
    }
    uint64_t q8_wmma_min_tokens = 64u;
    if (const char *min_tok_env = std::getenv("DS4_HIP_Q8_WMMA_MIN_TOKENS")) {
        const uint64_t v = std::strtoull(min_tok_env, nullptr, 10);
        if (v >= 16u && v <= 4096u) q8_wmma_min_tokens = v;
    }
    if (std::getenv("DS4_HIP_Q8_WMMA_FAST") != nullptr &&
        warp_threads_top == 32u && n_tok >= q8_wmma_min_tokens && (in_dim & 15u) == 0u && out_dim >= 1024u) {
        const ds4_hip_repacked_q8_wmma_tensor *rw = ds4_hip_q8_wmma_lookup(model_map, model_size, weight_offset, in_dim, out_dim);
        if (rw) {
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_Q8_MATMUL_PROFILE", &prof_start, &prof_stop);
            constexpr int tiles_n = 8;
            constexpr int bm = 16;
            constexpr int bn = 16;
            constexpr int bk = 16;
            const dim3 grid((unsigned)((out_dim + (tiles_n * bn) - 1u) / (tiles_n * bn)),
                            (unsigned)((n_tok + bm - 1u) / bm), 1u);
            const unsigned threads = 256u;
            const size_t shmem = (size_t)(bm * bk + tiles_n * bk * bn) * sizeof(half) +
                                 (size_t)(tiles_n * bm * bn) * sizeof(float);
            ds4_hip_matmul_q8_wmma_packed_multin_xsplit_kernel<tiles_n, bm, bn, bk><<<grid, threads, shmem, g_stream>>>(
                    (float *)out->ptr, rw->bhalf_kn, (const float *)x->ptr,
                    (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
            const bool ok = ds4_hip_launch_ok("Q8_0 WMMA packed matmul launch");
            ds4_hip_profile_end(prof, prof_start, prof_stop, "q8_matmul_wmma_packed_multin_xsplit", in_dim, out_dim, n_tok);
            return ok ? 1 : 0;
        }
    }

    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "Q8_0 matmul");
    if (!w) return 0;
    hipEvent_t prof_start{}, prof_stop{};
    const bool prof = ds4_hip_profile_begin("DS4_HIP_Q8_MATMUL_PROFILE", &prof_start, &prof_stop);
    const char *profile_label = "q8_matmul";
    if (out_dim >= 1024u) {
        const unsigned warp_threads = ds4_hip_warp_threads();
        if (std::getenv("DS4_HIP_Q8_BATCH_FAST") != nullptr && warp_threads == 32u && n_tok > 1u && (in_dim & 31u) == 0u) {
            unsigned rows_per_block = 16u;
            if (const char *rpb_env = std::getenv("DS4_HIP_Q8_BATCH_RPB")) {
                const unsigned v = (unsigned)std::strtoul(rpb_env, nullptr, 10);
                if (v >= 1u && v <= 32u) rows_per_block = v;
            }
            unsigned tile = 8u;
            if (const char *tile_env = std::getenv("DS4_HIP_Q8_BATCH_TILE")) {
                const unsigned v = (unsigned)std::strtoul(tile_env, nullptr, 10);
                if (v == 2u || v == 4u || v == 8u || v == 16u || v == 32u) tile = v;
            }
            const bool use_sharedx_batch = std::getenv("DS4_HIP_Q8_BATCH_SHARED_X") != nullptr;
            const bool use_2row = !use_sharedx_batch && std::getenv("DS4_HIP_Q8_BATCH_NO_2ROW") == nullptr && tile <= 16u;
            const unsigned out_rows_per_block = rows_per_block * (use_2row ? 2u : 1u);
            const dim3 grid((unsigned)((out_dim + out_rows_per_block - 1u) / out_rows_per_block),
                            (unsigned)((n_tok + tile - 1u) / tile), 1u);
            const unsigned threads = rows_per_block * 32u;
            if (use_sharedx_batch) {
                profile_label = "q8_matmul_sharedx_batch";
                unsigned chunk_blocks = 16u;
                if (const char *chunk_env = std::getenv("DS4_HIP_Q8_BATCH_SHARED_X_BLOCKS")) {
                    const unsigned v = (unsigned)std::strtoul(chunk_env, nullptr, 10);
                    if (v == 8u || v == 16u || v == 32u) chunk_blocks = v;
                }
                if (chunk_blocks == 8u) {
                    ds4_hip_launch_q8_0_batch_sharedx<8>((float *)out->ptr, w, (const float *)x->ptr,
                            (uint32_t)(in_dim >> 5), (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, grid, threads, tile);
                } else if (chunk_blocks == 32u) {
                    ds4_hip_launch_q8_0_batch_sharedx<32>((float *)out->ptr, w, (const float *)x->ptr,
                            (uint32_t)(in_dim >> 5), (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, grid, threads, tile);
                } else {
                    ds4_hip_launch_q8_0_batch_sharedx<16>((float *)out->ptr, w, (const float *)x->ptr,
                            (uint32_t)(in_dim >> 5), (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, grid, threads, tile);
                }
            } else if (use_2row) {
                if (tile == 16u) {
                    ds4_hip_matmul_q8_0_warp_rows_w32_toktile_2row_kernel<16><<<grid, threads, 0, g_stream>>>(
                            (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                            (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
                } else if (tile == 8u) {
                    ds4_hip_matmul_q8_0_warp_rows_w32_toktile_2row_kernel<8><<<grid, threads, 0, g_stream>>>(
                            (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                            (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
                } else if (tile == 2u) {
                    ds4_hip_matmul_q8_0_warp_rows_w32_toktile_2row_kernel<2><<<grid, threads, 0, g_stream>>>(
                            (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                            (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
                } else {
                    ds4_hip_matmul_q8_0_warp_rows_w32_toktile_2row_kernel<4><<<grid, threads, 0, g_stream>>>(
                            (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                            (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
                }
            } else if (tile == 32u) {
                ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel<32><<<grid, threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                        (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
            } else if (tile == 16u) {
                ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel<16><<<grid, threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                        (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
            } else if (tile == 8u) {
                ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel<8><<<grid, threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                        (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
            } else if (tile == 2u) {
                ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel<2><<<grid, threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                        (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
            } else {
                ds4_hip_matmul_q8_0_warp_rows_w32_toktile_kernel<4><<<grid, threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5),
                        (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
            }
        } else if (std::getenv("DS4_HIP_QB_2ROW") != nullptr && warp_threads == 32u && n_tok == 1u && in_dim == 1024u && out_dim == 32768u) {
            const unsigned waves_per_block = 32u;
            const unsigned rows_per_block = waves_per_block * 2u;
            ds4_hip_matmul_q8_0_warp_rows_w32_2row_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    waves_per_block * 32u, 0, g_stream>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5), out_dim, row_bytes);
        } else if (n_tok == 1u && in_dim <= 8192u && 
                   (std::getenv("DS4_HIP_FORCE_SHAREDX_Q8") != nullptr ||
                    (!(warp_threads == 32u && in_dim == 1024u && out_dim >= 32768u) &&
                     !(warp_threads == 32u && in_dim == 4096u && out_dim >= 65536u)))) {
            const unsigned rows_per_block = std::max(1u, 1024u / warp_threads);
            const unsigned threads = warp_threads * rows_per_block;
            if (warp_threads == 32u && (in_dim & 31u) == 0u) {
                ds4_hip_matmul_q8_0_sharedx_rows_w32_kernel<<<
                        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                        threads, (size_t)(in_dim * sizeof(float)), g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5), out_dim, row_bytes);
            } else {
                ds4_hip_matmul_q8_0_sharedx_rows_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                                                               1u), threads, (size_t)(in_dim * sizeof(float)), g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, row_bytes);
            }
        } else {
            unsigned rows_per_block = (std::getenv("DS4_HIP_FINAL_RPB16") != nullptr && warp_threads == 32u && n_tok == 1u && in_dim == 4096u && out_dim >= 65536u) ? 16u : std::max(1u, 1024u / warp_threads);
            if (const char *rpb_env = std::getenv("DS4_HIP_Q8_RPB")) {
                const unsigned v = (unsigned)std::strtoul(rpb_env, nullptr, 10);
                if (v >= 1u && v <= 32u) rows_per_block = v;
            }
            const unsigned threads = warp_threads * rows_per_block;
            if (std::getenv("DS4_HIP_W32_WARP_ROWS") != nullptr && warp_threads == 32u && n_tok == 1u && (in_dim & 31u) == 0u) {
                ds4_hip_matmul_q8_0_warp_rows_w32_kernel<<<
                        (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                        threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5), out_dim, row_bytes);
            } else {
                ds4_hip_matmul_q8_0_warp_rows_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                                                             (unsigned)n_tok), threads, 0, g_stream>>>(
                        (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, row_bytes);
            }
        }
    } else {
        const unsigned threads = (in_dim >= 8192u) ? 1024u : 256u;
        if (ds4_hip_warp_threads() == 32u && (in_dim & 31u) == 0u) {
            ds4_hip_matmul_q8_0_kernel_w32<<<dim3((unsigned)out_dim, (unsigned)n_tok), threads, 0, g_stream>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)(in_dim >> 5), out_dim, row_bytes);
        } else {
            ds4_hip_matmul_q8_0_kernel<<<dim3((unsigned)out_dim, (unsigned)n_tok), threads, 0, g_stream>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, row_bytes);
        }
    }
    const bool ok = ds4_hip_launch_ok("Q8_0 matmul launch");
    ds4_hip_profile_end(prof, prof_start, prof_stop, profile_label, in_dim, out_dim, n_tok);
    return ok ? 1 : 0;
}

extern "C" int ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!gate || !up || !mid || !x || in_dim == 0 || out_dim == 0 || out_dim > UINT32_MAX) return 0;
    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    if (x->bytes < x_bytes || gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    const unsigned char *wg = ds4_hip_model_ptr(model_map, model_size, gate_offset, weight_bytes, "shared gate Q8_0");
    const unsigned char *wu = ds4_hip_model_ptr(model_map, model_size, up_offset, weight_bytes, "shared up Q8_0");
    if (!wg || !wu) return 0;
    const bool store_gate_up = g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr;
    const unsigned warp_threads = ds4_hip_warp_threads();
    if (warp_threads == 32u && (in_dim & 31u) == 0u) {
        if (std::getenv("DS4_HIP_SHARED_GATE_UP_ROWS") != nullptr) {
            const unsigned rows_per_block = 32u;
            ds4_hip_shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<(unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                                                                 rows_per_block * 32u, 0, g_stream>>>(
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, wg, wu, (const float *)x->ptr,
                    (uint32_t)(in_dim >> 5), out_dim, row_bytes, store_gate_up);
        } else {
            ds4_hip_shared_gate_up_swiglu_q8_0_w32_kernel<<<(unsigned)out_dim, 32, 0, g_stream>>>(
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, wg, wu, (const float *)x->ptr,
                    (uint32_t)(in_dim >> 5), out_dim, row_bytes, store_gate_up);
        }
    } else {
        ds4_hip_shared_gate_up_swiglu_q8_0_kernel<<<(unsigned)out_dim, warp_threads, 0, g_stream>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, wg, wu, (const float *)x->ptr,
                in_dim, out_dim, row_bytes, store_gate_up);
    }
    return ds4_hip_launch_ok("shared gate/up Q8_0 launch") ? 1 : 0;
}

extern "C" int ds4_metal_matmul_f16_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !x || in_dim == 0 || out_dim == 0 || n_tok == 0 || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "F16 matmul");
    if (!w) return 0;
    ds4_hip_matmul_f16_kernel<<<dim3((unsigned)out_dim, (unsigned)n_tok), 256, 0, g_stream>>>(
            (float *)out->ptr, (const uint16_t *)w, (const float *)x->ptr, in_dim, out_dim);
    return ds4_hip_launch_ok("F16 matmul launch") ? 1 : 0;
}

extern "C" int ds4_metal_matmul_f16_pair_tensor(
        ds4_metal_tensor       *out_a,
        ds4_metal_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_a || !out_b || !x || n_tok != 1 || in_dim == 0 || out_dim == 0 || out_dim > UINT32_MAX) return 0;
    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    if (x->bytes < x_bytes || out_a->bytes < out_bytes || out_b->bytes < out_bytes) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    const unsigned char *wa = ds4_hip_model_ptr(model_map, model_size, weight_a_offset, weight_bytes, "F16 pair matmul A");
    const unsigned char *wb = ds4_hip_model_ptr(model_map, model_size, weight_b_offset, weight_bytes, "F16 pair matmul B");
    if (!wa || !wb) return 0;
    ds4_hip_matmul_f16_pair_kernel<<<(unsigned)out_dim, 256, 0, g_stream>>>(
            (float *)out_a->ptr, (float *)out_b->ptr, (const uint16_t *)wa, (const uint16_t *)wb,
            (const float *)x->ptr, in_dim, out_dim);
    return ds4_hip_launch_ok("F16 pair matmul launch") ? 1 : 0;
}

extern "C" int ds4_metal_matmul_f32_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !x || in_dim == 0 || out_dim == 0 || n_tok == 0 || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(float);
    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "F32 matmul");
    if (!w) return 0;
    ds4_hip_matmul_f32_kernel<<<dim3((unsigned)out_dim, (unsigned)n_tok), 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)w, (const float *)x->ptr, in_dim, out_dim);
    return ds4_hip_launch_ok("F32 matmul launch") ? 1 : 0;
}

extern "C" int ds4_metal_repeat_hc_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t out_bytes = row_bytes * n_hc;
    if (row->bytes < row_bytes || out->bytes < out_bytes) return 0;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    ds4_hip_repeat_hc_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return ds4_hip_launch_ok("HC repeat launch") ? 1 : 0;
}

extern "C" int ds4_metal_rms_norm_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_metal_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

extern "C" int ds4_metal_rms_norm_plain_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!out || !x || n == 0 || rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    if (x->bytes < bytes || out->bytes < bytes) return 0;
    ds4_hip_rms_norm_plain_kernel<<<rows, 256, 0, g_stream>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return ds4_hip_launch_ok("plain RMS norm launch") ? 1 : 0;
}

extern "C" int ds4_metal_rms_norm_weight_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_metal_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

extern "C" int ds4_metal_rms_norm_weight_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!out || !x || n == 0 || rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    if (x->bytes < bytes || out->bytes < bytes) return 0;
    const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, (uint64_t)n * sizeof(float), "RMS norm weight");
    if (!w) return 0;
    ds4_hip_rms_norm_weight_kernel<<<rows, 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)x->ptr, (const float *)w, n, rows, eps);
    return ds4_hip_launch_ok("weighted RMS norm launch") ? 1 : 0;
}

extern "C" int ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
        ds4_metal_tensor       *q_out,
        const ds4_metal_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_metal_tensor       *kv_out,
        const ds4_metal_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!q_out || !q || !kv_out || !kv || q_n == 0 || kv_n == 0 || rows == 0) return 0;
    const uint64_t q_bytes = (uint64_t)q_n * rows * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_n * rows * sizeof(float);
    if (q->bytes < q_bytes || q_out->bytes < q_bytes || kv->bytes < kv_bytes || kv_out->bytes < kv_bytes) return 0;
    const unsigned char *qw = ds4_hip_model_ptr(model_map, model_size, q_weight_offset, (uint64_t)q_n * sizeof(float), "Q RMS norm weight");
    const unsigned char *kvw = ds4_hip_model_ptr(model_map, model_size, kv_weight_offset, (uint64_t)kv_n * sizeof(float), "KV RMS norm weight");
    if (!qw || !kvw) return 0;
    ds4_hip_qkv_rms_norm_kernel<<<dim3(rows, 2), 256, 0, g_stream>>>(
            (float *)q_out->ptr, (const float *)q->ptr, (const float *)qw, q_n,
            (float *)kv_out->ptr, (const float *)kv->ptr, (const float *)kvw, kv_n,
            rows, eps);
    return ds4_hip_launch_ok("Q/KV RMS norm launch") ? 1 : 0;
}

extern "C" int ds4_metal_head_rms_norm_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;
    const uint64_t rows = (uint64_t)n_tok * n_head;
    ds4_hip_head_rms_norm_kernel<<<(unsigned)rows, 256, 0, g_stream>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return ds4_hip_launch_ok("head RMS norm launch") ? 1 : 0;
}

extern "C" int ds4_metal_dsv4_fp8_kv_quantize_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;
    const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;
    const uint32_t groups = (head_dim - n_rot + 63u) / 64u;
    ds4_hip_fp8_kv_quant_kernel<<<dim3(n_tok, groups), 64, 0, g_stream>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return ds4_hip_launch_ok("FP8 KV quantize launch") ? 1 : 0;
}

extern "C" int ds4_metal_rope_tail_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u)) return 0;
    if (n_rot == 0) return 1;
    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;
    if (std::getenv("DS4_HIP_ROPE_TOKEN_FAST") != nullptr && n_tok > 1u) {
        const size_t shmem = (size_t)n_rot * sizeof(float);
        ds4_hip_rope_tail_token_kernel<<<n_tok, 256, shmem, g_stream>>>(
                (float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        const uint64_t total_pairs = (uint64_t)n_tok * n_head * (n_rot / 2u);
        ds4_hip_rope_tail_kernel<<<(unsigned)((total_pairs + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    return ds4_hip_launch_ok("RoPE tail launch") ? 1 : 0;
}

extern "C" int ds4_metal_kv_fp8_store_raw_tensor(
        ds4_metal_tensor *kv,
        ds4_metal_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap || head_dim == 0 || n_rot > head_dim) return 0;
    if (!ds4_metal_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot)) return 0;
    return ds4_metal_store_raw_kv_tensor(raw_cache, kv, raw_cap, row, head_dim);
}

extern "C" int ds4_metal_store_raw_kv_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0) return 0;
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    if (raw_cache->bytes < raw_bytes || kv->bytes < kv_bytes) return 0;
    ds4_hip_store_raw_kernel<<<(head_dim + 255u) / 256u, 256, 0, g_stream>>>(
            (float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim, false, 0);
    return ds4_hip_launch_ok("raw KV store launch") ? 1 : 0;
}

extern "C" int ds4_metal_store_raw_kv_batch_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0) return 0;
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (raw_cache->bytes < raw_bytes || kv->bytes < kv_bytes) return 0;
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    ds4_hip_store_raw_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim, false, 0);
    return ds4_hip_launch_ok("raw KV batch store launch") ? 1 : 0;
}

static uint64_t ds4_hip_tensor_2d_bytes(uint32_t type, uint64_t width, uint64_t rows) {
    if (type == 0u) return width * rows * sizeof(float);
    if (type == 1u) return width * rows * sizeof(uint16_t);
    if (type == 8u) return rows * (((width + 31u) / 32u) * 34u);
    return 0;
}

static int ds4_hip_rope_tail_stride_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          pos_stride,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u)) return 0;
    if (n_rot == 0) return 1;
    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;
    const uint64_t total_pairs = (uint64_t)n_tok * n_head * (n_rot / 2u);
    ds4_hip_rope_tail_stride_kernel<<<(unsigned)((total_pairs + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, pos_stride, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return ds4_hip_launch_ok("RoPE tail stride launch") ? 1 : 0;
}

static int ds4_hip_compressor_finalize_rows(
        ds4_metal_tensor *rows,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          norm_offset,
        uint32_t          head_dim,
        uint32_t          n_rows,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          pos_stride,
        uint32_t          n_ctx_orig,
        bool              quantize_fp8,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             rms_eps) {
    if (n_rows == 0) return 1;
    if (!ds4_metal_rms_norm_weight_rows_tensor(rows, rows, model_map, model_size, norm_offset, head_dim, n_rows, rms_eps)) return 0;
    if (n_rot != 0 && !ds4_hip_rope_tail_stride_tensor(rows, n_rows, 1, head_dim, n_rot, pos0, pos_stride,
                                                       n_ctx_orig, false, freq_base, freq_scale, ext_factor,
                                                       attn_factor, beta_fast, beta_slow)) return 0;
    if (quantize_fp8 && !ds4_metal_dsv4_fp8_kv_quantize_tensor(rows, n_rows, head_dim, n_rot)) return 0;
    return 1;
}

extern "C" int ds4_metal_compressor_update_tensor(
        const ds4_metal_tensor *kv_cur,
        const ds4_metal_tensor *sc_cur,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        ds4_metal_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache || !model_map ||
        head_dim == 0 || ratio == 0 || n_rot > head_dim || (n_rot & 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_bytes = ds4_hip_tensor_2d_bytes(ape_type, width, ratio);
    if (ape_bytes == 0 || kv_cur->bytes < row_bytes || sc_cur->bytes < row_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const unsigned char *ape = ds4_hip_model_ptr(model_map, model_size, ape_offset, ape_bytes, "compressor APE");
    if (!ape) return 0;
    ds4_hip_compressor_store_batch_kernel<<<(unsigned)((width + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_cur->ptr, (const float *)sc_cur->ptr, ape,
            ape_type, width, ratio, pos, 1);
    if (!ds4_hip_launch_ok("compressor update store launch")) return 0;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    if (!emit) return 1;
    const uint64_t comp_need = (uint64_t)(comp_row + 1u) * head_dim * sizeof(float);
    if (comp_cache->bytes < comp_need) return 0;
    ds4_hip_compressor_pool_state_kernel<<<(head_dim + 255u) / 256u, 256, 0, g_stream>>>(
            (float *)comp_cache->ptr, (const float *)state_kv->ptr, (const float *)state_score->ptr,
            head_dim, ratio, comp_row);
    if (!ds4_hip_launch_ok("compressor update pool launch")) return 0;
    ds4_metal_tensor *row_view = ds4_metal_tensor_view(comp_cache,
                                                       (uint64_t)comp_row * head_dim * sizeof(float),
                                                       (uint64_t)head_dim * sizeof(float));
    if (!row_view) return 0;
    const uint32_t comp_pos = pos + 1u - ratio;
    int ok = ds4_hip_compressor_finalize_rows(row_view, model_map, model_size, norm_offset, head_dim, 1,
                                               n_rot, comp_pos, 1, n_ctx_orig, false, freq_base, freq_scale,
                                               ext_factor, attn_factor, beta_fast, beta_slow, rms_eps);
    ds4_metal_tensor_free(row_view);
    if (!ok) return 0;
    if (ratio == 4u) {
        ds4_hip_ratio4_shift_kernel<<<(unsigned)((4ull * width + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        if (!ds4_hip_launch_ok("compressor ratio4 shift launch")) return 0;
    }
    return 1;
}

extern "C" int ds4_metal_compressor_store_batch_tensor(
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map || head_dim == 0 || ratio == 0 || n_tokens == 0) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = ds4_hip_tensor_2d_bytes(ape_type, width, ratio);
    if (ape_bytes == 0 || kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const unsigned char *ape = ds4_hip_model_ptr(model_map, model_size, ape_offset, ape_bytes, "compressor APE");
    if (!ape) return 0;
    const uint64_t total = (uint64_t)n_tokens * width;
    ds4_hip_compressor_store_batch_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr, ape,
            ape_type, width, ratio, pos0, n_tokens);
    return ds4_hip_launch_ok("compressor batch store launch") ? 1 : 0;
}

extern "C" int ds4_metal_compressor_prefill_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 || n_rot > head_dim || (n_rot & 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = ds4_hip_tensor_2d_bytes(ape_type, width, ratio);
    if (ape_bytes == 0 || kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes || comp_cache->bytes < comp_bytes) return 0;
    const unsigned char *ape = ds4_hip_model_ptr(model_map, model_size, ape_offset, ape_bytes, "compressor APE");
    if (!ape) return 0;

    const uint64_t state_elems = (uint64_t)state_rows * width;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_kv->ptr, state_elems, 0.0f);
    if (!ds4_hip_launch_ok("compressor state fill launch")) return 0;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_score->ptr, state_elems, -1.0e30f);
    if (!ds4_hip_launch_ok("compressor score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            const uint32_t prev_start = cutoff - ratio;
            const uint64_t off = (uint64_t)prev_start * width;
            ds4_hip_compressor_store_rows_kernel<<<(unsigned)(((uint64_t)ratio * width + 255u) / 256u), 256, 0, g_stream>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr + off, (const float *)sc->ptr + off, ape,
                    ape_type, width, ratio, pos0 + prev_start, ratio, 0);
            if (!ds4_hip_launch_ok("compressor ratio4 prev-state store launch")) return 0;
        }
        if (rem != 0) {
            const uint64_t off = (uint64_t)cutoff * width;
            ds4_hip_compressor_store_rows_kernel<<<(unsigned)(((uint64_t)rem * width + 255u) / 256u), 256, 0, g_stream>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr + off, (const float *)sc->ptr + off, ape,
                    ape_type, width, ratio, pos0 + cutoff, rem, ratio);
            if (!ds4_hip_launch_ok("compressor ratio4 rem-state store launch")) return 0;
        }
    } else if (rem != 0) {
        const uint64_t off = (uint64_t)cutoff * width;
        ds4_hip_compressor_store_rows_kernel<<<(unsigned)(((uint64_t)rem * width + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr + off, (const float *)sc->ptr + off, ape,
                ape_type, width, ratio, pos0 + cutoff, rem, 0);
        if (!ds4_hip_launch_ok("compressor rem-state store launch")) return 0;
    }

    if (n_comp != 0) {
        const uint64_t total = (uint64_t)n_comp * head_dim;
        ds4_hip_compressor_pool_prefill_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)comp_cache->ptr, nullptr, nullptr, (const float *)kv->ptr, (const float *)sc->ptr,
                ape, ape_type, head_dim, ratio, pos0, n_comp, false);
        if (!ds4_hip_launch_ok("compressor prefill pool launch")) return 0;
        if (!ds4_hip_compressor_finalize_rows(comp_cache, model_map, model_size, norm_offset, head_dim, n_comp,
                                               n_rot, pos0, ratio, n_ctx_orig, quantize_fp8, freq_base, freq_scale,
                                               ext_factor, attn_factor, beta_fast, beta_slow, rms_eps)) return 0;
    }
    return 1;
}

extern "C" int ds4_metal_compressor_prefill_ratio4_replay_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    const uint32_t ratio = 4u;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map || head_dim == 0 ||
        n_tokens == 0 || (n_tokens & 3u) || (pos0 & 3u) || n_rot > head_dim || (n_rot & 1u) || norm_type != 0u) return 0;
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = 8ull * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = ds4_hip_tensor_2d_bytes(ape_type, width, ratio);
    if (ape_bytes == 0 || kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes || comp_cache->bytes < comp_bytes) return 0;
    const unsigned char *ape = ds4_hip_model_ptr(model_map, model_size, ape_offset, ape_bytes, "compressor APE");
    if (!ape) return 0;
    const uint64_t total = (uint64_t)n_comp * head_dim;
    ds4_hip_compressor_pool_prefill_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)comp_cache->ptr, (const float *)state_kv->ptr, (const float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr, ape,
            ape_type, head_dim, ratio, pos0, n_comp, true);
    if (!ds4_hip_launch_ok("compressor ratio4 replay pool launch")) return 0;
    if (!ds4_hip_compressor_finalize_rows(comp_cache, model_map, model_size, norm_offset, head_dim, n_comp,
                                           n_rot, pos0, ratio, n_ctx_orig, quantize_fp8, freq_base, freq_scale,
                                           ext_factor, attn_factor, beta_fast, beta_slow, rms_eps)) return 0;

    const uint64_t state_elems = 8ull * width;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_kv->ptr, state_elems, 0.0f);
    if (!ds4_hip_launch_ok("compressor replay state fill launch")) return 0;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_score->ptr, state_elems, -1.0e30f);
    if (!ds4_hip_launch_ok("compressor replay score fill launch")) return 0;
    const uint32_t prev_start = n_tokens - ratio;
    const uint64_t off = (uint64_t)prev_start * width;
    ds4_hip_compressor_store_rows_kernel<<<(unsigned)(((uint64_t)ratio * width + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr + off, (const float *)sc->ptr + off, ape,
            ape_type, width, ratio, pos0 + prev_start, ratio, 0);
    return ds4_hip_launch_ok("compressor replay tail-state store launch") ? 1 : 0;
}

extern "C" int ds4_metal_compressor_prefill_state_ratio4_tensor(
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv_tail,
        const ds4_metal_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    const uint32_t ratio = 4u;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map || head_dim == 0) return 0;
    const uint32_t width = 2u * head_dim;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_elems = 8ull * width;
    const uint64_t state_bytes = state_elems * sizeof(float);
    const uint64_t ape_bytes = ds4_hip_tensor_2d_bytes(ape_type, width, ratio);
    if (ape_bytes == 0 || kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const unsigned char *ape = ds4_hip_model_ptr(model_map, model_size, ape_offset, ape_bytes, "compressor APE");
    if (!ape) return 0;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_kv->ptr, state_elems, 0.0f);
    if (!ds4_hip_launch_ok("compressor state-ratio4 fill launch")) return 0;
    ds4_hip_fill_f32_kernel<<<(unsigned)((state_elems + 255u) / 256u), 256, 0, g_stream>>>((float *)state_score->ptr, state_elems, -1.0e30f);
    if (!ds4_hip_launch_ok("compressor state-ratio4 score fill launch")) return 0;
    ds4_hip_compressor_store_rows_kernel<<<(unsigned)(((uint64_t)ratio * width + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr, ape,
            ape_type, width, ratio, pos0, ratio, 0);
    return ds4_hip_launch_ok("compressor state-ratio4 store launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_decode_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_comp,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || n_head == 0 || head_dim == 0 || raw_cap == 0) return 0;
    if (n_comp != 0 && !comp_kv) return 0;
    if (use_mask && (!comp_mask || n_comp == 0)) return 0;
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    if (heads->bytes < q_bytes || q->bytes < q_bytes || raw_kv->bytes < raw_bytes ||
        (n_comp && comp_kv->bytes < comp_bytes) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    const uint32_t rows = n_raw + n_comp;
    const size_t shmem = (size_t)(rows ? rows : 1u) * sizeof(float);
    ds4_hip_attention_decode_mixed_one_fast_kernel<<<(unsigned)n_head, 256, shmem, g_stream>>>(
            (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : nullptr,
            use_mask ? (const float *)comp_mask->ptr : nullptr,
            (const float *)sinks, n_raw, raw_cap, raw_start, n_comp, use_mask, n_head, head_dim);
    return ds4_hip_launch_ok("decode mixed attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_prefill_raw_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || n_tokens == 0 || window == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (q->bytes < q_bytes || raw_kv->bytes < raw_bytes || heads->bytes < q_bytes) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_RAW_FAST") != nullptr) {
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const size_t shmem = ((size_t)window + block + 2u) * sizeof(float);
        ds4_hip_attention_prefill_raw_scores_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr, (const float *)sinks,
                n_tokens, window, n_head, head_dim);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_prefill_raw_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr, (const float *)sinks,
                n_tokens, window, n_head, head_dim);
    }
    return ds4_hip_launch_ok("prefill raw attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_decode_raw_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || n_tokens == 0 || n_head == 0 || head_dim == 0 || raw_cap == 0) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    if (heads->bytes < q_bytes || q->bytes < q_bytes || raw_kv->bytes < raw_bytes) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_RAW_FAST") != nullptr) {
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const uint32_t score_cap = window ? window : n_raw;
        const size_t shmem = ((size_t)score_cap + block + 1u) * sizeof(float);
        ds4_hip_attention_decode_raw_batch_scores_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr, (const float *)sinks,
                n_tokens, pos0, n_raw, raw_cap, raw_start, window,
                n_head, head_dim);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_decode_batch_mixed_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                nullptr, nullptr, nullptr, (const float *)sinks,
                n_tokens, pos0, n_raw, raw_cap, raw_start, 0, 0, window, 0,
                n_head, head_dim, 0);
    }
    return ds4_hip_launch_ok("decode raw batch attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_decode_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || n_tokens == 0 || n_head == 0 || head_dim == 0 || raw_cap == 0 || ratio == 0) return 0;
    if (use_comp_mask && !comp_mask) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_tokens * n_comp * sizeof(float);
    if (heads->bytes < q_bytes || q->bytes < q_bytes || raw_kv->bytes < raw_bytes || comp_kv->bytes < comp_bytes ||
        (use_comp_mask && comp_mask->bytes < mask_bytes)) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_MIXED_FAST") != nullptr) {
        if (std::getenv("DS4_HIP_ATTENTION_PATH_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP attention decode mixed batch fast tokens=%u pos0=%u n_raw=%u n_comp=%u window=%u ratio=%u heads=%u dim=%u mask=%u\n",
                         n_tokens, pos0, n_raw, n_comp, window, ratio, n_head, head_dim, use_comp_mask);
        }
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const size_t shmem = ((size_t)(window ? window : n_raw) + n_comp + head_dim) * sizeof(float);
        ds4_hip_attention_decode_mixed_batch_warprows_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, use_comp_mask ? (const float *)comp_mask->ptr : nullptr,
                (const float *)sinks, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                window, ratio, n_head, head_dim, use_comp_mask != 0);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_decode_batch_mixed_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, use_comp_mask ? (const float *)comp_mask->ptr : nullptr, nullptr,
                (const float *)sinks, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, 0,
                window, ratio, n_head, head_dim, use_comp_mask ? 1u : 0u);
    }
    return ds4_hip_launch_ok("decode mixed batch attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_indexed_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !topk || n_tokens == 0 || n_head == 0 || head_dim == 0 || raw_cap == 0 || ratio == 0 || top_k == 0) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t topk_bytes = (uint64_t)n_tokens * top_k * sizeof(int);
    if (heads->bytes < q_bytes || q->bytes < q_bytes || raw_kv->bytes < raw_bytes || comp_kv->bytes < comp_bytes || topk->bytes < topk_bytes) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_MIXED_FAST") != nullptr) {
        if (std::getenv("DS4_HIP_ATTENTION_PATH_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP attention indexed mixed batch fast tokens=%u pos0=%u n_raw=%u n_comp=%u top_k=%u window=%u ratio=%u heads=%u dim=%u\n",
                         n_tokens, pos0, n_raw, n_comp, top_k, window, ratio, n_head, head_dim);
        }
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const size_t shmem = ((size_t)(window ? window : n_raw) + top_k + head_dim) * sizeof(float);
        ds4_hip_attention_indexed_mixed_batch_warprows_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, (const int *)topk->ptr, (const float *)sinks,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                window, ratio, n_head, head_dim);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_decode_batch_mixed_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, nullptr, (const int *)topk->ptr, (const float *)sinks,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                window, ratio, n_head, head_dim, 2u);
    }
    return ds4_hip_launch_ok("decode indexed mixed batch attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_prefill_static_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || n_tokens == 0 || ratio == 0 || n_head == 0 || head_dim == 0) return 0;
    if (n_comp != 0 && !comp_kv) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    if (q->bytes < q_bytes || raw_kv->bytes < raw_bytes || heads->bytes < q_bytes ||
        (n_comp && comp_kv->bytes < comp_bytes)) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_MIXED_FAST") != nullptr) {
        if (std::getenv("DS4_HIP_ATTENTION_PATH_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP attention prefill static mixed fast tokens=%u n_comp=%u window=%u ratio=%u heads=%u dim=%u\n",
                         n_tokens, n_comp, window, ratio, n_head, head_dim);
        }
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const size_t shmem = ((size_t)(window ? window : n_tokens) + n_comp + head_dim) * sizeof(float);
        ds4_hip_attention_prefill_static_mixed_warprows_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : nullptr, nullptr, (const float *)sinks,
                n_tokens, n_comp, window, ratio, n_head, head_dim, false);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_prefill_static_mixed_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : nullptr, nullptr, (const float *)sinks,
                n_tokens, n_comp, window, ratio, n_head, head_dim, false);
    }
    return ds4_hip_launch_ok("prefill static mixed attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_prefill_masked_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !comp_mask || n_tokens == 0 || n_comp == 0 || ratio == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_tokens * n_comp * sizeof(float);
    if (q->bytes < q_bytes || raw_kv->bytes < raw_bytes || comp_kv->bytes < comp_bytes ||
        comp_mask->bytes < mask_bytes || heads->bytes < q_bytes) return 0;
    const unsigned char *sinks = ds4_hip_model_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), "attention sinks");
    if (!sinks) return 0;
    if (std::getenv("DS4_HIP_PREFILL_MIXED_FAST") != nullptr) {
        if (std::getenv("DS4_HIP_ATTENTION_PATH_LOG") != nullptr) {
            std::fprintf(stderr, "ds4: HIP attention prefill masked mixed fast tokens=%u n_comp=%u window=%u ratio=%u heads=%u dim=%u\n",
                         n_tokens, n_comp, window, ratio, n_head, head_dim);
        }
        const unsigned block = 256u;
        const dim3 grid((unsigned)n_tokens, (unsigned)n_head, 1u);
        const size_t shmem = ((size_t)(window ? window : n_tokens) + n_comp + head_dim) * sizeof(float);
        ds4_hip_attention_prefill_static_mixed_warprows_kernel<<<grid, block, shmem, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, (const float *)comp_mask->ptr, (const float *)sinks,
                n_tokens, n_comp, window, ratio, n_head, head_dim, true);
    } else {
        const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        ds4_hip_attention_prefill_static_mixed_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)heads->ptr, (const float *)q->ptr, (const float *)raw_kv->ptr,
                (const float *)comp_kv->ptr, (const float *)comp_mask->ptr, (const float *)sinks,
                n_tokens, n_comp, window, ratio, n_head, head_dim, true);
    }
    return ds4_hip_launch_ok("prefill masked mixed attention launch") ? 1 : 0;
}

extern "C" int ds4_metal_attention_output_q8_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *low,
        ds4_metal_tensor       *group_tmp,
        ds4_metal_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_metal_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || group_dim == 0 || rank == 0 || n_groups == 0 || n_tokens == 0) return 0;
    const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
    const uint64_t low_elems = (uint64_t)n_tokens * n_groups * rank;
    const uint64_t low_bytes = low_elems * sizeof(float);
    if (heads->bytes < heads_bytes || low->bytes < low_bytes) return 0;
    const uint64_t blocks = (group_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t a_bytes = (uint64_t)n_groups * rank * row_bytes;
    const unsigned char *wa = ds4_hip_model_ptr(model_map, model_size, out_a_offset, a_bytes, "attention output A Q8_0");
    if (!wa) return 0;
    const unsigned warp_threads = ds4_hip_warp_threads();
    if ((std::getenv("DS4_HIP_Q8_GROUPED_BATCH_FAST") != nullptr || std::getenv("DS4_HIP_Q8_BATCH_FAST") != nullptr) &&
        warp_threads == 32u && n_tokens > 1u && (group_dim & 31u) == 0u && rank <= UINT32_MAX && n_groups <= UINT32_MAX) {
        unsigned tile = 4u;
        if (const char *tile_env = std::getenv("DS4_HIP_Q8_GROUPED_BATCH_TILE")) {
            const unsigned v = (unsigned)std::strtoul(tile_env, nullptr, 10);
            if (v == 2u || v == 4u) tile = v;
        }
        unsigned rows_per_block = 16u;
        if (const char *rpb_env = std::getenv("DS4_HIP_Q8_GROUPED_BATCH_RPB")) {
            const unsigned v = (unsigned)std::strtoul(rpb_env, nullptr, 10);
            if (v >= 1u && v <= 32u) rows_per_block = v;
        }
        const uint32_t row_blocks = (uint32_t)((rank + rows_per_block - 1u) / rows_per_block);
        const dim3 grid((unsigned)(n_groups * row_blocks), (unsigned)((n_tokens + tile - 1u) / tile), 1u);
        const unsigned threads = rows_per_block * 32u;
        const size_t shmem = (size_t)tile * group_dim * sizeof(float);
        if (tile == 2u) {
            ds4_hip_q8_grouped_sharedx_rows_w32_toktile_kernel<2><<<grid, threads, shmem, g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, n_tokens, n_groups,
                    (uint32_t)(group_dim >> 5), (uint32_t)rank, row_bytes);
        } else {
            ds4_hip_q8_grouped_sharedx_rows_w32_toktile_kernel<4><<<grid, threads, shmem, g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, n_tokens, n_groups,
                    (uint32_t)(group_dim >> 5), (uint32_t)rank, row_bytes);
        }
    } else if (group_dim <= 4096u && (rank % std::max(1u, 1024u / warp_threads)) == 0) {
        const unsigned rows_per_block = std::max(1u, 1024u / warp_threads);
        const unsigned threads = warp_threads * rows_per_block;
        if (warp_threads == 32u && (group_dim & 31u) == 0u) {
            ds4_hip_q8_grouped_sharedx_rows_w32_kernel<<<(unsigned)((low_elems + rows_per_block - 1u) / rows_per_block),
                                                          threads, (size_t)(group_dim * sizeof(float)), g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, n_tokens, n_groups,
                    (uint32_t)(group_dim >> 5), rank, row_bytes);
        } else {
            ds4_hip_q8_grouped_sharedx_rows_kernel<<<(unsigned)((low_elems + rows_per_block - 1u) / rows_per_block),
                                                      threads, (size_t)(group_dim * sizeof(float)), g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, n_tokens, n_groups, group_dim, rank, row_bytes);
        }
    } else {
        ds4_hip_q8_grouped_kernel<<<(unsigned)low_elems, warp_threads, 0, g_stream>>>(
                (float *)low->ptr, wa, (const float *)heads->ptr, n_tokens, n_groups, group_dim, rank, row_bytes);
    }
    if (!ds4_hip_launch_ok("attention output low Q8_0 launch")) return 0;
    return ds4_metal_matmul_q8_0_tensor(out, model_map, model_size, out_b_offset,
                                        (uint64_t)n_groups * rank, out_dim, low, n_tokens);
}

extern "C" int ds4_metal_attention_output_low_q8_tensor(
        ds4_metal_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_metal_tensor *heads) {
    if (!low || !heads || group_dim == 0 || rank == 0 || n_groups == 0) return 0;
    const uint64_t heads_bytes = (uint64_t)n_groups * group_dim * sizeof(float);
    const uint64_t low_elems = (uint64_t)n_groups * rank;
    const uint64_t low_bytes = low_elems * sizeof(float);
    if (heads->bytes < heads_bytes || low->bytes < low_bytes) return 0;
    const uint64_t blocks = (group_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t a_bytes = (uint64_t)n_groups * rank * row_bytes;
    const unsigned char *wa = ds4_hip_model_ptr(model_map, model_size, out_a_offset, a_bytes, "attention output A Q8_0");
    if (!wa) return 0;
    hipEvent_t prof_start{}, prof_stop{};
    const bool prof = ds4_hip_profile_begin("DS4_HIP_ATTN_OUT_STAGE_PROFILE", &prof_start, &prof_stop);
    const unsigned warp_threads = ds4_hip_warp_threads();
    const ds4_hip_repacked_q8_split16_tensor *swa = nullptr;
    if (warp_threads == 32u && group_dim == 4096u && rank == 1024u && n_groups == 8u &&
        std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") != nullptr) {
        swa = ds4_hip_q8_split16_repack_get(model_map, model_size, out_a_offset, group_dim, (uint64_t)n_groups * rank,
                                            "attention output A Q8_0");
    }
    if (std::getenv("DS4_HIP_DISABLE_SPLITK_ATTN_OUT_LOW") == nullptr && warp_threads == 32u && group_dim == 4096u && rank == 1024u && n_groups == 8u) {
        const uint32_t n_splits = 8u;
        const unsigned rows_per_block = 32u;
        float *partial = ds4_hip_q8_partial_scratch((uint64_t)n_splits * low_elems);
        if (!partial) return 0;
        if (swa) {
            ds4_hip_q8_grouped_split16_partial_w32_kernel<<<dim3((unsigned)((low_elems + rows_per_block - 1u) / rows_per_block), n_splits),
                                                            rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                    partial, swa->pack, (const float *)heads->ptr, n_groups, (uint32_t)rank);
        } else {
            ds4_hip_q8_grouped_partial16_w32_kernel<<<dim3((unsigned)((low_elems + rows_per_block - 1u) / rows_per_block), n_splits),
                                                      rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                    partial, wa, (const float *)heads->ptr, n_groups, (uint32_t)rank, row_bytes);
        }
        if (ds4_hip_launch_ok("attention output low split-K partial launch")) {
            ds4_hip_q8_partial_sum8_kernel<<<(unsigned)((low_elems + 255u) / 256u), 256, 0, g_stream>>>(
                    (float *)low->ptr, partial, (uint32_t)low_elems);
        }
    } else if (group_dim <= 4096u && (rank % std::max(1u, 1024u / warp_threads)) == 0) {
        const unsigned rows_per_block = std::max(1u, 1024u / warp_threads);
        const unsigned threads = warp_threads * rows_per_block;
        if (warp_threads == 32u && (group_dim & 31u) == 0u) {
            const unsigned out_rows_per_block = rows_per_block * 2u;
            ds4_hip_q8_grouped_sharedx_rows_w32_2row_kernel<<<(unsigned)((low_elems + out_rows_per_block - 1u) / out_rows_per_block),
                                                               threads, (size_t)(group_dim * sizeof(float)), g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, 1, n_groups,
                    (uint32_t)(group_dim >> 5), rank, row_bytes);
        } else {
            ds4_hip_q8_grouped_sharedx_rows_kernel<<<(unsigned)((low_elems + rows_per_block - 1u) / rows_per_block),
                                                      threads, (size_t)(group_dim * sizeof(float)), g_stream>>>(
                    (float *)low->ptr, wa, (const float *)heads->ptr, 1, n_groups, group_dim, rank, row_bytes);
        }
    } else {
        ds4_hip_q8_grouped_kernel<<<(unsigned)low_elems, warp_threads, 0, g_stream>>>(
                (float *)low->ptr, wa, (const float *)heads->ptr, 1, n_groups, group_dim, rank, row_bytes);
    }
    const bool ok = ds4_hip_launch_ok("attention output low Q8_0 launch");
    ds4_hip_profile_end(prof, prof_start, prof_stop, "attn_out_low", group_dim, low_elems, 1);
    return ok ? 1 : 0;
}

extern "C" int ds4_metal_swiglu_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *gate,
        const ds4_metal_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!out || !gate || !up || n == 0) return 0;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (out->bytes < bytes || gate->bytes < bytes || up->bytes < bytes) return 0;
    ds4_hip_swiglu_kernel<<<(n + 255u) / 256u, 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return ds4_hip_launch_ok("SwiGLU launch") ? 1 : 0;
}

extern "C" int ds4_metal_add_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *a,
        const ds4_metal_tensor *b,
        uint32_t                n) {
    if (!out || !a || !b || n == 0) return 0;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (out->bytes < bytes || a->bytes < bytes || b->bytes < bytes) return 0;
    ds4_hip_add_kernel<<<(n + 255u) / 256u, 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return ds4_hip_launch_ok("add launch") ? 1 : 0;
}

extern "C" int ds4_metal_router_select_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits) {
    (void)n_expert_groups;
    (void)n_group_used;
    if (!selected || !weights || !logits) return 0;
    if (selected->bytes < 6u * sizeof(int) || weights->bytes < 6u * sizeof(float) ||
        logits->bytes < 256u * sizeof(float)) return 0;
    if (probs && probs->bytes < 256u * sizeof(float)) return 0;
    const float *bias = nullptr;
    const int *hash = nullptr;
    uint32_t hash_row_count = hash_rows;
    if (has_bias) {
        const unsigned char *p = ds4_hip_model_ptr(model_map, model_size, bias_offset, 256u * sizeof(float), "router bias");
        if (!p) return 0;
        bias = (const float *)p;
    }
    if (hash_mode) {
        hash_row_count = hash_rows ? hash_rows : 129280u;
        const unsigned char *p = ds4_hip_model_ptr(model_map, model_size, hash_offset, (uint64_t)hash_row_count * 6u * sizeof(int), "router hash table");
        if (!p) return 0;
        hash = (const int *)p;
    }
    const bool store_probs = probs && (g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr);
    ds4_hip_router_select_parallel_kernel<<<1, 256, 0, g_stream>>>(
            (int *)selected->ptr, (float *)weights->ptr, store_probs ? (float *)probs->ptr : nullptr,
            (const float *)logits->ptr, nullptr, bias, hash, hash_row_count, token, 1, has_bias, hash_mode);
    return ds4_hip_launch_ok("router select launch") ? 1 : 0;
}

extern "C" int ds4_metal_router_select_batch_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits,
        const ds4_metal_tensor *tokens,
        uint32_t                n_tokens) {
    (void)n_expert_groups;
    (void)n_group_used;
    if (!selected || !weights || !logits || n_tokens == 0) return 0;
    if (selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float) ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float)) return 0;
    if (probs && probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float)) return 0;
    if (tokens && tokens->bytes < (uint64_t)n_tokens * sizeof(int)) return 0;
    const float *bias = nullptr;
    const int *hash = nullptr;
    uint32_t hash_row_count = hash_rows;
    if (has_bias) {
        const unsigned char *p = ds4_hip_model_ptr(model_map, model_size, bias_offset, 256u * sizeof(float), "router bias");
        if (!p) return 0;
        bias = (const float *)p;
    }
    if (hash_mode) {
        hash_row_count = hash_rows ? hash_rows : 129280u;
        const unsigned char *p = ds4_hip_model_ptr(model_map, model_size, hash_offset, (uint64_t)hash_row_count * 6u * sizeof(int), "router hash table");
        if (!p) return 0;
        hash = (const int *)p;
    }
    const bool store_probs = probs && (g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr);
    ds4_hip_router_select_parallel_kernel<<<n_tokens, 256, 0, g_stream>>>(
            (int *)selected->ptr, (float *)weights->ptr, store_probs ? (float *)probs->ptr : nullptr,
            (const float *)logits->ptr, tokens ? (const int *)tokens->ptr : nullptr,
            bias, hash, hash_row_count, 0, n_tokens, has_bias, hash_mode);
    return ds4_hip_launch_ok("router select launch") ? 1 : 0;
}

extern "C" int ds4_metal_routed_moe_one_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x) {
    return ds4_metal_routed_moe_batch_tensor(out, gate, up, mid, experts, model_map, model_size,
                                             gate_offset, up_offset, down_offset, gate_type, down_type,
                                             gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                                             expert_in_dim, expert_mid_dim, out_dim, selected, weights, n_expert,
                                             clamp, x, 1);
}

extern "C" int ds4_metal_routed_moe_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x,
        uint32_t                n_tokens) {
    (void)experts;
    if (gate_type != 10u || down_type != 10u) {
        std::fprintf(stderr, "ds4: HIP routed MoE currently supports Q2_K experts only\n");
        return 0;
    }
    if (!out || !gate || !up || !mid || !selected || !weights || !x || n_tokens == 0 || n_expert == 0) return 0;
    const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * 6u * expert_mid_dim;
    const uint64_t mid_bytes = mid_elems * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
    if (x->bytes < x_bytes || gate->bytes < mid_bytes || up->bytes < mid_bytes || mid->bytes < mid_bytes ||
        out->bytes < out_bytes || selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) return 0;
    const uint32_t routed_expert_count = 256u;
    const uint64_t gate_bytes = (uint64_t)routed_expert_count * gate_expert_bytes;
    const uint64_t up_bytes = (uint64_t)routed_expert_count * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)routed_expert_count * down_expert_bytes;
    const unsigned char *gw = ds4_hip_model_ptr(model_map, model_size, gate_offset, gate_bytes, "routed gate Q2_K");
    const unsigned char *uw = ds4_hip_model_ptr(model_map, model_size, up_offset, up_bytes, "routed up Q2_K");
    const unsigned char *dw = ds4_hip_model_ptr(model_map, model_size, down_offset, down_bytes, "routed down Q2_K");
    if (!gw || !uw || !dw) return 0;
    const unsigned warp_threads = ds4_hip_warp_threads();
    unsigned moe_gate_rows_per_block = (std::getenv("DS4_HIP_MOE_GATE_RPB4") != nullptr && warp_threads == 32u) ? 4u : 1u;
    if (const char *moe_gate_rpb = std::getenv("DS4_HIP_MOE_GATE_RPB")) {
        const unsigned v = (unsigned)std::strtoul(moe_gate_rpb, nullptr, 10);
        if (v >= 1u && v <= 32u) moe_gate_rows_per_block = v;
    }
    unsigned moe_down_rows_per_block = 1u;
    if (const char *moe_down_rpb = std::getenv("DS4_HIP_MOE_DOWN_RPB")) {
        const unsigned v = (unsigned)std::strtoul(moe_down_rpb, nullptr, 10);
        if (v >= 1u && v <= 32u) moe_down_rows_per_block = v;
    }
    const unsigned moe_gate_threads = warp_threads * moe_gate_rows_per_block;
    const unsigned moe_down_threads = warp_threads * moe_down_rows_per_block;
    dim3 gate_grid((unsigned)((expert_mid_dim + moe_gate_rows_per_block - 1u) / moe_gate_rows_per_block), (unsigned)(n_tokens * 6u), 1u);
    const bool store_gate_up = g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr;

    const uint32_t pair_stride = n_tokens * 6u;
    uint32_t expert_batch_min_tokens = 32u;
    if (const char *min_env = std::getenv("DS4_HIP_MOE_EXPERT_MIN_TOKENS")) {
        const uint32_t v = (uint32_t)std::strtoul(min_env, nullptr, 10);
        if (v >= 1u) expert_batch_min_tokens = v;
    }
    const uint64_t expert_scratch_bytes = (256ull + 256ull * pair_stride) * sizeof(int);
    const uint64_t experts_bytes = (uint64_t)n_tokens * 6u * out_dim * sizeof(float);
    const bool expert_batch = std::getenv("DS4_HIP_MOE_EXPERT_BATCH") != nullptr && n_tokens >= expert_batch_min_tokens &&
                              warp_threads == 32u && !store_gate_up && experts &&
                              gate->bytes >= expert_scratch_bytes && experts->bytes >= experts_bytes;
    if (expert_batch) {
        int *counts = (int *)gate->ptr;
        int *buckets = counts + 256u;
        if (!ds4_hip_check(hipMemsetAsync(counts, 0, 256u * sizeof(int), g_stream), "routed MoE bucket memset")) return 0;
        ds4_hip_moe_bucket_pairs_kernel<<<(unsigned)((pair_stride + 255u) / 256u), 256, 0, g_stream>>>(
                counts, buckets, (const int *)selected->ptr, n_tokens, pair_stride);
        if (!ds4_hip_launch_ok("routed MoE bucket launch")) return 0;
        ds4_hip_moe_maybe_dump_routing_counts(counts, n_tokens, routed_expert_count);
        auto parse_moe_pair_tile = [](const char *name, unsigned fallback) -> unsigned {
            if (const char *tile_env = std::getenv(name)) {
                const unsigned v = (unsigned)std::strtoul(tile_env, nullptr, 10);
                if (v == 4u || v == 8u || v == 16u) return v;
            }
            return fallback;
        };
        unsigned pair_tile = parse_moe_pair_tile("DS4_HIP_MOE_EXPERT_TILE", 8u);
        const dim3 expert_gate_grid((unsigned)((expert_mid_dim + moe_gate_rows_per_block - 1u) / moe_gate_rows_per_block),
                                    routed_expert_count, 1u);
        const dim3 expert_down_grid((unsigned)((out_dim + moe_down_rows_per_block - 1u) / moe_down_rows_per_block),
                                    routed_expert_count, 1u);
        const bool moe_shared_x = std::getenv("DS4_HIP_MOE_EXPERT_SHARED_X") != nullptr && moe_gate_rows_per_block > 1u;
        const bool moe_shared_mid = std::getenv("DS4_HIP_MOE_EXPERT_SHARED_MID") != nullptr && moe_down_rows_per_block > 1u;
        auto launch_gate_range = [&](const char *label, uint32_t min_count, uint32_t max_count, unsigned tile) -> bool {
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_MOE_PROFILE", &prof_start, &prof_stop);
            const size_t shmem = moe_shared_x ? (size_t)tile * 256u * sizeof(float) : 0u;
            if (moe_shared_x) {
                if (tile == 4u) {
                    ds4_hip_moe_q2_gate_up_expert_batch_sharedx_kernel<4><<<expert_gate_grid, moe_gate_threads, shmem, g_stream>>>(
                            (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                            counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                            gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
                } else if (tile == 16u) {
                    ds4_hip_moe_q2_gate_up_expert_batch_sharedx_kernel<16><<<expert_gate_grid, moe_gate_threads, shmem, g_stream>>>(
                            (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                            counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                            gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
                } else {
                    ds4_hip_moe_q2_gate_up_expert_batch_sharedx_kernel<8><<<expert_gate_grid, moe_gate_threads, shmem, g_stream>>>(
                            (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                            counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                            gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
                }
            } else if (tile == 4u) {
                ds4_hip_moe_q2_gate_up_expert_batch_kernel<4><<<expert_gate_grid, moe_gate_threads, 0, g_stream>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
            } else if (tile == 16u) {
                ds4_hip_moe_q2_gate_up_expert_batch_kernel<16><<<expert_gate_grid, moe_gate_threads, 0, g_stream>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
            } else {
                ds4_hip_moe_q2_gate_up_expert_batch_kernel<8><<<expert_gate_grid, moe_gate_threads, 0, g_stream>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr, (const float *)weights->ptr,
                        counts, buckets, pair_stride, min_count, max_count, expert_in_dim, expert_mid_dim,
                        gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp);
            }
            if (!ds4_hip_launch_ok("routed MoE Q2_K expert-batch gate/up launch")) return false;
            ds4_hip_profile_end(prof, prof_start, prof_stop, label, expert_in_dim, expert_mid_dim, n_tokens);
            return true;
        };
        auto launch_down_range = [&](const char *label, uint32_t min_count, uint32_t max_count, unsigned tile) -> bool {
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_MOE_PROFILE", &prof_start, &prof_stop);
            const size_t shmem = moe_shared_mid ? (size_t)tile * 256u * sizeof(float) : 0u;
            if (moe_shared_mid) {
                if (tile == 4u) {
                    ds4_hip_moe_q2_down_expert_batch_sharedmid_kernel<4><<<expert_down_grid, moe_down_threads, shmem, g_stream>>>(
                            (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                            min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (tile == 16u) {
                    ds4_hip_moe_q2_down_expert_batch_sharedmid_kernel<16><<<expert_down_grid, moe_down_threads, shmem, g_stream>>>(
                            (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                            min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    ds4_hip_moe_q2_down_expert_batch_sharedmid_kernel<8><<<expert_down_grid, moe_down_threads, shmem, g_stream>>>(
                            (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                            min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else if (tile == 4u) {
                ds4_hip_moe_q2_down_expert_batch_kernel<4><<<expert_down_grid, moe_down_threads, 0, g_stream>>>(
                        (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                        min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            } else if (tile == 16u) {
                ds4_hip_moe_q2_down_expert_batch_kernel<16><<<expert_down_grid, moe_down_threads, 0, g_stream>>>(
                        (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                        min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            } else {
                ds4_hip_moe_q2_down_expert_batch_kernel<8><<<expert_down_grid, moe_down_threads, 0, g_stream>>>(
                        (float *)experts->ptr, dw, (const float *)mid->ptr, counts, buckets, pair_stride,
                        min_count, max_count, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            }
            if (!ds4_hip_launch_ok("routed MoE Q2_K expert-batch down launch")) return false;
            ds4_hip_profile_end(prof, prof_start, prof_stop, label, expert_mid_dim, out_dim, n_tokens);
            return true;
        };
        if (!launch_gate_range("moe_q2_gate_up_expert", 1u, 0u, pair_tile)) return 0;
        if (!launch_down_range("moe_q2_down_expert", 1u, 0u, pair_tile)) return 0;
        ds4_hip_moe_experts_reduce_kernel<<<(unsigned)(((uint64_t)n_tokens * out_dim + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)out->ptr, (const float *)experts->ptr, n_tokens, out_dim);
        const bool ok = ds4_hip_launch_ok("routed MoE expert reduce launch");
        return ok ? 1 : 0;
    }

    hipEvent_t prof_start{}, prof_stop{};
    bool prof = ds4_hip_profile_begin("DS4_HIP_MOE_PROFILE", &prof_start, &prof_stop);
    ds4_hip_moe_q2_gate_up_kernel<<<gate_grid, moe_gate_threads, 0, g_stream>>>(
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, gw, uw,
            (const float *)x->ptr, (const int *)selected->ptr, (const float *)weights->ptr,
            n_tokens, expert_in_dim, expert_mid_dim,
            gate_expert_bytes, gate_row_bytes, gate_expert_bytes, gate_row_bytes, clamp, store_gate_up);
    if (!ds4_hip_launch_ok("routed MoE Q2_K gate/up launch")) return 0;
    ds4_hip_profile_end(prof, prof_start, prof_stop, "moe_q2_gate_up", expert_in_dim, expert_mid_dim, n_tokens);
    prof = ds4_hip_profile_begin("DS4_HIP_MOE_PROFILE", &prof_start, &prof_stop);
    dim3 down_grid((unsigned)((out_dim + moe_down_rows_per_block - 1u) / moe_down_rows_per_block), (unsigned)n_tokens, 1u);
    ds4_hip_moe_q2_down_kernel<<<down_grid, moe_down_threads, 0, g_stream>>>(
            (float *)out->ptr, dw, (const float *)mid->ptr, (const int *)selected->ptr,
            n_tokens, expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
    const bool ok = ds4_hip_launch_ok("routed MoE Q2_K down launch");
    ds4_hip_profile_end(prof, prof_start, prof_stop, "moe_q2_down", expert_mid_dim, out_dim, n_tokens);
    return ok ? 1 : 0;
}

extern "C" int ds4_metal_hc_split_sinkhorn_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !mix || n_hc == 0 || n_hc > 16) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t row_bytes = mix_hc * sizeof(float);
    if (mix->bytes < row_bytes || out->bytes < row_bytes) return 0;
    uint64_t rows = mix->bytes / row_bytes;
    uint64_t out_rows = out->bytes / row_bytes;
    if (rows > out_rows) rows = out_rows;
    if (rows == 0 || rows > UINT32_MAX) return 0;
    const unsigned char *scale = ds4_hip_model_ptr(model_map, model_size, scale_offset, 3u * sizeof(float), "HC split scale");
    const unsigned char *base = ds4_hip_model_ptr(model_map, model_size, base_offset, row_bytes, "HC split base");
    if (!scale || !base) return 0;
    if (n_hc == 4u) {
        ds4_hip_hc_split4_kernel<<<(unsigned)((rows + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)out->ptr, (const float *)mix->ptr, (const float *)scale, (const float *)base,
                sinkhorn_iters, (uint32_t)rows, mix_hc, eps);
    } else {
        ds4_hip_hc_split_kernel<<<(unsigned)((rows + 255u) / 256u), 256, 0, g_stream>>>(
                (float *)out->ptr, (const float *)mix->ptr, (const float *)scale, (const float *)base,
                n_hc, sinkhorn_iters, (uint32_t)rows, mix_hc, eps);
    }
    return ds4_hip_launch_ok("HC split launch") ? 1 : 0;
}

static int ds4_hip_hc_weighted_sum_strided(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *weights,
        uint64_t                weight_offset_floats,
        uint64_t                weight_stride_floats,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0 || weight_stride_floats < n_hc) return 0;
    const uint64_t out_row = (uint64_t)n_embd * sizeof(float);
    if (out->bytes < out_row || out->bytes % out_row != 0) return 0;
    const uint64_t n_tok = out->bytes / out_row;
    if (n_tok == 0 || n_tok > UINT32_MAX) return 0;
    const uint64_t residual_bytes = n_tok * (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t weights_need = (weight_offset_floats + (n_tok - 1u) * weight_stride_floats + n_hc) * sizeof(float);
    if (residual_hc->bytes < residual_bytes || weights->bytes < weights_need) return 0;
    const uint64_t n = n_tok * (uint64_t)n_embd;
    ds4_hip_hc_weighted_sum_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out->ptr, (const float *)residual_hc->ptr,
            (const float *)weights->ptr + weight_offset_floats,
            n_embd, n_hc, (uint32_t)n_tok, weight_stride_floats);
    return ds4_hip_launch_ok("HC weighted sum launch") ? 1 : 0;
}

extern "C" int ds4_metal_hc_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return ds4_hip_hc_weighted_sum_strided(out, residual_hc, weights, 0, n_hc, n_embd, n_hc);
}

extern "C" int ds4_metal_hc_weighted_sum_split_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return ds4_hip_hc_weighted_sum_strided(out, residual_hc, split, 0, mix_hc, n_embd, n_hc);
}

extern "C" int ds4_metal_hc_split_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!ds4_metal_hc_split_sinkhorn_tensor(split, mix, model_map, model_size, scale_offset, base_offset,
                                            n_hc, sinkhorn_iters, eps)) return 0;
    return ds4_metal_hc_weighted_sum_split_tensor(out, residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_metal_hc_split_weighted_sum_norm_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *norm_out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
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
    if (n_hc == 4u && out && norm_out && split && mix && residual_hc && n_embd != 0) {
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t split_bytes = 24u * sizeof(float);
        if (out->bytes >= row_bytes && out->bytes / row_bytes == 1u &&
            norm_out->bytes >= row_bytes && norm_out->bytes / row_bytes == 1u &&
            split->bytes >= split_bytes && mix->bytes >= split_bytes && residual_hc->bytes >= 4u * row_bytes) {
            const unsigned char *scale = ds4_hip_model_ptr(model_map, model_size, scale_offset, 3u * sizeof(float), "HC split scale");
            const unsigned char *base = ds4_hip_model_ptr(model_map, model_size, base_offset, split_bytes, "HC split base");
            const unsigned char *norm_w = ds4_hip_model_ptr(model_map, model_size, norm_weight_offset, row_bytes, "RMS norm weight");
            if (!scale || !base || !norm_w) return 0;
            ds4_hip_hc_split4_weighted_sum_norm_kernel<<<1, 256, 0, g_stream>>>(
                    (float *)out->ptr, (float *)norm_out->ptr, (float *)split->ptr,
                    (const float *)mix->ptr, (const float *)residual_hc->ptr,
                    (const float *)scale, (const float *)base, (const float *)norm_w,
                    n_embd, sinkhorn_iters, eps, norm_eps);
            return ds4_hip_launch_ok("HC split weighted sum norm launch") ? 1 : 0;
        }
    }
    if (!ds4_metal_hc_split_weighted_sum_tensor(out, split, mix, residual_hc, model_map, model_size,
                                                scale_offset, base_offset, n_embd, n_hc, sinkhorn_iters, eps)) return 0;
    return ds4_metal_rms_norm_weight_rows_tensor(norm_out, out, model_map, model_size, norm_weight_offset, n_embd,
                                                 (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float))), norm_eps);
}

extern "C" int ds4_metal_output_hc_weights_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || n_hc == 0) return 0;
    const uint64_t bytes = (uint64_t)n_hc * sizeof(float);
    if (out->bytes < bytes || pre->bytes < bytes) return 0;
    const unsigned char *scale = ds4_hip_model_ptr(model_map, model_size, scale_offset, sizeof(float), "output HC scale");
    const unsigned char *base = ds4_hip_model_ptr(model_map, model_size, base_offset, bytes, "output HC base");
    if (!scale || !base) return 0;
    ds4_hip_output_hc_weights_kernel<<<1, 32, 0, g_stream>>>(
            (float *)out->ptr, (const float *)pre->ptr, (const float *)scale, (const float *)base, n_hc, eps);
    return ds4_hip_launch_ok("output HC weights launch") ? 1 : 0;
}

extern "C" int ds4_metal_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *post,
        const ds4_metal_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint64_t n_tokens = out_hc->bytes / row_bytes;
    if (n_tokens == 0 || n_tokens > UINT32_MAX) return 0;
    if (residual_hc->bytes < n_tokens * row_bytes || block_out->bytes < n_tokens * (uint64_t)n_embd * sizeof(float) ||
        post->bytes < n_tokens * (uint64_t)n_hc * sizeof(float) || comb->bytes < n_tokens * (uint64_t)n_hc * n_hc * sizeof(float)) return 0;
    const uint64_t n = n_tokens * (uint64_t)n_embd * n_hc;
    ds4_hip_hc_expand_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out_hc->ptr, (const float *)block_out->ptr, nullptr, (const float *)residual_hc->ptr,
            (const float *)post->ptr, (const float *)comb->ptr, n_embd, n_hc, (uint32_t)n_tokens, false,
            n_hc, (uint64_t)n_hc * n_hc);
    return ds4_hip_launch_ok("HC expand launch") ? 1 : 0;
}

extern "C" int ds4_metal_hc_expand_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint64_t n_tokens = out_hc->bytes / row_bytes;
    if (n_tokens == 0 || n_tokens > UINT32_MAX) return 0;
    if (residual_hc->bytes < n_tokens * row_bytes || block_out->bytes < n_tokens * (uint64_t)n_embd * sizeof(float) ||
        split->bytes < n_tokens * mix_hc * sizeof(float)) return 0;
    const uint64_t n = n_tokens * (uint64_t)n_embd * n_hc;
    ds4_hip_hc_expand_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out_hc->ptr, (const float *)block_out->ptr, nullptr, (const float *)residual_hc->ptr,
            (const float *)split->ptr, nullptr, n_embd, n_hc, (uint32_t)n_tokens, true, mix_hc, 0);
    return ds4_hip_launch_ok("HC expand split launch") ? 1 : 0;
}

extern "C" int ds4_metal_hc_expand_add_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *block_add,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint64_t n_tokens = out_hc->bytes / row_bytes;
    if (n_tokens == 0 || n_tokens > UINT32_MAX) return 0;
    if (residual_hc->bytes < n_tokens * row_bytes ||
        block_out->bytes < n_tokens * (uint64_t)n_embd * sizeof(float) ||
        block_add->bytes < n_tokens * (uint64_t)n_embd * sizeof(float) ||
        split->bytes < n_tokens * mix_hc * sizeof(float)) return 0;
    const uint64_t n = n_tokens * (uint64_t)n_embd * n_hc;
    ds4_hip_hc_expand_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, g_stream>>>(
            (float *)out_hc->ptr, (const float *)block_out->ptr, (const float *)block_add->ptr,
            (const float *)residual_hc->ptr, (const float *)split->ptr, nullptr, n_embd, n_hc,
            (uint32_t)n_tokens, true, mix_hc, 0);
    return ds4_hip_launch_ok("HC expand add split launch") ? 1 : 0;
}

extern "C" int ds4_metal_shared_down_hc_expand_q8_0_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *shared_mid,
        const ds4_metal_tensor *routed_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (out_hc && shared_out && shared_mid && routed_out && residual_hc && split && n_embd == out_dim && n_hc != 0 &&
        ds4_hip_warp_threads() == 32u && (in_dim & 31u) == 0u && out_dim >= 1024u && in_dim <= 8192u) {
        const uint64_t out_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * out_dim * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (out_hc->bytes >= hc_bytes && shared_out->bytes >= out_bytes && shared_mid->bytes >= in_dim * sizeof(float) &&
            routed_out->bytes >= out_bytes && residual_hc->bytes >= hc_bytes && split->bytes >= mix_hc * sizeof(float)) {
            const uint64_t blocks = (in_dim + 31u) / 32u;
            const uint64_t row_bytes = blocks * 34u;
            const uint64_t weight_bytes = out_dim * row_bytes;
            const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "shared down HC Q8_0");
            if (!w) return 0;
            const ds4_hip_repacked_q8_split16_tensor *sw = nullptr;
            if (std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") != nullptr) {
                sw = ds4_hip_q8_split16_repack_get(model_map, model_size, weight_offset, in_dim, out_dim,
                                                   "shared down HC Q8_0");
            }
            const unsigned rows_per_block = 32u;
            const bool store_block = g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr;
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_ATTN_OUT_STAGE_PROFILE", &prof_start, &prof_stop);
            bool ok = true;
            if (std::getenv("DS4_HIP_DISABLE_SPLITK_SHARED_DOWN") == nullptr && in_dim == 2048u && out_dim == 4096u && n_hc == 4u) {
                const uint32_t n_splits = 4u;
                float *partial = ds4_hip_q8_partial_scratch((uint64_t)n_splits * out_dim);
                if (!partial) return 0;
                if (sw) {
                    ds4_hip_q8_split16_partial_w32_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block), n_splits),
                                                            rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                            partial, sw->pack, (const float *)shared_mid->ptr, (uint32_t)out_dim);
                } else {
                    ds4_hip_matmul_q8_0_hc_partial16_w32_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block), n_splits),
                                                                   rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                            partial, w, (const float *)shared_mid->ptr, (uint32_t)out_dim, row_bytes);
                }
                ok = ds4_hip_launch_ok("shared down HC partial Q8_0 launch");
                if (ok) {
                    ds4_hip_hc_expand_add_partial4_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256, 0, g_stream>>>(
                            (float *)out_hc->ptr, (float *)shared_out->ptr, partial, (const float *)routed_out->ptr,
                            (const float *)residual_hc->ptr, (const float *)split->ptr,
                            (uint32_t)out_dim, n_hc, store_block);
                    ok = ds4_hip_launch_ok("shared down HC partial expand launch");
                }
            } else {
                ds4_hip_matmul_q8_0_hc_expand_w32_kernel<<<(unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                                                            1024, (size_t)(in_dim * sizeof(float)), g_stream>>>(
                        (float *)out_hc->ptr, (float *)shared_out->ptr, w, (const float *)shared_mid->ptr,
                        (const float *)routed_out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
                        (uint32_t)(in_dim >> 5), out_dim, row_bytes, n_hc, store_block);
                ok = ds4_hip_launch_ok("shared down HC Q8_0 launch");
            }
            ds4_hip_profile_end(prof, prof_start, prof_stop, "shared_down_hc_q8", in_dim, out_dim, 1);
            return ok ? 1 : 0;
        }
    }
    if (!ds4_metal_matmul_q8_0_tensor(shared_out, model_map, model_size, weight_offset, in_dim, out_dim, shared_mid, 1)) return 0;
    return ds4_metal_hc_expand_add_split_tensor(out_hc, routed_out, shared_out, residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_metal_matmul_q8_0_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (out_hc && block_out && x && residual_hc && split && n_embd == out_dim && n_hc != 0 &&
        ds4_hip_warp_threads() == 32u && (in_dim & 31u) == 0u && out_dim >= 1024u && in_dim <= 8192u) {
        const uint64_t out_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * out_dim * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (out_hc->bytes >= hc_bytes && block_out->bytes >= out_bytes && x->bytes >= in_dim * sizeof(float) &&
            residual_hc->bytes >= hc_bytes && split->bytes >= mix_hc * sizeof(float)) {
            const uint64_t blocks = (in_dim + 31u) / 32u;
            const uint64_t row_bytes = blocks * 34u;
            const uint64_t weight_bytes = out_dim * row_bytes;
            const unsigned char *w = ds4_hip_model_ptr(model_map, model_size, weight_offset, weight_bytes, "Q8_0 HC expand matmul");
            if (!w) return 0;
            const ds4_hip_repacked_q8_split16_tensor *sw = nullptr;
            if (std::getenv("DS4_HIP_Q8_REPACK_SPLIT16") != nullptr) {
                sw = ds4_hip_q8_split16_repack_get(model_map, model_size, weight_offset, in_dim, out_dim,
                                                   "Q8_0 HC expand matmul");
            }
            const bool use_splitk_attn_out_b = std::getenv("DS4_HIP_DISABLE_SPLITK_ATTN_OUT_B") == nullptr && in_dim == 8192u && out_dim == 4096u && n_hc == 4u;
            const unsigned rows_per_block = (std::getenv("DS4_HIP_ATTN_B_RPB16") != nullptr && use_splitk_attn_out_b) ? 16u : 32u;
            const bool store_block = g_quality || std::getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != nullptr;
            hipEvent_t prof_start{}, prof_stop{};
            const bool prof = ds4_hip_profile_begin("DS4_HIP_ATTN_OUT_STAGE_PROFILE", &prof_start, &prof_stop);
            bool ok = true;
            if (use_splitk_attn_out_b) {
                const uint32_t n_splits = 16u;
                float *partial = ds4_hip_q8_partial_scratch((uint64_t)n_splits * out_dim);
                if (!partial) return 0;
                if (sw) {
                    ds4_hip_q8_split16_partial_w32_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block), n_splits),
                                                            rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                            partial, sw->pack, (const float *)x->ptr, (uint32_t)out_dim);
                } else {
                    ds4_hip_matmul_q8_0_hc_partial16_w32_kernel<<<dim3((unsigned)((out_dim + rows_per_block - 1u) / rows_per_block), n_splits),
                                                                   rows_per_block * 32u, 512u * sizeof(float), g_stream>>>(
                            partial, w, (const float *)x->ptr, (uint32_t)out_dim, row_bytes);
                }
                ok = ds4_hip_launch_ok("Q8_0 HC partial matmul launch");
                if (ok) {
                    ds4_hip_hc_expand_partial16_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256, 0, g_stream>>>(
                            (float *)out_hc->ptr, (float *)block_out->ptr, partial,
                            (const float *)residual_hc->ptr, (const float *)split->ptr,
                            (uint32_t)out_dim, n_hc, store_block);
                    ok = ds4_hip_launch_ok("Q8_0 HC partial expand launch");
                }
            } else {
                ds4_hip_matmul_q8_0_hc_expand_w32_kernel<<<(unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                                                            1024, (size_t)(in_dim * sizeof(float)), g_stream>>>(
                        (float *)out_hc->ptr, (float *)block_out->ptr, w, (const float *)x->ptr,
                        nullptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
                        (uint32_t)(in_dim >> 5), out_dim, row_bytes, n_hc, store_block);
                ok = ds4_hip_launch_ok("Q8_0 HC expand matmul launch");
            }
            ds4_hip_profile_end(prof, prof_start, prof_stop, "attn_out_b_hc_q8", in_dim, out_dim, 1);
            return ok ? 1 : 0;
        }
    }
    if (!ds4_metal_matmul_q8_0_tensor(block_out, model_map, model_size, weight_offset, in_dim, out_dim, x, 1)) return 0;
    return ds4_metal_hc_expand_split_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc);
}
