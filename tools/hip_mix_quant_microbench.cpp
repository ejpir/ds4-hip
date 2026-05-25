// Standalone Mix-Quant-style ROCm microbench.
// Build with: make hip-mix-quant-bench
//
// This is intentionally outside the engine.  It checks the core premise behind
// Mix-Quant for our ROCm target: can a low-bit prefill GEMM beat a precise
// f16/BF16-class GEMM enough to justify a phase split?
//
// Important: this is NOT NVIDIA NVFP4 tensor-core execution.  gfx1151 has no
// NVFP4 path, so the FP4 variants below use an E2M1/NVFP4-ish 4-bit format with
// per-16-value scales, dequantize into LDS half tiles, then use rocWMMA.  If this
// software-dequant version is not competitive, DS4 would need real hardware FP4
// or a much more specialized kernel before the Mix-Quant approach is attractive.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma-version.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(x) do { \
    hipError_t err__ = (x); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

__host__ __device__ static inline float e2m1_abs_value(unsigned m) {
    switch (m & 7u) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__host__ __device__ static inline float e2m1_decode(unsigned code) {
    const float v = e2m1_abs_value(code & 7u);
    return (code & 8u) ? -v : v;
}

__host__ __device__ static inline unsigned e2m1_quantize(float x, float scale) {
    if (!(scale > 0.0f) || x == 0.0f) return 0u;
    const unsigned sign = x < 0.0f ? 8u : 0u;
    const float y = fabsf(x) / scale;
    unsigned best = 0u;
    float best_err = fabsf(y - e2m1_abs_value(0));
#pragma unroll
    for (unsigned m = 1; m < 8; ++m) {
        const float err = fabsf(y - e2m1_abs_value(m));
        if (err < best_err) {
            best_err = err;
            best = m;
        }
    }
    return sign | best;
}

__device__ static inline float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down(v, offset, 32));
    return v;
}

__device__ static inline float fp4_load(const unsigned char* __restrict__ q,
                                        const half* __restrict__ scales,
                                        uint64_t idx,
                                        uint64_t scale_idx) {
    const unsigned char byte = q[idx >> 1];
    const unsigned code = (idx & 1u) ? (byte >> 4) : (byte & 15u);
    return e2m1_decode(code) * __half2float(scales[scale_idx]);
}

__global__ void fp4_quantize_a_kernel(unsigned char* __restrict__ q,
                                      half* __restrict__ scales,
                                      const half* __restrict__ a,
                                      unsigned M,
                                      unsigned K) {
    const unsigned m = blockIdx.x;
    const unsigned kg = blockIdx.y;
    const unsigned lane = threadIdx.x & 31u;
    if (m >= M) return;

    float av = 0.0f;
    if (lane < 16u) {
        const unsigned k = kg * 16u + lane;
        av = fabsf(__half2float(a[(uint64_t)m * K + k]));
    }
    float max_abs = warp_reduce_max(av);
    max_abs = __shfl(max_abs, 0, 32);
    const float scale = max_abs > 0.0f ? (max_abs / 6.0f) : 1.0f;
    if (lane == 0) scales[(uint64_t)m * (K / 16u) + kg] = __float2half(scale);

    if (lane < 8u) {
        const unsigned k0 = kg * 16u + lane * 2u;
        const uint64_t idx0 = (uint64_t)m * K + k0;
        const float v0 = __half2float(a[idx0]);
        const float v1 = __half2float(a[idx0 + 1u]);
        const unsigned c0 = e2m1_quantize(v0, scale);
        const unsigned c1 = e2m1_quantize(v1, scale);
        q[idx0 >> 1] = (unsigned char)(c0 | (c1 << 4));
    }
}

template <int BM=16, int BN=16, int BK=16>
__global__ void wmma_f16_kernel(float* __restrict__ out,
                                const half* __restrict__ a,
                                const half* __restrict__ b,
                                unsigned M,
                                unsigned K,
                                unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
    const unsigned tid = threadIdx.x;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a af;
    frag_b bf;
    frag_c acc;
    if (tid < 32u) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            shA[j] = a[(uint64_t)(m0 + mm) * K + k0 + kk];
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            shB[j] = b[(uint64_t)(k0 + kk) * N + n0 + nn];
        }
        __syncthreads();
        if (tid < 32u) {
            rocwmma::load_matrix_sync(af, shA, BK);
            rocwmma::load_matrix_sync(bf, shB, BN);
            rocwmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    if (tid < 32u) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
}

template <int BM=16, int BN=16, int BK=16>
__global__ void wmma_fp4w_kernel(float* __restrict__ out,
                                 const half* __restrict__ a,
                                 const unsigned char* __restrict__ bq,
                                 const half* __restrict__ bscale,
                                 unsigned M,
                                 unsigned K,
                                 unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
    const unsigned tid = threadIdx.x;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a af;
    frag_b bf;
    frag_c acc;
    if (tid < 32u) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            shA[j] = a[(uint64_t)(m0 + mm) * K + k0 + kk];
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned k = k0 + kk;
            const unsigned n = n0 + nn;
            shB[j] = __float2half(fp4_load(bq, bscale, (uint64_t)k * N + n, (uint64_t)(k >> 4) * N + n));
        }
        __syncthreads();
        if (tid < 32u) {
            rocwmma::load_matrix_sync(af, shA, BK);
            rocwmma::load_matrix_sync(bf, shB, BN);
            rocwmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    if (tid < 32u) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
}

template <int BM=16, int BN=16, int BK=16>
__global__ void wmma_fp4xw_kernel(float* __restrict__ out,
                                  const unsigned char* __restrict__ aq,
                                  const half* __restrict__ ascale,
                                  const unsigned char* __restrict__ bq,
                                  const half* __restrict__ bscale,
                                  unsigned M,
                                  unsigned K,
                                  unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
    const unsigned tid = threadIdx.x;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a af;
    frag_b bf;
    frag_c acc;
    if (tid < 32u) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            const unsigned m = m0 + mm;
            const unsigned k = k0 + kk;
            shA[j] = __float2half(fp4_load(aq, ascale, (uint64_t)m * K + k, (uint64_t)m * (K / 16u) + (k >> 4)));
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned k = k0 + kk;
            const unsigned n = n0 + nn;
            shB[j] = __float2half(fp4_load(bq, bscale, (uint64_t)k * N + n, (uint64_t)(k >> 4) * N + n));
        }
        __syncthreads();
        if (tid < 32u) {
            rocwmma::load_matrix_sync(af, shA, BK);
            rocwmma::load_matrix_sync(bf, shB, BN);
            rocwmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    if (tid < 32u) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
}

static half f32_to_half(float x) {
    return __float2half(x);
}

static void set_q4(std::vector<unsigned char>& q, uint64_t idx, unsigned code) {
    unsigned char& byte = q[idx >> 1];
    if (idx & 1u) byte = (unsigned char)((byte & 0x0fu) | ((code & 15u) << 4));
    else byte = (unsigned char)((byte & 0xf0u) | (code & 15u));
}

static void quantize_b_fp4(std::vector<unsigned char>& bq,
                           std::vector<half>& bscale,
                           const std::vector<float>& b,
                           unsigned K,
                           unsigned N) {
    std::fill(bq.begin(), bq.end(), 0);
    for (unsigned kg = 0; kg < K / 16u; ++kg) {
        for (unsigned n = 0; n < N; ++n) {
            float max_abs = 0.0f;
            for (unsigned kk = 0; kk < 16u; ++kk) {
                max_abs = std::max(max_abs, std::fabs(b[(uint64_t)(kg * 16u + kk) * N + n]));
            }
            const float scale = max_abs > 0.0f ? (max_abs / 6.0f) : 1.0f;
            bscale[(uint64_t)kg * N + n] = f32_to_half(scale);
            for (unsigned kk = 0; kk < 16u; ++kk) {
                const uint64_t idx = (uint64_t)(kg * 16u + kk) * N + n;
                set_q4(bq, idx, e2m1_quantize(b[idx], scale));
            }
        }
    }
}

template <typename LaunchFn>
static float time_kernel(LaunchFn launch, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / std::max(1, iters);
}

static void compare_outputs(const char* name,
                            const std::vector<float>& ref,
                            const std::vector<float>& got) {
    double max_abs = 0.0, rms = 0.0, denom = 0.0, dot = 0.0, got_norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double g = got[i];
        const double e = g - r;
        max_abs = std::max(max_abs, std::abs(e));
        rms += e * e;
        denom += r * r;
        got_norm += g * g;
        dot += r * g;
    }
    rms = std::sqrt(rms / std::max<size_t>(1, ref.size()));
    const double rel_rms = std::sqrt((rms * rms * ref.size()) / std::max(1.0, denom));
    const double cos = dot / std::sqrt(std::max(1.0e-30, denom * got_norm));
    std::printf("%s_vs_f16: max_abs=%.6g rms=%.6g rel_rms=%.6g cosine=%.8f\n",
                name, max_abs, rms, rel_rms, cos);
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [M K N iters]\n"
        "  M=tokens, K=input dim, N=output dim; require M%%16=0 K%%16=0 N%%16=0\n"
        "  default: M=128 K=4096 N=4096 iters=50\n"
        "  examples:\n"
        "    %s 128 4096 4096 50   # prefill-like DS4 hidden projection\n"
        "    %s 16 4096 4096 100    # decode-tile proxy; Mix-Quant would keep this precise\n"
        "    %s 512 4096 2048 30    # routed-expert gate/up-ish output width\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char** argv) {
    unsigned M = 128, K = 4096, N = 4096;
    int iters = 50;
    if (argc > 1 && (!std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help"))) {
        usage(argv[0]);
        return 0;
    }
    if (argc >= 4) {
        M = (unsigned)std::strtoul(argv[1], nullptr, 10);
        K = (unsigned)std::strtoul(argv[2], nullptr, 10);
        N = (unsigned)std::strtoul(argv[3], nullptr, 10);
    } else if (argc != 1) {
        usage(argv[0]);
        return 2;
    }
    if (argc >= 5) iters = std::atoi(argv[4]);
    if ((M % 16u) || (K % 16u) || (N % 16u) || iters <= 0) {
        usage(argv[0]);
        return 2;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    std::printf("mix-quant microbench: M=%u tokens K=%u N=%u iters=%d\n", M, K, N, iters);
    std::printf("note: FP4 is software E2M1 dequant -> LDS half -> rocWMMA, not NVFP4 tensor cores.\n");

    const uint64_t a_elems = (uint64_t)M * K;
    const uint64_t b_elems = (uint64_t)K * N;
    const uint64_t c_elems = (uint64_t)M * N;
    const uint64_t aq_bytes = (a_elems + 1u) / 2u;
    const uint64_t bq_bytes = (b_elems + 1u) / 2u;
    const uint64_t ascale_elems = (uint64_t)M * (K / 16u);
    const uint64_t bscale_elems = (uint64_t)(K / 16u) * N;

    std::vector<half> hA(a_elems), hBhalf(b_elems), hBscale(bscale_elems);
    std::vector<float> hB(b_elems);
    std::vector<unsigned char> hBq(bq_bytes);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::normal_distribution<float> bdist(0.0f, 0.02f);
    for (uint64_t i = 0; i < a_elems; ++i) hA[i] = f32_to_half(adist(rng));
    for (uint64_t i = 0; i < b_elems; ++i) {
        const float v = bdist(rng);
        hB[i] = v;
        hBhalf[i] = f32_to_half(v);
    }
    quantize_b_fp4(hBq, hBscale, hB, K, N);

    const double b_f16_mib = (double)b_elems * sizeof(half) / 1048576.0;
    const double b_fp4_mib = ((double)bq_bytes + (double)bscale_elems * sizeof(half)) / 1048576.0;
    const double a_f16_mib = (double)a_elems * sizeof(half) / 1048576.0;
    const double a_fp4_mib = ((double)aq_bytes + (double)ascale_elems * sizeof(half)) / 1048576.0;
    std::printf("B precise half footprint: %.2f MiB\n", b_f16_mib);
    std::printf("B FP4+scale footprint:    %.2f MiB  compression %.2fx\n", b_fp4_mib, b_f16_mib / b_fp4_mib);
    std::printf("A half per chunk:         %.2f MiB\n", a_f16_mib);
    std::printf("A FP4+scale per chunk:    %.2f MiB  compression %.2fx\n", a_fp4_mib, a_f16_mib / a_fp4_mib);

    half *dA = nullptr, *dB = nullptr, *dAscale = nullptr, *dBscale = nullptr;
    unsigned char *dAq = nullptr, *dBq = nullptr;
    float *dF16 = nullptr, *dFP4W = nullptr, *dFP4XW = nullptr;
    HIP_CHECK(hipMalloc(&dA, a_elems * sizeof(half)));
    HIP_CHECK(hipMalloc(&dB, b_elems * sizeof(half)));
    HIP_CHECK(hipMalloc(&dAq, aq_bytes));
    HIP_CHECK(hipMalloc(&dBq, bq_bytes));
    HIP_CHECK(hipMalloc(&dAscale, ascale_elems * sizeof(half)));
    HIP_CHECK(hipMalloc(&dBscale, bscale_elems * sizeof(half)));
    HIP_CHECK(hipMalloc(&dF16, c_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dFP4W, c_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dFP4XW, c_elems * sizeof(float)));

    HIP_CHECK(hipMemcpy(dA, hA.data(), a_elems * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hBhalf.data(), b_elems * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBq, hBq.data(), bq_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBscale, hBscale.data(), bscale_elems * sizeof(half), hipMemcpyHostToDevice));

    dim3 quant_grid(M, K / 16u, 1);
    dim3 quant_block(32, 1, 1);
    auto launch_quant_a = [&]() {
        hipLaunchKernelGGL(fp4_quantize_a_kernel, quant_grid, quant_block, 0, 0, dAq, dAscale, dA, M, K);
    };

    dim3 wmma_grid(N / 16u, M / 16u, 1);
    dim3 wmma_block(256, 1, 1);
    size_t wmma_shmem = (16 * 16 + 16 * 16) * sizeof(half);
    auto launch_f16 = [&]() {
        hipLaunchKernelGGL((wmma_f16_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0, dF16, dA, dB, M, K, N);
    };
    auto launch_fp4w = [&]() {
        hipLaunchKernelGGL((wmma_fp4w_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0, dFP4W, dA, dBq, dBscale, M, K, N);
    };
    auto launch_fp4xw = [&]() {
        hipLaunchKernelGGL((wmma_fp4xw_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0, dFP4XW, dAq, dAscale, dBq, dBscale, M, K, N);
    };

    launch_quant_a(); HIP_CHECK(hipGetLastError());
    launch_f16(); HIP_CHECK(hipGetLastError());
    launch_fp4w(); HIP_CHECK(hipGetLastError());
    launch_fp4xw(); HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hF16(c_elems), hFP4W(c_elems), hFP4XW(c_elems);
    HIP_CHECK(hipMemcpy(hF16.data(), dF16, c_elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hFP4W.data(), dFP4W, c_elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hFP4XW.data(), dFP4XW, c_elems * sizeof(float), hipMemcpyDeviceToHost));
    compare_outputs("fp4_weight", hF16, hFP4W);
    compare_outputs("fp4_activation_weight", hF16, hFP4XW);

    const float quant_a_ms = time_kernel(launch_quant_a, iters);
    const float f16_ms = time_kernel(launch_f16, iters);
    const float fp4w_ms = time_kernel(launch_fp4w, iters);
    const float fp4xw_ms = time_kernel(launch_fp4xw, iters);
    const double flops = 2.0 * (double)M * (double)N * (double)K;

    std::printf("A quantize half->FP4:    %.4f ms  %.2f GiB/s input+output\n",
                quant_a_ms,
                (((double)a_elems * sizeof(half) + (double)aq_bytes + (double)ascale_elems * sizeof(half)) / 1073741824.0) / (quant_a_ms * 1.0e-3));
    std::printf("precise f16 WMMA:        %.4f ms  %.2f TFLOP/s\n",
                f16_ms, flops / (f16_ms * 1.0e-3) / 1.0e12);
    std::printf("FP4 weight prefill:      %.4f ms  %.2f effective TFLOP/s  speedup %.3fx\n",
                fp4w_ms, flops / (fp4w_ms * 1.0e-3) / 1.0e12, f16_ms / fp4w_ms);
    std::printf("FP4 activation+weight:   %.4f ms  %.2f effective TFLOP/s  speedup %.3fx\n",
                fp4xw_ms, flops / (fp4xw_ms * 1.0e-3) / 1.0e12, f16_ms / fp4xw_ms);
    std::printf("FP4 a+w incl A quantize: %.4f ms  speedup %.3fx\n",
                fp4xw_ms + quant_a_ms, f16_ms / (fp4xw_ms + quant_a_ms));

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dAq));
    HIP_CHECK(hipFree(dBq));
    HIP_CHECK(hipFree(dAscale));
    HIP_CHECK(hipFree(dBscale));
    HIP_CHECK(hipFree(dF16));
    HIP_CHECK(hipFree(dFP4W));
    HIP_CHECK(hipFree(dFP4XW));
    return 0;
}
