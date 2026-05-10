// Standalone Q2_K routed MoE down-projection rocWMMA microbench.
// Build with: make hip-q2-moe-wmma-bench
//
// This intentionally stays outside the engine. It compares a current-style
// wave-row Q2_K down kernel against a first rocWMMA prototype that dequantizes
// Q2_K weight tiles and float activations into LDS FP16, then uses
// FP16 x FP16 -> FP32 WMMA. It is a microbench, not an integrated path.

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

static constexpr int QK_K = 256;
static constexpr int Q2K_BYTES = 84;

struct block_q2_K_host {
    unsigned char scales[16];
    unsigned char qs[64];
    unsigned short d;
    unsigned short dmin;
};
static_assert(sizeof(block_q2_K_host) == Q2K_BYTES, "Q2_K block size");

__host__ __device__ static inline float f16_bits_to_f32(unsigned short h) {
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static unsigned short f32_to_f16_bits(float x) {
    half h = __float2half(x);
    unsigned short u;
    std::memcpy(&u, &h, sizeof(u));
    return u;
}

__host__ __device__ static inline float q2_k_dequant(const unsigned char* blk, unsigned i) {
    const unsigned char* sc = blk;
    const unsigned char* qs = blk + 16u;
    const unsigned g = i >> 4;
    const unsigned within = g & 7u;
    const unsigned chunk = g >> 3;
    const unsigned shift = (within >> 1) * 2u;
    const unsigned half = within & 1u;
    const unsigned lane = i & 15u;
    const unsigned qi = chunk * 32u + half * 16u + lane;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float d = f16_bits_to_f32((unsigned short)blk[80] | ((unsigned short)blk[81] << 8));
    const float dmin = f16_bits_to_f32((unsigned short)blk[82] | ((unsigned short)blk[83] << 8));
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4);
    return d * scale * q - dmin * mn;
}

__device__ static inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}

template <unsigned PAIR_TILE>
__global__ void q2_down_current_like_kernel(float* __restrict__ out,
                                            const unsigned char* __restrict__ w,
                                            const float* __restrict__ mid,
                                            unsigned M, unsigned K, unsigned N,
                                            unsigned row_bytes) {
    extern __shared__ float shmid[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned row = blockIdx.x * rows_per_block + wave;
    if (row >= N) return;
    const unsigned nb = K >> 8;
    const unsigned char* wrow = w + (uint64_t)row * row_bytes;
    for (unsigned p0 = 0; p0 < M; p0 += PAIR_TILE) {
        float acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) acc[u] = 0.0f;
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t mbase = (uint64_t)b * 256u;
            for (unsigned j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shmid[j] = (p0 + u < M) ? mid[(uint64_t)(p0 + u) * K + mbase + k] : 0.0f;
            }
            __syncthreads();
            const unsigned char* blk = wrow + (uint64_t)b * Q2K_BYTES;
#pragma unroll
            for (unsigned kk = 0; kk < 8u; ++kk) {
                const unsigned i = lane + (kk << 5);
                const float wv = q2_k_dequant(blk, i);
#pragma unroll
                for (unsigned u = 0; u < PAIR_TILE; ++u) acc[u] += wv * shmid[(u << 8) + i];
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) acc[u] = warp_reduce_sum(acc[u]);
        if (lane == 0) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (p0 + u < M) out[(uint64_t)(p0 + u) * N + row] = acc[u];
            }
        }
    }
}

__global__ void q2_repack_half_kn_kernel(half* __restrict__ bhalf,
                                          const unsigned char* __restrict__ w,
                                          unsigned K, unsigned N,
                                          unsigned row_bytes) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)K * N;
    if (idx >= total) return;
    const unsigned k = (unsigned)(idx / N);
    const unsigned row = (unsigned)(idx - (uint64_t)k * N);
    const unsigned char* blk = w + (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
    bhalf[idx] = __float2half(q2_k_dequant(blk, k & 255u));
}

template <int BM=16, int BN=16, int BK=16>
__global__ void q2_down_wmma_kernel(float* __restrict__ out,
                                    const unsigned char* __restrict__ w,
                                    const float* __restrict__ mid,
                                    unsigned M, unsigned K, unsigned N,
                                    unsigned row_bytes) {
    extern __shared__ half sh[];
    half* shA = sh;                 // BM x BK row-major
    half* shB = sh + BM * BK;       // BK x BN row-major

    const unsigned tile_m = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m0 = tile_m * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    if (tid < 32) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            shA[j] = __float2half(mid[(uint64_t)(m0 + mm) * K + k0 + kk]);
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            const unsigned k = k0 + kk;
            const unsigned char* blk = w + (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
            shB[j] = __float2half(q2_k_dequant(blk, k & 255u));
        }
        __syncthreads();
        if (tid < 32) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (tid < 32) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
}

template <int BM=16, int BN=16, int BK=16>
__global__ void q2_down_wmma_repacked_kernel(float* __restrict__ out,
                                             const half* __restrict__ bhalf,
                                             const float* __restrict__ mid,
                                             unsigned M, unsigned K, unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;

    const unsigned tile_m = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m0 = tile_m * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    if (tid < 32) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            shA[j] = __float2half(mid[(uint64_t)(m0 + mm) * K + k0 + kk]);
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            shB[j] = bhalf[(uint64_t)(k0 + kk) * N + n0 + nn];
        }
        __syncthreads();
        if (tid < 32) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (tid < 32) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
}

static void fill_q2_weights(std::vector<unsigned char>& w, unsigned K, unsigned N) {
    const unsigned nb = K / QK_K;
    const unsigned row_bytes = nb * Q2K_BYTES;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> qdist(0, 3);
    std::uniform_int_distribution<int> sdist(1, 15);
    for (unsigned row = 0; row < N; ++row) {
        for (unsigned b = 0; b < nb; ++b) {
            auto* blk = reinterpret_cast<block_q2_K_host*>(w.data() + (uint64_t)row * row_bytes + b * Q2K_BYTES);
            for (int g = 0; g < 16; ++g) blk->scales[g] = (unsigned char)(sdist(rng) & 0x0f); // min nibble = 0
            std::fill(std::begin(blk->qs), std::end(blk->qs), 0);
            for (int i = 0; i < 256; ++i) {
                const unsigned g = i >> 4;
                const unsigned within = g & 7u;
                const unsigned chunk = g >> 3;
                const unsigned shift = (within >> 1) * 2u;
                const unsigned half = within & 1u;
                const unsigned lane = i & 15u;
                const unsigned qi = chunk * 32u + half * 16u + lane;
                blk->qs[qi] |= (unsigned char)(qdist(rng) << shift);
            }
            blk->d = f32_to_f16_bits(0.035f);
            blk->dmin = f32_to_f16_bits(0.0f);
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
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / iters;
}

int main(int argc, char** argv) {
    unsigned M = 64;    // routed (token,slot) pairs for one expert bucket
    unsigned K = 2048;  // expert mid dim
    unsigned N = 4096;  // model/output dim
    int iters = 50;
    if (argc >= 4) {
        M = (unsigned)std::strtoul(argv[1], nullptr, 10);
        K = (unsigned)std::strtoul(argv[2], nullptr, 10);
        N = (unsigned)std::strtoul(argv[3], nullptr, 10);
    }
    if (argc >= 5) iters = std::atoi(argv[4]);
    if ((M % 16) || (K % 256) || (N % 16)) {
        std::fprintf(stderr, "usage: %s [M K N iters], with M%%16=0 K%%256=0 N%%16=0\n", argv[0]);
        return 2;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    std::printf("Q2_K down: M=%u pairs K=%u N=%u iters=%d\n", M, K, N, iters);

    const unsigned row_bytes = (K / QK_K) * Q2K_BYTES;
    std::vector<unsigned char> hW((uint64_t)N * row_bytes);
    std::vector<float> hMid((uint64_t)M * K);
    fill_q2_weights(hW, K, N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : hMid) x = dist(rng);

    unsigned char* dW = nullptr;
    half* dBhalf = nullptr;
    float *dMid = nullptr, *dCur = nullptr, *dWmma = nullptr, *dWmmaPacked = nullptr;
    HIP_CHECK(hipMalloc(&dW, hW.size()));
    HIP_CHECK(hipMalloc(&dBhalf, (uint64_t)K * N * sizeof(half)));
    HIP_CHECK(hipMalloc(&dMid, hMid.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dCur, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmma, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmmaPacked, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dMid, hMid.data(), hMid.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmma, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmmaPacked, 0, (uint64_t)M * N * sizeof(float)));

    auto launch_repack = [&]() {
        hipLaunchKernelGGL(q2_repack_half_kn_kernel, dim3((unsigned)(((uint64_t)K * N + 255u) / 256u)), dim3(256), 0, 0,
                           dBhalf, dW, K, N, row_bytes);
    };
    launch_repack();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    constexpr unsigned PAIR_TILE = 8;
    dim3 cur_block(32 * 16, 1, 1);
    dim3 cur_grid((N + 15) / 16, 1, 1);
    size_t cur_shmem = PAIR_TILE * 256 * sizeof(float);
    auto launch_cur = [&]() {
        hipLaunchKernelGGL((q2_down_current_like_kernel<PAIR_TILE>), cur_grid, cur_block, cur_shmem, 0,
                           dCur, dW, dMid, M, K, N, row_bytes);
    };

    dim3 wmma_block(256, 1, 1);
    dim3 wmma_grid(N / 16, M / 16, 1);
    size_t wmma_shmem = (16 * 16 + 16 * 16) * sizeof(half);
    auto launch_wmma = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dWmma, dW, dMid, M, K, N, row_bytes);
    };
    auto launch_wmma_packed = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_repacked_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dWmmaPacked, dBhalf, dMid, M, K, N);
    };

    launch_cur();
    HIP_CHECK(hipGetLastError());
    launch_wmma();
    HIP_CHECK(hipGetLastError());
    launch_wmma_packed();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur((uint64_t)M * N), hWmma((uint64_t)M * N), hWmmaPacked((uint64_t)M * N);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hWmma.data(), dWmma, hWmma.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hWmmaPacked.data(), dWmmaPacked, hWmmaPacked.size() * sizeof(float), hipMemcpyDeviceToHost));
    auto compare = [&](const char* name, const std::vector<float>& got) {
        double max_abs = 0.0, rms = 0.0, denom = 0.0;
        for (size_t i = 0; i < hCur.size(); ++i) {
            double e = (double)got[i] - (double)hCur[i];
            max_abs = std::max(max_abs, std::abs(e));
            rms += e * e;
            denom += (double)hCur[i] * (double)hCur[i];
        }
        rms = std::sqrt(rms / std::max<size_t>(1, hCur.size()));
        double rel_rms = std::sqrt(rms * rms * hCur.size() / std::max(1.0, denom));
        std::printf("%s_vs_current: max_abs=%.6g rms=%.6g rel_rms=%.6g\n", name, max_abs, rms, rel_rms);
    };
    compare("wmma_direct", hWmma);
    compare("wmma_packed", hWmmaPacked);

    float repack_ms = time_kernel(launch_repack, std::max(1, std::min(iters, 20)));
    float cur_ms = time_kernel(launch_cur, iters);
    float wmma_ms = time_kernel(launch_wmma, iters);
    float wmma_packed_ms = time_kernel(launch_wmma_packed, iters);
    double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("repack Q2_K->KxN half: %.4f ms  %.2f GiB/s output\n", repack_ms,
                ((double)K * N * sizeof(half) / 1073741824.0) / (repack_ms * 1.0e-3));
    std::printf("current-like:       %.4f ms  %.2f TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("wmma direct proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_ms, flops / (wmma_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_ms);
    std::printf("wmma packed proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_packed_ms, flops / (wmma_packed_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_packed_ms);

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dBhalf));
    HIP_CHECK(hipFree(dMid));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dWmma));
    HIP_CHECK(hipFree(dWmmaPacked));
    return 0;
}
