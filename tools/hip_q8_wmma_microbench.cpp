// Standalone dense Q8_0 rocWMMA microbench for DS4 MLA projection shapes.
// Build with: make hip-q8-wmma-bench
//
// Compares the current shared-X batched Q8_0 prefill kernel shape against two
// WMMA prototypes:
//   1) direct raw Q8_0 dequant into LDS half tiles
//   2) pre-repacked KxN half weights + WMMA
// This is intentionally outside the engine.

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

static constexpr unsigned Q8_BLOCK = 32;
static constexpr unsigned Q8_BYTES = 34;

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

__device__ static inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}

__device__ static inline float q8_0_value(const unsigned char* row, unsigned k) {
    const unsigned b = k >> 5;
    const unsigned lane = k & 31u;
    const unsigned char* blk = row + (uint64_t)b * Q8_BYTES;
    const unsigned short d_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
    return f16_bits_to_f32(d_bits) * (float)((const signed char*)(blk + 2))[lane];
}

template <unsigned TOK_TILE, unsigned BLOCKS_TILE>
__global__ void q8_current_sharedx_kernel(float* __restrict__ out,
                                          const unsigned char* __restrict__ w,
                                          const float* __restrict__ x,
                                          unsigned n_blocks,
                                          unsigned N,
                                          unsigned M,
                                          unsigned row_bytes) {
    extern __shared__ float shx[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned o = blockIdx.x * rows_per_block + wave;
    const unsigned t0 = blockIdx.y * TOK_TILE;
    if (t0 >= M) return;
    const bool row_valid = o < N;
    const unsigned char* row = w + (uint64_t)(row_valid ? o : 0u) * row_bytes;
    const unsigned K = n_blocks << 5;
    float acc[TOK_TILE];
#pragma unroll
    for (unsigned u = 0; u < TOK_TILE; ++u) acc[u] = 0.0f;
    for (unsigned b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const unsigned b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (unsigned j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const unsigned u = j / (BLOCKS_TILE * 32u);
            const unsigned r = j - u * (BLOCKS_TILE * 32u);
            const unsigned bb = r >> 5;
            const unsigned k = r & 31u;
            const unsigned t = t0 + u;
            shx[j] = (t < M && bb < b_count) ? x[(uint64_t)t * K + ((uint64_t)(b0 + bb) << 5) + k] : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (unsigned bb = 0; bb < b_count; ++bb) {
                const unsigned char* blk = row + (uint64_t)(b0 + bb) * Q8_BYTES;
                float d = 0.0f;
                if (lane == 0) {
                    const unsigned short d_bits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
                    d = f16_bits_to_f32(d_bits);
                }
                d = __shfl(d, 0, 32);
                const float wv = d * (float)((const signed char*)(blk + 2))[lane];
#pragma unroll
                for (unsigned u = 0; u < TOK_TILE; ++u) acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned u = 0; u < TOK_TILE; ++u) acc[u] = warp_reduce_sum(acc[u]);
    if (lane == 0 && row_valid) {
#pragma unroll
        for (unsigned u = 0; u < TOK_TILE; ++u) {
            const unsigned t = t0 + u;
            if (t < M) out[(uint64_t)t * N + o] = acc[u];
        }
    }
}

__global__ void q8_repack_half_kn_kernel(half* __restrict__ bhalf,
                                         const unsigned char* __restrict__ w,
                                         unsigned K, unsigned N,
                                         unsigned row_bytes) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)K * N;
    if (idx >= total) return;
    const unsigned k = (unsigned)(idx / N);
    const unsigned row = (unsigned)(idx - (uint64_t)k * N);
    bhalf[idx] = __float2half(q8_0_value(w + (uint64_t)row * row_bytes, k));
}

template <int BM=16, int BN=16, int BK=16>
__global__ void q8_wmma_direct_kernel(float* __restrict__ out,
                                      const unsigned char* __restrict__ w,
                                      const float* __restrict__ x,
                                      unsigned M, unsigned K, unsigned N,
                                      unsigned row_bytes) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
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
            shA[j] = __float2half(x[(uint64_t)(m0 + mm) * K + k0 + kk]);
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            shB[j] = __float2half(q8_0_value(w + (uint64_t)row * row_bytes, k0 + kk));
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
__global__ void q8_wmma_packed_kernel(float* __restrict__ out,
                                      const half* __restrict__ bhalf,
                                      const float* __restrict__ x,
                                      unsigned M, unsigned K, unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;
    half* shB = sh + BM * BK;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
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
            shA[j] = __float2half(x[(uint64_t)(m0 + mm) * K + k0 + kk]);
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

template <int TILES_N, int BM=16, int BN=16, int BK=16>
__global__ void q8_wmma_packed_multin_kernel(float* __restrict__ out,
                                             const half* __restrict__ bhalf,
                                             const float* __restrict__ x,
                                             unsigned M, unsigned K, unsigned N) {
    extern __shared__ half sh[];
    half* shA = sh;                         // BM x BK
    half* shB = sh + BM * BK;               // TILES_N x BK x BN
    const unsigned wave = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned tid = threadIdx.x;
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * (TILES_N * BN);
    const unsigned n_wave = n0 + wave * BN;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < BM * BK; j += blockDim.x) {
            const unsigned mm = j / BK;
            const unsigned kk = j - mm * BK;
            shA[j] = __float2half(x[(uint64_t)(m0 + mm) * K + k0 + kk]);
        }
        for (unsigned j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const unsigned tile = j / (BK * BN);
            const unsigned r = j - tile * (BK * BN);
            const unsigned kk = r / BN;
            const unsigned nn = r - kk * BN;
            shB[j] = bhalf[(uint64_t)(k0 + kk) * N + n0 + tile * BN + nn];
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (wave < TILES_N) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n_wave, acc, N, rocwmma::mem_row_major);
    (void)lane;
}

static void fill_q8_weights(std::vector<unsigned char>& w, unsigned K, unsigned N) {
    const unsigned n_blocks = K / Q8_BLOCK;
    const unsigned row_bytes = n_blocks * Q8_BYTES;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> qdist(-64, 63);
    for (unsigned row = 0; row < N; ++row) {
        for (unsigned b = 0; b < n_blocks; ++b) {
            unsigned char* blk = w.data() + (uint64_t)row * row_bytes + b * Q8_BYTES;
            const unsigned short d = f32_to_f16_bits(0.02f + 0.00001f * (float)((row + b) & 31u));
            blk[0] = (unsigned char)(d & 0xffu);
            blk[1] = (unsigned char)(d >> 8);
            signed char* q = (signed char*)(blk + 2);
            for (unsigned i = 0; i < 32; ++i) q[i] = (signed char)qdist(rng);
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

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [M K N iters]\n"
        "  M=tokens, K=input dim, N=output dim; require M%%16=0 K%%32=0 N%%16=0\n"
        "  examples:\n"
        "    %s 128 4096 8192 50   # attn_output_a-like\n"
        "    %s 128 8192 4096 50   # attn_output_b-like\n"
        "    %s 128 1024 32768 30  # attn_q_b-like\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char** argv) {
    unsigned M = 128, K = 4096, N = 8192;
    int iters = 50;
    if (argc >= 4) {
        M = (unsigned)std::strtoul(argv[1], nullptr, 10);
        K = (unsigned)std::strtoul(argv[2], nullptr, 10);
        N = (unsigned)std::strtoul(argv[3], nullptr, 10);
    }
    if (argc >= 5) iters = std::atoi(argv[4]);
    if ((M % 16) || (K % 32) || (N % 16)) {
        usage(argv[0]);
        return 2;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    std::printf("dense Q8_0: M=%u tokens K=%u N=%u iters=%d\n", M, K, N, iters);

    const unsigned n_blocks = K / Q8_BLOCK;
    const unsigned row_bytes = n_blocks * Q8_BYTES;
    std::vector<unsigned char> hW((uint64_t)N * row_bytes);
    std::vector<float> hX((uint64_t)M * K);
    fill_q8_weights(hW, K, N);
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    for (float& x : hX) x = xdist(rng);

    unsigned char* dW = nullptr;
    half* dBhalf = nullptr;
    float *dX = nullptr, *dCur = nullptr, *dDirect = nullptr, *dPacked = nullptr, *dPackedMulti = nullptr;
    HIP_CHECK(hipMalloc(&dW, hW.size()));
    HIP_CHECK(hipMalloc(&dBhalf, (uint64_t)K * N * sizeof(half)));
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dCur, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dDirect, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPacked, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dPackedMulti, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dDirect, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dPacked, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dPackedMulti, 0, (uint64_t)M * N * sizeof(float)));

    auto launch_repack = [&]() {
        hipLaunchKernelGGL(q8_repack_half_kn_kernel, dim3((unsigned)(((uint64_t)K * N + 255u) / 256u)), dim3(256), 0, 0,
                           dBhalf, dW, K, N, row_bytes);
    };
    launch_repack();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    constexpr unsigned TOK_TILE = 8;
    constexpr unsigned BLOCKS_TILE = 16;
    constexpr unsigned ROWS_PER_BLOCK = 32;
    dim3 cur_block(ROWS_PER_BLOCK * 32, 1, 1);
    dim3 cur_grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, (M + TOK_TILE - 1) / TOK_TILE, 1);
    size_t cur_shmem = TOK_TILE * BLOCKS_TILE * 32 * sizeof(float);
    auto launch_cur = [&]() {
        hipLaunchKernelGGL((q8_current_sharedx_kernel<TOK_TILE, BLOCKS_TILE>), cur_grid, cur_block, cur_shmem, 0,
                           dCur, dW, dX, n_blocks, N, M, row_bytes);
    };

    dim3 wmma_block(256, 1, 1);
    dim3 wmma_grid(N / 16, M / 16, 1);
    size_t wmma_shmem = (16 * 16 + 16 * 16) * sizeof(half);
    auto launch_direct = [&]() {
        hipLaunchKernelGGL((q8_wmma_direct_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dDirect, dW, dX, M, K, N, row_bytes);
    };
    auto launch_packed = [&]() {
        hipLaunchKernelGGL((q8_wmma_packed_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dPacked, dBhalf, dX, M, K, N);
    };
    constexpr int MULTI_N = 8;
    dim3 wmma_multi_grid(N / (16 * MULTI_N), M / 16, 1);
    size_t wmma_multi_shmem = (16 * 16 + MULTI_N * 16 * 16) * sizeof(half);
    auto launch_packed_multi = [&]() {
        hipLaunchKernelGGL((q8_wmma_packed_multin_kernel<MULTI_N,16,16,16>), wmma_multi_grid, wmma_block, wmma_multi_shmem, 0,
                           dPackedMulti, dBhalf, dX, M, K, N);
    };

    launch_cur(); HIP_CHECK(hipGetLastError());
    launch_direct(); HIP_CHECK(hipGetLastError());
    launch_packed(); HIP_CHECK(hipGetLastError());
    launch_packed_multi(); HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur((uint64_t)M * N), hDirect((uint64_t)M * N), hPacked((uint64_t)M * N), hPackedMulti((uint64_t)M * N);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hDirect.data(), dDirect, hDirect.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hPacked.data(), dPacked, hPacked.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hPackedMulti.data(), dPackedMulti, hPackedMulti.size() * sizeof(float), hipMemcpyDeviceToHost));
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
    compare("wmma_direct", hDirect);
    compare("wmma_packed", hPacked);
    compare("wmma_packed_multin", hPackedMulti);

    float repack_ms = time_kernel(launch_repack, std::max(1, std::min(iters, 20)));
    float cur_ms = time_kernel(launch_cur, iters);
    float direct_ms = time_kernel(launch_direct, iters);
    float packed_ms = time_kernel(launch_packed, iters);
    float packed_multi_ms = time_kernel(launch_packed_multi, iters);
    double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("repack Q8_0->KxN half: %.4f ms  %.2f GiB/s output\n", repack_ms,
                ((double)K * N * sizeof(half) / 1073741824.0) / (repack_ms * 1.0e-3));
    std::printf("current shared-X:   %.4f ms  %.2f TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("wmma direct proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", direct_ms, flops / (direct_ms * 1.0e-3) / 1.0e12, cur_ms / direct_ms);
    std::printf("wmma packed proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", packed_ms, flops / (packed_ms * 1.0e-3) / 1.0e12, cur_ms / packed_ms);
    std::printf("wmma packed multiN: %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", packed_multi_ms, flops / (packed_multi_ms * 1.0e-3) / 1.0e12, cur_ms / packed_multi_ms);

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dBhalf));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dDirect));
    HIP_CHECK(hipFree(dPacked));
    HIP_CHECK(hipFree(dPackedMulti));
    return 0;
}
