// Standalone Q2_K routed MoE down-projection rocWMMA microbench.
// Build with: make hip-q2-moe-wmma-bench
//
// This intentionally stays outside the engine. It compares a current-style
// wave-row Q2_K down kernel against:
//   - a Q8_K activation + int8 dot diagnostic, and
//   - rocWMMA prototypes that dequantize Q2_K weight tiles and float
//     activations into LDS FP16, then use FP16 x FP16 -> FP32 WMMA,
//     including multi-M-tile variants that dequantize each B tile once and
//     reuse it across 4 or 8 token/expert rows.
// It is a microbench, not an integrated path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma-version.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

struct block_q8_K_host {
    float d;
    signed char qs[QK_K];
    short bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K_host) == 292, "Q8_K block size");

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

__device__ static inline float silu_dev(float x) {
    return x / (1.0f + expf(-x));
}

__device__ static inline int warp_reduce_sum_int(int v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
    return v;
}

__device__ static inline char4 pack_q2_4_i8(const unsigned char* src, unsigned shift) {
    return make_char4((char)((src[0] >> shift) & 3u),
                      (char)((src[1] >> shift) & 3u),
                      (char)((src[2] >> shift) & 3u),
                      (char)((src[3] >> shift) & 3u));
}

__global__ void quantize_q8k_kernel(block_q8_K_host* __restrict__ out,
                                    const float* __restrict__ x,
                                    unsigned M, unsigned K) {
    __shared__ float sh_abs[QK_K];
    __shared__ float sh_val[QK_K];
    __shared__ signed char sh_qs[QK_K];
    __shared__ float sh_d;

    const unsigned row = blockIdx.x;
    const unsigned b = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned nb = K >> 8;
    if (row >= M || tid >= QK_K) return;
    const float xv = x[(uint64_t)row * K + (uint64_t)b * QK_K + tid];
    sh_abs[tid] = fabsf(xv);
    sh_val[tid] = xv;
    __syncthreads();

    for (unsigned stride = QK_K >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && sh_abs[tid + stride] > sh_abs[tid]) {
            sh_abs[tid] = sh_abs[tid + stride];
            sh_val[tid] = sh_val[tid + stride];
        }
        __syncthreads();
    }

    block_q8_K_host* y = out + (uint64_t)row * nb + b;
    if (tid == 0) {
        const float amax = sh_abs[0];
        const float maxv = sh_val[0];
        sh_d = (amax == 0.0f) ? 0.0f : (-maxv / 127.0f);
        y->d = sh_d;
    }
    __syncthreads();

    signed char q = 0;
    if (sh_d != 0.0f) {
        int v = (int)lrintf(xv / sh_d);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        q = (signed char)v;
    }
    sh_qs[tid] = q;
    y->qs[tid] = q;
    __syncthreads();

    if (tid < 16u) {
        int sum = 0;
#pragma unroll
        for (unsigned i = 0; i < 16u; ++i) sum += (int)sh_qs[tid * 16u + i];
        y->bsums[tid] = (short)sum;
    }
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

template <unsigned PAIR_TILE>
__global__ void q2_down_q8k_dot4_kernel(float* __restrict__ out,
                                        const unsigned char* __restrict__ w,
                                        const block_q8_K_host* __restrict__ midq,
                                        unsigned M, unsigned K, unsigned N,
                                        unsigned row_bytes) {
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
            const unsigned char* blk = wrow + (uint64_t)b * Q2K_BYTES;
            const unsigned char* sc = blk;
            const unsigned char* qs = blk + 16u;
            const float qd = f16_bits_to_f32((unsigned short)blk[80] | ((unsigned short)blk[81] << 8));
            const float qdmin = f16_bits_to_f32((unsigned short)blk[82] | ((unsigned short)blk[83] << 8));

            int isum[PAIR_TILE];
            int summs[PAIR_TILE];
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                isum[u] = 0;
                const bool valid = p0 + u < M;
                const block_q8_K_host* y = midq + (uint64_t)(p0 + u) * nb + b;
                int s = 0;
                if (valid && lane < 16u) s = (int)(sc[lane] >> 4) * (int)y->bsums[lane];
                summs[u] = warp_reduce_sum_int(s);
            }

            if (lane < 8u) {
                const unsigned lane4 = lane;
                const bool high_half = lane4 >= 4u;
                const unsigned half_off = high_half ? 16u : 0u;
                const unsigned lane_off = (lane4 & 3u) * 4u;
#pragma unroll
                for (unsigned chunk = 0; chunk < 2u; ++chunk) {
#pragma unroll
                    for (unsigned sg = 0; sg < 4u; ++sg) {
                        const unsigned shift = sg * 2u;
                        const unsigned scale_idx = chunk * 8u + sg * 2u + (high_half ? 1u : 0u);
                        const char4 q2pack = pack_q2_4_i8(qs + chunk * 32u + half_off + lane_off, shift);
                        const int scale = (int)(sc[scale_idx] & 0x0fu);
#pragma unroll
                        for (unsigned u = 0; u < PAIR_TILE; ++u) {
                            if (p0 + u < M) {
                                const block_q8_K_host* y = midq + (uint64_t)(p0 + u) * nb + b;
                                const signed char* q8 = y->qs + chunk * 128u + sg * 32u + lane4 * 4u;
                                const char4 q8pack = *reinterpret_cast<const char4*>(q8);
                                const int dot4 = amd_mixed_dot(q2pack, q8pack, 0, false);
                                isum[u] += dot4 * scale;
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                const int dot = warp_reduce_sum_int(isum[u]);
                if (lane == 0 && p0 + u < M) {
                    const block_q8_K_host* y = midq + (uint64_t)(p0 + u) * nb + b;
                    acc[u] += y->d * (qd * (float)dot - qdmin * (float)summs[u]);
                }
            }
        }

        if (lane == 0) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (p0 + u < M) out[(uint64_t)(p0 + u) * N + row] = acc[u];
            }
        }
    }
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

template <int MTILES=4, int BM=16, int BN=16, int BK=16>
__global__ void q2_down_wmma_multim_kernel(float* __restrict__ out,
                                           const unsigned char* __restrict__ w,
                                           const float* __restrict__ mid,
                                           unsigned M, unsigned K, unsigned N,
                                           unsigned row_bytes) {
    extern __shared__ half sh[];
    half* shA = sh;                       // MTILES x BM x BK row-major
    half* shB = sh + MTILES * BM * BK;    // BK x BN row-major, shared by all M waves

    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned row_m = m_group0 + mt * BM + mm;
            shA[j] = (row_m < M) ? __float2half(mid[(uint64_t)row_m * K + k0 + kk]) : __float2half(0.0f);
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
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (wave < MTILES) {
        const unsigned m0 = m_group0 + wave * BM;
        if (m0 + BM <= M) rocwmma::store_matrix_sync(out + (uint64_t)m0 * N + n0, acc, N, rocwmma::mem_row_major);
    }
}

template <unsigned PAIR_TILE>
__global__ void q2_gate_up_current_like_kernel(float* __restrict__ mid,
                                               const unsigned char* __restrict__ gate_w,
                                               const unsigned char* __restrict__ up_w,
                                               const float* __restrict__ x,
                                               unsigned M, unsigned K, unsigned N,
                                               unsigned row_bytes) {
    extern __shared__ float shx[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned row = blockIdx.x * rows_per_block + wave;
    if (row >= N) return;
    const unsigned nb = K >> 8;
    const unsigned char* grow = gate_w + (uint64_t)row * row_bytes;
    const unsigned char* urow = up_w + (uint64_t)row * row_bytes;
    for (unsigned p0 = 0; p0 < M; p0 += PAIR_TILE) {
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t xbase = (uint64_t)b * QK_K;
            for (unsigned j = tid; j < PAIR_TILE * QK_K; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shx[j] = (p0 + u < M) ? x[(uint64_t)(p0 + u) * K + xbase + k] : 0.0f;
            }
            __syncthreads();
            const unsigned char* gblk = grow + (uint64_t)b * Q2K_BYTES;
            const unsigned char* ublk = urow + (uint64_t)b * Q2K_BYTES;
#pragma unroll
            for (unsigned kk = 0; kk < 8u; ++kk) {
                const unsigned i = lane + (kk << 5);
                const float gwv = q2_k_dequant(gblk, i);
                const float uwv = q2_k_dequant(ublk, i);
#pragma unroll
                for (unsigned u = 0; u < PAIR_TILE; ++u) {
                    const float xv = shx[(u << 8) + i];
                    g_acc[u] += gwv * xv;
                    u_acc[u] += uwv * xv;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            g_acc[u] = warp_reduce_sum(g_acc[u]);
            u_acc[u] = warp_reduce_sum(u_acc[u]);
        }
        if (lane == 0) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (p0 + u < M) mid[(uint64_t)(p0 + u) * N + row] = silu_dev(g_acc[u]) * u_acc[u];
            }
        }
    }
}

template <int MTILES=4, int BM=16, int BN=16, int BK=16>
__global__ void q2_gate_up_wmma_multim_kernel(float* __restrict__ mid,
                                              const unsigned char* __restrict__ gate_w,
                                              const unsigned char* __restrict__ up_w,
                                              const float* __restrict__ x,
                                              unsigned M, unsigned K, unsigned N,
                                              unsigned row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half* shA = reinterpret_cast<half*>(raw_sh);                         // MTILES x BM x BK
    half* shBg = shA + MTILES * BM * BK;                                 // BK x BN
    half* shBu = shBg + BK * BN;                                         // BK x BN
    float* shCg = reinterpret_cast<float*>(shBu + BK * BN);              // MTILES x BM x BN
    float* shCu = shCg + MTILES * BM * BN;                               // MTILES x BM x BN

    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b bg;
    frag_b bu;
    frag_c accg;
    frag_c accu;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg, 0.0f);
        rocwmma::fill_fragment(accu, 0.0f);
    }

    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned row_m = m_group0 + mt * BM + mm;
            shA[j] = (row_m < M) ? __float2half(x[(uint64_t)row_m * K + k0 + kk]) : __float2half(0.0f);
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            const unsigned k = k0 + kk;
            const uint64_t off = (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
            shBg[j] = __float2half(q2_k_dequant(gate_w + off, k & 255u));
            shBu[j] = __float2half(q2_k_dequant(up_w + off, k & 255u));
        }
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg, shBg, BN);
            rocwmma::load_matrix_sync(bu, shBu, BN);
            rocwmma::mma_sync(accg, a, bg, accg);
            rocwmma::mma_sync(accu, a, bu, accu);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg + wave * BM * BN, accg, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu + wave * BM * BN, accu, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (unsigned j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const unsigned mt = j / (BM * BN);
        const unsigned rem = j - mt * BM * BN;
        const unsigned mm = rem / BN;
        const unsigned nn = rem - mm * BN;
        const unsigned row_m = m_group0 + mt * BM + mm;
        const unsigned row_n = n0 + nn;
        if (row_m < M && row_n < N) {
            const float g = shCg[j];
            const float u = shCu[j];
            mid[(uint64_t)row_m * N + row_n] = silu_dev(g) * u;
        }
    }
}

template <unsigned PAIR_TILE>
__global__ void q2_down_bucket_current_like_kernel(float* __restrict__ out,
                                                   const unsigned char* __restrict__ w,
                                                   const float* __restrict__ mid,
                                                   const int* __restrict__ buckets,
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
        int pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            pair[u] = (p0 + u < M) ? buckets[p0 + u] : -1;
            acc[u] = 0.0f;
        }
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t mbase = (uint64_t)b * QK_K;
            for (unsigned j = tid; j < PAIR_TILE * QK_K; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shmid[j] = (pair[u] >= 0) ? mid[(uint64_t)(unsigned)pair[u] * K + mbase + k] : 0.0f;
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
                if (pair[u] >= 0) out[(uint64_t)(unsigned)pair[u] * N + row] = acc[u];
            }
        }
    }
}

template <int MTILES=4, int BM=16, int BN=16, int BK=16>
__global__ void q2_down_bucket_wmma_multim_kernel(float* __restrict__ out,
                                                  const unsigned char* __restrict__ w,
                                                  const float* __restrict__ mid,
                                                  const int* __restrict__ buckets,
                                                  unsigned M, unsigned K, unsigned N,
                                                  unsigned row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half* shA = reinterpret_cast<half*>(raw_sh);
    half* shB = shA + MTILES * BM * BK;
    float* shC = reinterpret_cast<float*>(shB + BK * BN);
    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;
    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < M) {
                const unsigned pair = (unsigned)buckets[bucket_row];
                shA[j] = __float2half(mid[(uint64_t)pair * K + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
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
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (unsigned j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const unsigned mt = j / (BM * BN);
        const unsigned rem = j - mt * BM * BN;
        const unsigned mm = rem / BN;
        const unsigned nn = rem - mm * BN;
        const unsigned bucket_row = m_group0 + mt * BM + mm;
        const unsigned row_n = n0 + nn;
        if (bucket_row < M && row_n < N) {
            const unsigned pair = (unsigned)buckets[bucket_row];
            out[(uint64_t)pair * N + row_n] = shC[j];
        }
    }
}

template <unsigned PAIR_TILE>
__global__ void q2_gate_up_bucket_current_like_kernel(float* __restrict__ mid,
                                                      const unsigned char* __restrict__ gate_w,
                                                      const unsigned char* __restrict__ up_w,
                                                      const float* __restrict__ x,
                                                      const int* __restrict__ buckets,
                                                      unsigned M, unsigned K, unsigned N,
                                                      unsigned row_bytes) {
    extern __shared__ float shx[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned row = blockIdx.x * rows_per_block + wave;
    if (row >= N) return;
    const unsigned nb = K >> 8;
    const unsigned char* grow = gate_w + (uint64_t)row * row_bytes;
    const unsigned char* urow = up_w + (uint64_t)row * row_bytes;
    for (unsigned p0 = 0; p0 < M; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            pair[u] = (p0 + u < M) ? buckets[p0 + u] : -1;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t xbase = (uint64_t)b * QK_K;
            for (unsigned j = tid; j < PAIR_TILE * QK_K; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shx[j] = (pair[u] >= 0) ? x[(uint64_t)((unsigned)pair[u] / 6u) * K + xbase + k] : 0.0f;
            }
            __syncthreads();
            const unsigned char* gblk = grow + (uint64_t)b * Q2K_BYTES;
            const unsigned char* ublk = urow + (uint64_t)b * Q2K_BYTES;
#pragma unroll
            for (unsigned kk = 0; kk < 8u; ++kk) {
                const unsigned i = lane + (kk << 5);
                const float gwv = q2_k_dequant(gblk, i);
                const float uwv = q2_k_dequant(ublk, i);
#pragma unroll
                for (unsigned u = 0; u < PAIR_TILE; ++u) {
                    const float xv = shx[(u << 8) + i];
                    g_acc[u] += gwv * xv;
                    u_acc[u] += uwv * xv;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            g_acc[u] = warp_reduce_sum(g_acc[u]);
            u_acc[u] = warp_reduce_sum(u_acc[u]);
        }
        if (lane == 0) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (pair[u] >= 0) mid[(uint64_t)(unsigned)pair[u] * N + row] = silu_dev(g_acc[u]) * u_acc[u];
            }
        }
    }
}

template <int MTILES=4, int BM=16, int BN=16, int BK=16>
__global__ void q2_gate_up_bucket_wmma_multim_kernel(float* __restrict__ mid,
                                                     const unsigned char* __restrict__ gate_w,
                                                     const unsigned char* __restrict__ up_w,
                                                     const float* __restrict__ x,
                                                     const int* __restrict__ buckets,
                                                     unsigned M, unsigned K, unsigned N,
                                                     unsigned row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half* shA = reinterpret_cast<half*>(raw_sh);
    half* shBg = shA + MTILES * BM * BK;
    half* shBu = shBg + BK * BN;
    float* shCg = reinterpret_cast<float*>(shBu + BK * BN);
    float* shCu = shCg + MTILES * BM * BN;
    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;
    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg;
    frag_b bu;
    frag_c accg;
    frag_c accu;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg, 0.0f);
        rocwmma::fill_fragment(accu, 0.0f);
    }
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < M) {
                const unsigned token = (unsigned)buckets[bucket_row] / 6u;
                shA[j] = __float2half(x[(uint64_t)token * K + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            const unsigned k = k0 + kk;
            const uint64_t off = (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
            shBg[j] = __float2half(q2_k_dequant(gate_w + off, k & 255u));
            shBu[j] = __float2half(q2_k_dequant(up_w + off, k & 255u));
        }
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg, shBg, BN);
            rocwmma::load_matrix_sync(bu, shBu, BN);
            rocwmma::mma_sync(accg, a, bg, accg);
            rocwmma::mma_sync(accu, a, bu, accu);
        }
        __syncthreads();
    }
    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg + wave * BM * BN, accg, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu + wave * BM * BN, accu, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (unsigned j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const unsigned mt = j / (BM * BN);
        const unsigned rem = j - mt * BM * BN;
        const unsigned mm = rem / BN;
        const unsigned nn = rem - mm * BN;
        const unsigned bucket_row = m_group0 + mt * BM + mm;
        const unsigned row_n = n0 + nn;
        if (bucket_row < M && row_n < N) {
            const unsigned pair = (unsigned)buckets[bucket_row];
            mid[(uint64_t)pair * N + row_n] = silu_dev(shCg[j]) * shCu[j];
        }
    }
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

static uint64_t align_up_u64(uint64_t x, uint64_t a) {
    return a ? ((x + a - 1u) / a) * a : x;
}

class GgufView {
public:
    struct Tensor {
        std::string name;
        uint32_t ndim = 0;
        uint64_t dims[4] = {0, 0, 0, 0};
        uint32_t type = 0;
        uint64_t rel_offset = 0;
        uint64_t abs_offset = 0;
        uint64_t bytes = 0;
    };

    explicit GgufView(const char* path) {
        fd_ = open(path, O_RDONLY);
        if (fd_ < 0) throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
        struct stat st{};
        if (fstat(fd_, &st) != 0) throw std::runtime_error(std::string("stat failed: ") + std::strerror(errno));
        size_ = (uint64_t)st.st_size;
        map_ = (const unsigned char*)mmap(nullptr, (size_t)size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            map_ = nullptr;
            throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
        }
        parse();
    }

    ~GgufView() {
        if (map_) munmap((void*)map_, (size_t)size_);
        if (fd_ >= 0) close(fd_);
    }

    GgufView(const GgufView&) = delete;
    GgufView& operator=(const GgufView&) = delete;

    const Tensor* find(const std::string& name) const {
        for (const Tensor& t : tensors_) if (t.name == name) return &t;
        return nullptr;
    }

    const std::vector<Tensor>& tensors() const { return tensors_; }
    const unsigned char* data() const { return map_; }
    uint64_t size() const { return size_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t tensor_data_pos() const { return tensor_data_pos_; }

    static const char* type_name(uint32_t type) {
        switch (type) {
            case 0: return "F32";
            case 1: return "F16";
            case 2: return "Q4_0";
            case 3: return "Q4_1";
            case 6: return "Q5_0";
            case 7: return "Q5_1";
            case 8: return "Q8_0";
            case 9: return "Q8_1";
            case 10: return "Q2_K";
            case 11: return "Q3_K";
            case 12: return "Q4_K";
            case 13: return "Q5_K";
            case 14: return "Q6_K";
            case 15: return "Q8_K";
            case 16: return "IQ2_XXS";
            case 17: return "IQ2_XS";
            case 22: return "IQ2_S";
            case 26: return "I32";
            default: return "?";
        }
    }

private:
    uint32_t rd_u32(uint64_t& p) const {
        if (p + 4 > size_) throw std::runtime_error("truncated GGUF u32");
        uint32_t v;
        std::memcpy(&v, map_ + p, 4);
        p += 4;
        return v;
    }

    uint64_t rd_u64(uint64_t& p) const {
        if (p + 8 > size_) throw std::runtime_error("truncated GGUF u64");
        uint64_t v;
        std::memcpy(&v, map_ + p, 8);
        p += 8;
        return v;
    }

    std::string rd_string(uint64_t& p) const {
        const uint64_t n = rd_u64(p);
        if (n > size_ || p + n > size_) throw std::runtime_error("truncated GGUF string");
        std::string s((const char*)map_ + p, (size_t)n);
        p += n;
        return s;
    }

    static uint64_t scalar_size(uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: return 1;
            case 2: case 3: return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: return 0;
        }
    }

    void skip_value(uint64_t& p, uint32_t type) const {
        if (type == 8) {
            (void)rd_string(p);
            return;
        }
        if (type == 9) {
            const uint32_t elem_type = rd_u32(p);
            const uint64_t n = rd_u64(p);
            for (uint64_t i = 0; i < n; ++i) skip_value(p, elem_type);
            return;
        }
        const uint64_t sz = scalar_size(type);
        if (sz == 0) throw std::runtime_error("unsupported GGUF metadata type");
        if (p + sz > size_) throw std::runtime_error("truncated GGUF metadata value");
        p += sz;
    }

    static uint64_t tensor_nbytes(uint32_t type, uint64_t elems) {
        switch (type) {
            case 0: return elems * 4u;
            case 1: return elems * 2u;
            case 8: return ((elems + 31u) / 32u) * 34u;
            case 10: return ((elems + 255u) / 256u) * 84u;
            case 12: return ((elems + 255u) / 256u) * 144u;
            case 15: return ((elems + 255u) / 256u) * 292u;
            case 16: return ((elems + 255u) / 256u) * 66u;
            case 26: return elems * 4u;
            default: return 0;
        }
    }

    void parse() {
        uint64_t p = 0;
        const uint32_t magic = rd_u32(p);
        if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file");
        const uint32_t version = rd_u32(p);
        if (version != 3) throw std::runtime_error("only GGUF v3 is supported");
        const uint64_t n_tensors = rd_u64(p);
        const uint64_t n_kv = rd_u64(p);

        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = rd_string(p);
            const uint32_t type = rd_u32(p);
            const uint64_t value_pos = p;
            if (key == "general.alignment" && type == 4) {
                uint64_t q = value_pos;
                const uint32_t a = rd_u32(q);
                if (a != 0) alignment_ = a;
            }
            skip_value(p, type);
        }

        tensors_.reserve((size_t)n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            Tensor t;
            t.name = rd_string(p);
            t.ndim = rd_u32(p);
            if (t.ndim == 0 || t.ndim > 4) throw std::runtime_error("unsupported tensor rank");
            uint64_t elems = 1;
            for (uint32_t d = 0; d < t.ndim; ++d) {
                t.dims[d] = rd_u64(p);
                elems *= t.dims[d];
            }
            t.type = rd_u32(p);
            t.rel_offset = rd_u64(p);
            t.bytes = tensor_nbytes(t.type, elems);
            tensors_.push_back(std::move(t));
        }

        tensor_data_pos_ = align_up_u64(p, alignment_);
        for (Tensor& t : tensors_) {
            t.abs_offset = tensor_data_pos_ + t.rel_offset;
            if (t.bytes && (t.abs_offset > size_ || t.bytes > size_ - t.abs_offset)) {
                throw std::runtime_error("tensor payload points outside file");
            }
        }
    }

    int fd_ = -1;
    const unsigned char* map_ = nullptr;
    uint64_t size_ = 0;
    uint64_t alignment_ = 32;
    uint64_t tensor_data_pos_ = 0;
    std::vector<Tensor> tensors_;
};

static void print_gguf_tensor(const GgufView::Tensor& t) {
    std::printf("%-36s type=%-8s rank=%u dims=[", t.name.c_str(), GgufView::type_name(t.type), t.ndim);
    for (uint32_t i = 0; i < t.ndim; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)t.dims[i]);
    std::printf("] bytes=%.3f MiB abs=0x%llx\n", (double)t.bytes / 1048576.0, (unsigned long long)t.abs_offset);
}

static void fill_q2_weights(std::vector<unsigned char>& w, unsigned K, unsigned N) {
    const unsigned nb = K / QK_K;
    const unsigned row_bytes = nb * Q2K_BYTES;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> qdist(0, 3);
    std::uniform_int_distribution<int> sdist(1, 15);
    std::uniform_int_distribution<int> mdist(0, 15);
    for (unsigned row = 0; row < N; ++row) {
        for (unsigned b = 0; b < nb; ++b) {
            auto* blk = reinterpret_cast<block_q2_K_host*>(w.data() + (uint64_t)row * row_bytes + b * Q2K_BYTES);
            for (int g = 0; g < 16; ++g) blk->scales[g] = (unsigned char)((mdist(rng) << 4) | (sdist(rng) & 0x0f));
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
            blk->dmin = f32_to_f16_bits(0.012f);
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

static void compare_vectors(const char* name, const std::vector<float>& ref, const std::vector<float>& got) {
    double max_abs = 0.0, rms = 0.0, denom = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = (double)got[i] - (double)ref[i];
        max_abs = std::max(max_abs, std::abs(e));
        rms += e * e;
        denom += (double)ref[i] * (double)ref[i];
    }
    rms = std::sqrt(rms / std::max<size_t>(1, ref.size()));
    double rel_rms = std::sqrt(rms * rms * ref.size() / std::max(1.0, denom));
    std::printf("%s_vs_current: max_abs=%.6g rms=%.6g rel_rms=%.6g\n", name, max_abs, rms, rel_rms);
}

static std::vector<int> make_sorted_bucket(unsigned M, unsigned pair_stride) {
    if (M > pair_stride) throw std::runtime_error("bucket M exceeds pair_stride");
    std::vector<int> ids(pair_stride);
    for (unsigned i = 0; i < pair_stride; ++i) ids[i] = (int)i;
    std::mt19937 rng(1234);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(M);
    std::sort(ids.begin(), ids.end());
    return ids;
}

static void compare_bucket_vectors(const char* name,
                                   const std::vector<float>& ref,
                                   const std::vector<float>& got,
                                   const std::vector<int>& buckets,
                                   unsigned N) {
    double max_abs = 0.0, rms = 0.0, denom = 0.0;
    uint64_t n = 0;
    for (int pair : buckets) {
        const uint64_t base = (uint64_t)(unsigned)pair * N;
        for (unsigned j = 0; j < N; ++j) {
            const double e = (double)got[base + j] - (double)ref[base + j];
            max_abs = std::max(max_abs, std::abs(e));
            rms += e * e;
            denom += (double)ref[base + j] * (double)ref[base + j];
            ++n;
        }
    }
    rms = std::sqrt(rms / std::max<uint64_t>(1, n));
    double rel_rms = std::sqrt(rms * rms * (double)n / std::max(1.0, denom));
    std::printf("%s_vs_current: max_abs=%.6g rms=%.6g rel_rms=%.6g\n", name, max_abs, rms, rel_rms);
}

static int run_down_bucket_bench(const std::vector<unsigned char>& hW,
                                 unsigned M, unsigned K, unsigned N,
                                 unsigned pair_stride,
                                 int iters,
                                 const std::string& label) {
    const unsigned row_bytes = (K / QK_K) * Q2K_BYTES;
    std::vector<int> hBuckets = make_sorted_bucket(M, pair_stride);
    std::vector<float> hMid((uint64_t)pair_stride * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : hMid) x = dist(rng);

    unsigned char* dW = nullptr;
    float *dMid = nullptr, *dCur = nullptr, *dM2 = nullptr, *dM4 = nullptr, *dM8 = nullptr;
    int* dBuckets = nullptr;
    const uint64_t out_elems = (uint64_t)pair_stride * N;
    HIP_CHECK(hipMalloc(&dW, hW.size()));
    HIP_CHECK(hipMalloc(&dMid, hMid.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dBuckets, hBuckets.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dCur, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM2, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM4, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM8, out_elems * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dMid, hMid.data(), hMid.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBuckets, hBuckets.data(), hBuckets.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM2, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM4, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM8, 0, out_elems * sizeof(float)));

    constexpr unsigned PAIR_TILE = 8;
    dim3 cur_block(32 * 16, 1, 1);
    dim3 cur_grid((N + 15) / 16, 1, 1);
    size_t cur_shmem = PAIR_TILE * QK_K * sizeof(float);
    auto launch_cur = [&]() {
        hipLaunchKernelGGL((q2_down_bucket_current_like_kernel<PAIR_TILE>), cur_grid, cur_block, cur_shmem, 0,
                           dCur, dW, dMid, dBuckets, M, K, N, row_bytes);
    };
    dim3 m2_block(32 * 2, 1, 1);
    dim3 m2_grid(N / 16, (M + 2 * 16 - 1) / (2 * 16), 1);
    size_t m2_shmem = (2 * 16 * 16 + 16 * 16) * sizeof(half) + (2 * 16 * 16) * sizeof(float);
    auto launch_m2 = [&]() {
        hipLaunchKernelGGL((q2_down_bucket_wmma_multim_kernel<2,16,16,16>), m2_grid, m2_block, m2_shmem, 0,
                           dM2, dW, dMid, dBuckets, M, K, N, row_bytes);
    };
    dim3 m4_block(32 * 4, 1, 1);
    dim3 m4_grid(N / 16, (M + 4 * 16 - 1) / (4 * 16), 1);
    size_t m4_shmem = (4 * 16 * 16 + 16 * 16) * sizeof(half) + (4 * 16 * 16) * sizeof(float);
    auto launch_m4 = [&]() {
        hipLaunchKernelGGL((q2_down_bucket_wmma_multim_kernel<4,16,16,16>), m4_grid, m4_block, m4_shmem, 0,
                           dM4, dW, dMid, dBuckets, M, K, N, row_bytes);
    };
    dim3 m8_block(32 * 8, 1, 1);
    dim3 m8_grid(N / 16, (M + 8 * 16 - 1) / (8 * 16), 1);
    size_t m8_shmem = (8 * 16 * 16 + 16 * 16) * sizeof(half) + (8 * 16 * 16) * sizeof(float);
    auto launch_m8 = [&]() {
        hipLaunchKernelGGL((q2_down_bucket_wmma_multim_kernel<8,16,16,16>), m8_grid, m8_block, m8_shmem, 0,
                           dM8, dW, dMid, dBuckets, M, K, N, row_bytes);
    };

    launch_cur(); HIP_CHECK(hipGetLastError());
    launch_m2();  HIP_CHECK(hipGetLastError());
    launch_m4();  HIP_CHECK(hipGetLastError());
    launch_m8();  HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur(out_elems), hM2(out_elems), hM4(out_elems), hM8(out_elems);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM2.data(), dM2, hM2.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM4.data(), dM4, hM4.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM8.data(), dM8, hM8.size() * sizeof(float), hipMemcpyDeviceToHost));
    compare_bucket_vectors("bucket_down_wmma_multi2", hCur, hM2, hBuckets, N);
    compare_bucket_vectors("bucket_down_wmma_multi4", hCur, hM4, hBuckets, N);
    compare_bucket_vectors("bucket_down_wmma_multi8", hCur, hM8, hBuckets, N);

    const float cur_ms = time_kernel(launch_cur, iters);
    const float m2_ms = time_kernel(launch_m2, iters);
    const float m4_ms = time_kernel(launch_m4, iters);
    const float m8_ms = time_kernel(launch_m8, iters);
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("Q2_K bucket down slice: %s\n", label.c_str());
    std::printf("Q2_K bucket down shape: M=%u pair_stride=%u K=%u N=%u iters=%d\n", M, pair_stride, K, N, iters);
    std::printf("bucket down current-like: %.4f ms  %.2f logical TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("bucket down multi2:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m2_ms, flops / (m2_ms * 1.0e-3) / 1.0e12, cur_ms / m2_ms);
    std::printf("bucket down multi4:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m4_ms, flops / (m4_ms * 1.0e-3) / 1.0e12, cur_ms / m4_ms);
    std::printf("bucket down multi8:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m8_ms, flops / (m8_ms * 1.0e-3) / 1.0e12, cur_ms / m8_ms);

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dMid));
    HIP_CHECK(hipFree(dBuckets));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dM2));
    HIP_CHECK(hipFree(dM4));
    HIP_CHECK(hipFree(dM8));
    return 0;
}

static int run_gate_up_bucket_bench(const std::vector<unsigned char>& hGate,
                                    const std::vector<unsigned char>& hUp,
                                    unsigned M, unsigned K, unsigned N,
                                    unsigned pair_stride,
                                    int iters,
                                    const std::string& label) {
    const unsigned row_bytes = (K / QK_K) * Q2K_BYTES;
    if (pair_stride % 6u != 0u) throw std::runtime_error("pair_stride must be a multiple of 6");
    const unsigned n_tokens = pair_stride / 6u;
    std::vector<int> hBuckets = make_sorted_bucket(M, pair_stride);
    std::vector<float> hX((uint64_t)n_tokens * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : hX) x = dist(rng);

    unsigned char *dGate = nullptr, *dUp = nullptr;
    float *dX = nullptr, *dCur = nullptr, *dM2 = nullptr, *dM4 = nullptr, *dM8 = nullptr;
    int* dBuckets = nullptr;
    const uint64_t out_elems = (uint64_t)pair_stride * N;
    HIP_CHECK(hipMalloc(&dGate, hGate.size()));
    HIP_CHECK(hipMalloc(&dUp, hUp.size()));
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dBuckets, hBuckets.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dCur, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM2, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM4, out_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM8, out_elems * sizeof(float)));
    HIP_CHECK(hipMemcpy(dGate, hGate.data(), hGate.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dUp, hUp.data(), hUp.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBuckets, hBuckets.data(), hBuckets.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM2, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM4, 0, out_elems * sizeof(float)));
    HIP_CHECK(hipMemset(dM8, 0, out_elems * sizeof(float)));

    constexpr unsigned PAIR_TILE = 8;
    dim3 cur_block(32 * 16, 1, 1);
    dim3 cur_grid((N + 15) / 16, 1, 1);
    size_t cur_shmem = PAIR_TILE * QK_K * sizeof(float);
    auto launch_cur = [&]() {
        hipLaunchKernelGGL((q2_gate_up_bucket_current_like_kernel<PAIR_TILE>), cur_grid, cur_block, cur_shmem, 0,
                           dCur, dGate, dUp, dX, dBuckets, M, K, N, row_bytes);
    };
    dim3 m2_block(32 * 2, 1, 1);
    dim3 m2_grid(N / 16, (M + 2 * 16 - 1) / (2 * 16), 1);
    size_t m2_shmem = (2 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 2 * 16 * 16) * sizeof(float);
    auto launch_m2 = [&]() {
        hipLaunchKernelGGL((q2_gate_up_bucket_wmma_multim_kernel<2,16,16,16>), m2_grid, m2_block, m2_shmem, 0,
                           dM2, dGate, dUp, dX, dBuckets, M, K, N, row_bytes);
    };
    dim3 m4_block(32 * 4, 1, 1);
    dim3 m4_grid(N / 16, (M + 4 * 16 - 1) / (4 * 16), 1);
    size_t m4_shmem = (4 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 4 * 16 * 16) * sizeof(float);
    auto launch_m4 = [&]() {
        hipLaunchKernelGGL((q2_gate_up_bucket_wmma_multim_kernel<4,16,16,16>), m4_grid, m4_block, m4_shmem, 0,
                           dM4, dGate, dUp, dX, dBuckets, M, K, N, row_bytes);
    };
    dim3 m8_block(32 * 8, 1, 1);
    dim3 m8_grid(N / 16, (M + 8 * 16 - 1) / (8 * 16), 1);
    size_t m8_shmem = (8 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 8 * 16 * 16) * sizeof(float);
    auto launch_m8 = [&]() {
        hipLaunchKernelGGL((q2_gate_up_bucket_wmma_multim_kernel<8,16,16,16>), m8_grid, m8_block, m8_shmem, 0,
                           dM8, dGate, dUp, dX, dBuckets, M, K, N, row_bytes);
    };

    launch_cur(); HIP_CHECK(hipGetLastError());
    launch_m2();  HIP_CHECK(hipGetLastError());
    launch_m4();  HIP_CHECK(hipGetLastError());
    launch_m8();  HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur(out_elems), hM2(out_elems), hM4(out_elems), hM8(out_elems);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM2.data(), dM2, hM2.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM4.data(), dM4, hM4.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM8.data(), dM8, hM8.size() * sizeof(float), hipMemcpyDeviceToHost));
    compare_bucket_vectors("bucket_gate_up_wmma_multi2", hCur, hM2, hBuckets, N);
    compare_bucket_vectors("bucket_gate_up_wmma_multi4", hCur, hM4, hBuckets, N);
    compare_bucket_vectors("bucket_gate_up_wmma_multi8", hCur, hM8, hBuckets, N);

    const float cur_ms = time_kernel(launch_cur, iters);
    const float m2_ms = time_kernel(launch_m2, iters);
    const float m4_ms = time_kernel(launch_m4, iters);
    const float m8_ms = time_kernel(launch_m8, iters);
    const double flops = 4.0 * (double)M * (double)N * (double)K;
    std::printf("Q2_K bucket gate+up slice: %s\n", label.c_str());
    std::printf("Q2_K bucket gate+up shape: M=%u pair_stride=%u K=%u N=%u iters=%d\n", M, pair_stride, K, N, iters);
    std::printf("bucket gate_up current-like: %.4f ms  %.2f logical TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("bucket gate_up multi2:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m2_ms, flops / (m2_ms * 1.0e-3) / 1.0e12, cur_ms / m2_ms);
    std::printf("bucket gate_up multi4:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m4_ms, flops / (m4_ms * 1.0e-3) / 1.0e12, cur_ms / m4_ms);
    std::printf("bucket gate_up multi8:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m8_ms, flops / (m8_ms * 1.0e-3) / 1.0e12, cur_ms / m8_ms);

    HIP_CHECK(hipFree(dGate));
    HIP_CHECK(hipFree(dUp));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dBuckets));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dM2));
    HIP_CHECK(hipFree(dM4));
    HIP_CHECK(hipFree(dM8));
    return 0;
}

static int run_gate_up_bench(const std::vector<unsigned char>& hGate,
                             const std::vector<unsigned char>& hUp,
                             unsigned M, unsigned K, unsigned N,
                             int iters,
                             const std::string& label) {
    const unsigned row_bytes = (K / QK_K) * Q2K_BYTES;
    std::vector<float> hX((uint64_t)M * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : hX) x = dist(rng);

    unsigned char *dGate = nullptr, *dUp = nullptr;
    float *dX = nullptr, *dCur = nullptr, *dM4 = nullptr, *dM8 = nullptr;
    HIP_CHECK(hipMalloc(&dGate, hGate.size()));
    HIP_CHECK(hipMalloc(&dUp, hUp.size()));
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dCur, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM4, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dM8, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dGate, hGate.data(), hGate.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dUp, hUp.data(), hUp.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dM4, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dM8, 0, (uint64_t)M * N * sizeof(float)));

    constexpr unsigned PAIR_TILE = 8;
    dim3 cur_block(32 * 16, 1, 1);
    dim3 cur_grid((N + 15) / 16, 1, 1);
    size_t cur_shmem = PAIR_TILE * QK_K * sizeof(float);
    auto launch_cur = [&]() {
        hipLaunchKernelGGL((q2_gate_up_current_like_kernel<PAIR_TILE>), cur_grid, cur_block, cur_shmem, 0,
                           dCur, dGate, dUp, dX, M, K, N, row_bytes);
    };

    dim3 m4_block(32 * 4, 1, 1);
    dim3 m4_grid(N / 16, (M + 4 * 16 - 1) / (4 * 16), 1);
    size_t m4_shmem = (4 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 4 * 16 * 16) * sizeof(float);
    auto launch_m4 = [&]() {
        hipLaunchKernelGGL((q2_gate_up_wmma_multim_kernel<4,16,16,16>), m4_grid, m4_block, m4_shmem, 0,
                           dM4, dGate, dUp, dX, M, K, N, row_bytes);
    };

    dim3 m8_block(32 * 8, 1, 1);
    dim3 m8_grid(N / 16, (M + 8 * 16 - 1) / (8 * 16), 1);
    size_t m8_shmem = (8 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 8 * 16 * 16) * sizeof(float);
    auto launch_m8 = [&]() {
        hipLaunchKernelGGL((q2_gate_up_wmma_multim_kernel<8,16,16,16>), m8_grid, m8_block, m8_shmem, 0,
                           dM8, dGate, dUp, dX, M, K, N, row_bytes);
    };

    launch_cur(); HIP_CHECK(hipGetLastError());
    launch_m4();  HIP_CHECK(hipGetLastError());
    launch_m8();  HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur((uint64_t)M * N), hM4((uint64_t)M * N), hM8((uint64_t)M * N);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM4.data(), dM4, hM4.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hM8.data(), dM8, hM8.size() * sizeof(float), hipMemcpyDeviceToHost));
    compare_vectors("gate_up_wmma_multi4", hCur, hM4);
    compare_vectors("gate_up_wmma_multi8", hCur, hM8);

    const float cur_ms = time_kernel(launch_cur, iters);
    const float m4_ms = time_kernel(launch_m4, iters);
    const float m8_ms = time_kernel(launch_m8, iters);
    const double flops = 4.0 * (double)M * (double)N * (double)K; // gate and up matmuls
    std::printf("Q2_K gate+up slice: %s\n", label.c_str());
    std::printf("Q2_K gate+up shape: M=%u pairs K=%u N=%u iters=%d\n", M, K, N, iters);
    std::printf("gate_up current-like: %.4f ms  %.2f logical TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("gate_up multi4:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m4_ms, flops / (m4_ms * 1.0e-3) / 1.0e12, cur_ms / m4_ms);
    std::printf("gate_up multi8:       %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", m8_ms, flops / (m8_ms * 1.0e-3) / 1.0e12, cur_ms / m8_ms);

    HIP_CHECK(hipFree(dGate));
    HIP_CHECK(hipFree(dUp));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dM4));
    HIP_CHECK(hipFree(dM8));
    return 0;
}

template <unsigned PAIR_TILE>
__global__ void q2_gate_up_all_current_sharedx_kernel(float* __restrict__ mid,
                                                       const unsigned char* __restrict__ gate_w,
                                                       const unsigned char* __restrict__ up_w,
                                                       const float* __restrict__ x,
                                                       const int* __restrict__ counts,
                                                       const int* __restrict__ buckets,
                                                       unsigned stride,
                                                       unsigned min_count,
                                                       unsigned max_count,
                                                       unsigned K,
                                                       unsigned N,
                                                       unsigned row_bytes,
                                                       uint64_t expert_bytes) {
    extern __shared__ float shx[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned row = blockIdx.x * rows_per_block + wave;
    const unsigned expert = blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < N;
    const unsigned count = (unsigned)counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char* grow = gate_w + (uint64_t)expert * expert_bytes + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const unsigned char* urow = up_w + (uint64_t)expert * expert_bytes + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const unsigned nb = K >> 8;
    for (unsigned p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t xbase = (uint64_t)b * QK_K;
            for (unsigned j = tid; j < PAIR_TILE * QK_K; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shx[j] = (pair[u] >= 0) ? x[(uint64_t)((unsigned)pair[u] / 6u) * K + xbase + k] : 0.0f;
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char* gblk = grow + (uint64_t)b * Q2K_BYTES;
                const unsigned char* ublk = urow + (uint64_t)b * Q2K_BYTES;
#pragma unroll
                for (unsigned kk = 0; kk < 8u; ++kk) {
                    const unsigned i = lane + (kk << 5);
                    const float gwv = q2_k_dequant(gblk, i);
                    const float uwv = q2_k_dequant(ublk, i);
#pragma unroll
                    for (unsigned u = 0; u < PAIR_TILE; ++u) {
                        const float xv = shx[(u << 8) + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            g_acc[u] = warp_reduce_sum(g_acc[u]);
            u_acc[u] = warp_reduce_sum(u_acc[u]);
        }
        if (lane == 0 && row_valid) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (pair[u] >= 0) mid[(uint64_t)(unsigned)pair[u] * N + row] = silu_dev(g_acc[u]) * u_acc[u];
            }
        }
    }
}


template <unsigned PAIR_TILE>
__global__ void q2_down_all_current_sharedmid_kernel(float* __restrict__ out,
                                                     const unsigned char* __restrict__ w,
                                                     const float* __restrict__ mid,
                                                     const int* __restrict__ counts,
                                                     const int* __restrict__ buckets,
                                                     unsigned stride,
                                                     unsigned min_count,
                                                     unsigned max_count,
                                                     unsigned K,
                                                     unsigned N,
                                                     unsigned row_bytes,
                                                     uint64_t expert_bytes) {
    extern __shared__ float shmid[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wave = tid >> 5;
    const unsigned rows_per_block = blockDim.x >> 5;
    const unsigned row = blockIdx.x * rows_per_block + wave;
    const unsigned expert = blockIdx.y;
    if (expert >= 256u) return;
    const bool row_valid = row < N;
    const unsigned count = (unsigned)counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const unsigned char* wrow = w + (uint64_t)expert * expert_bytes + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const unsigned nb = K >> 8;
    for (unsigned p0 = 0; p0 < count; p0 += PAIR_TILE) {
        int pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) {
            pair[u] = (p0 + u < count) ? buckets[(uint64_t)expert * stride + p0 + u] : -1;
            acc[u] = 0.0f;
        }
        for (unsigned b = 0; b < nb; ++b) {
            const uint64_t mbase = (uint64_t)b * QK_K;
            for (unsigned j = tid; j < PAIR_TILE * QK_K; j += blockDim.x) {
                const unsigned u = j >> 8;
                const unsigned k = j & 255u;
                shmid[j] = (pair[u] >= 0) ? mid[(uint64_t)(unsigned)pair[u] * K + mbase + k] : 0.0f;
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char* blk = wrow + (uint64_t)b * Q2K_BYTES;
#pragma unroll
                for (unsigned kk = 0; kk < 8u; ++kk) {
                    const unsigned i = lane + (kk << 5);
                    const float wv = q2_k_dequant(blk, i);
#pragma unroll
                    for (unsigned u = 0; u < PAIR_TILE; ++u) acc[u] += wv * shmid[(u << 8) + i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned u = 0; u < PAIR_TILE; ++u) acc[u] = warp_reduce_sum(acc[u]);
        if (lane == 0 && row_valid) {
#pragma unroll
            for (unsigned u = 0; u < PAIR_TILE; ++u) {
                if (pair[u] >= 0) out[(uint64_t)(unsigned)pair[u] * N + row] = acc[u];
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16>
__global__ void q2_gate_up_hotlist_wmma_multim_kernel(float* __restrict__ mid,
                                                       const unsigned char* __restrict__ gate_w,
                                                       const unsigned char* __restrict__ up_w,
                                                       const float* __restrict__ x,
                                                       const int* __restrict__ counts,
                                                       const int* __restrict__ buckets,
                                                       const int* __restrict__ hot_experts,
                                                       unsigned stride,
                                                       unsigned K,
                                                       unsigned N,
                                                       unsigned row_bytes,
                                                       uint64_t expert_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half* shA = reinterpret_cast<half*>(raw_sh);
    half* shBg = shA + MTILES * BM * BK;
    half* shBu = shBg + BK * BN;
    float* shCg = reinterpret_cast<float*>(shBu + BK * BN);
    float* shCu = shCg + MTILES * BM * BN;
    const unsigned expert = (unsigned)hot_experts[blockIdx.z];
    const unsigned count = (unsigned)counts[expert];
    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    if (m_group0 >= count) return;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;
    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg;
    frag_b bu;
    frag_c accg;
    frag_c accu;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg, 0.0f);
        rocwmma::fill_fragment(accu, 0.0f);
    }
    const unsigned char* gew = gate_w + (uint64_t)expert * expert_bytes;
    const unsigned char* uew = up_w + (uint64_t)expert * expert_bytes;
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < count) {
                const unsigned pair = (unsigned)buckets[(uint64_t)expert * stride + bucket_row];
                const unsigned token = pair / 6u;
                shA[j] = __float2half(x[(uint64_t)token * K + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            const unsigned k = k0 + kk;
            const uint64_t off = (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
            shBg[j] = __float2half(q2_k_dequant(gew + off, k & 255u));
            shBu[j] = __float2half(q2_k_dequant(uew + off, k & 255u));
        }
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg, shBg, BN);
            rocwmma::load_matrix_sync(bu, shBu, BN);
            rocwmma::mma_sync(accg, a, bg, accg);
            rocwmma::mma_sync(accu, a, bu, accu);
        }
        __syncthreads();
    }
    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg + wave * BM * BN, accg, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu + wave * BM * BN, accu, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (unsigned j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const unsigned mt = j / (BM * BN);
        const unsigned rem = j - mt * BM * BN;
        const unsigned mm = rem / BN;
        const unsigned nn = rem - mm * BN;
        const unsigned bucket_row = m_group0 + mt * BM + mm;
        const unsigned row_n = n0 + nn;
        if (bucket_row < count && row_n < N) {
            const unsigned pair = (unsigned)buckets[(uint64_t)expert * stride + bucket_row];
            mid[(uint64_t)pair * N + row_n] = silu_dev(shCg[j]) * shCu[j];
        }
    }
}


template <int MTILES=8, int BM=16, int BN=16, int BK=16>
__global__ void q2_down_hotlist_wmma_multim_kernel(float* __restrict__ out,
                                                   const unsigned char* __restrict__ w,
                                                   const float* __restrict__ mid,
                                                   const int* __restrict__ counts,
                                                   const int* __restrict__ buckets,
                                                   const int* __restrict__ hot_experts,
                                                   unsigned stride,
                                                   unsigned K,
                                                   unsigned N,
                                                   unsigned row_bytes,
                                                   uint64_t expert_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half* shA = reinterpret_cast<half*>(raw_sh);
    half* shB = shA + MTILES * BM * BK;
    float* shC = reinterpret_cast<float*>(shB + BK * BN);
    const unsigned expert = (unsigned)hot_experts[blockIdx.z];
    const unsigned count = (unsigned)counts[expert];
    const unsigned tile_m_group = blockIdx.y;
    const unsigned tile_n = blockIdx.x;
    const unsigned m_group0 = tile_m_group * MTILES * BM;
    if (m_group0 >= count) return;
    const unsigned n0 = tile_n * BN;
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid >> 5;
    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);
    const unsigned char* ew = w + (uint64_t)expert * expert_bytes;
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        for (unsigned j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const unsigned mt = j / (BM * BK);
            const unsigned rem = j - mt * BM * BK;
            const unsigned mm = rem / BK;
            const unsigned kk = rem - mm * BK;
            const unsigned bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < count) {
                const unsigned pair = (unsigned)buckets[(uint64_t)expert * stride + bucket_row];
                shA[j] = __float2half(mid[(uint64_t)pair * K + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        for (unsigned j = tid; j < BK * BN; j += blockDim.x) {
            const unsigned kk = j / BN;
            const unsigned nn = j - kk * BN;
            const unsigned row = n0 + nn;
            const unsigned k = k0 + kk;
            const unsigned char* blk = ew + (uint64_t)row * row_bytes + (uint64_t)(k >> 8) * Q2K_BYTES;
            shB[j] = __float2half(q2_k_dequant(blk, k & 255u));
        }
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (wave < MTILES) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (unsigned j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const unsigned mt = j / (BM * BN);
        const unsigned rem = j - mt * BM * BN;
        const unsigned mm = rem / BN;
        const unsigned nn = rem - mm * BN;
        const unsigned bucket_row = m_group0 + mt * BM + mm;
        const unsigned row_n = n0 + nn;
        if (bucket_row < count && row_n < N) {
            const unsigned pair = (unsigned)buckets[(uint64_t)expert * stride + bucket_row];
            out[(uint64_t)pair * N + row_n] = shC[j];
        }
    }
}

struct ReplayRoutingDump {
    unsigned idx = 0;
    unsigned tokens = 0;
    std::vector<unsigned> counts;
};

static std::vector<ReplayRoutingDump> parse_routing_dumps_file(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("open routing log failed: ") + path);
    std::vector<ReplayRoutingDump> dumps;
    ReplayRoutingDump cur;
    bool have = false;
    std::string line;
    while (std::getline(in, line)) {
        const size_t dump_pos = line.find("HIP MoE routing dump #");
        if (dump_pos != std::string::npos) {
            if (have && cur.counts.size() == 256) dumps.push_back(cur);
            cur = ReplayRoutingDump{};
            have = true;
            const char* s = line.c_str() + dump_pos;
            (void)std::sscanf(s, "HIP MoE routing dump #%u tokens=%u", &cur.idx, &cur.tokens);
            continue;
        }
        if (!have) continue;
        const size_t epos = line.find("ds4:   e");
        if (epos == std::string::npos) continue;
        const size_t colon = line.find(':', epos + 7);
        if (colon == std::string::npos) continue;
        std::istringstream iss(line.substr(colon + 1));
        unsigned v = 0;
        while (iss >> v) cur.counts.push_back(v);
        if (cur.counts.size() == 256) {
            dumps.push_back(cur);
            cur = ReplayRoutingDump{};
            have = false;
        }
    }
    if (have && cur.counts.size() == 256) dumps.push_back(cur);
    return dumps;
}

static std::string replay_tensor_name(unsigned layer, const char* suffix) {
    return std::string("blk.") + std::to_string(layer) + suffix;
}

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

static void compare_device_vectors(const char* name, const float* dRef, const float* dGot, uint64_t elems) {
    std::vector<float> hRef(elems), hGot(elems);
    HIP_CHECK(hipMemcpy(hRef.data(), dRef, elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hGot.data(), dGot, elems * sizeof(float), hipMemcpyDeviceToHost));
    compare_vectors(name, hRef, hGot);
}

static int run_replay_routing_layer(const char* model_path,
                                    unsigned layer,
                                    const char* routing_path,
                                    unsigned dump_index,
                                    int iters,
                                    unsigned gate_hot_threshold,
                                    unsigned down_hot_threshold) {
    if (iters <= 0) iters = 1;
    std::vector<ReplayRoutingDump> dumps = parse_routing_dumps_file(routing_path);
    if (dumps.empty()) throw std::runtime_error("no complete routing dumps found");
    const ReplayRoutingDump* chosen = nullptr;
    for (const ReplayRoutingDump& d : dumps) {
        if (d.idx == dump_index) { chosen = &d; break; }
    }
    if (!chosen && dump_index > 0 && dump_index <= dumps.size()) chosen = &dumps[dump_index - 1];
    if (!chosen) throw std::runtime_error("requested routing dump not found");

    uint64_t pair_stride64 = 0;
    unsigned active = 0, hot_gate = 0, hot_down = 0;
    uint64_t gate_hot_work = 0, down_hot_work = 0;
    for (unsigned c : chosen->counts) {
        pair_stride64 += c;
        active += c ? 1u : 0u;
        if (c >= gate_hot_threshold) { ++hot_gate; gate_hot_work += c; }
        if (c >= down_hot_threshold) { ++hot_down; down_hot_work += c; }
    }
    if (pair_stride64 == 0 || pair_stride64 > UINT32_MAX) throw std::runtime_error("bad routing assignment count");
    if ((pair_stride64 % 6u) != 0u) throw std::runtime_error("routing assignment count is not divisible by top-k=6");
    const unsigned pair_stride = (unsigned)pair_stride64;
    const unsigned tokens = pair_stride / 6u;

    GgufView gguf(model_path);
    const std::string gate_name = replay_tensor_name(layer, ".ffn_gate_exps.weight");
    const std::string up_name = replay_tensor_name(layer, ".ffn_up_exps.weight");
    const std::string down_name = replay_tensor_name(layer, ".ffn_down_exps.weight");
    const GgufView::Tensor* tg = gguf.find(gate_name);
    const GgufView::Tensor* tu = gguf.find(up_name);
    const GgufView::Tensor* td = gguf.find(down_name);
    if (!tg || !tu || !td) throw std::runtime_error("layer routed MoE tensors not found");
    if (tg->type != 10u || tu->type != 10u || td->type != 10u || tg->ndim != 3 || tu->ndim != 3 || td->ndim != 3) {
        throw std::runtime_error("replay expects rank-3 Q2_K gate/up/down expert tensors");
    }
    if (tg->dims[2] != chosen->counts.size() || tu->dims[2] != chosen->counts.size() || td->dims[2] != chosen->counts.size()) {
        throw std::runtime_error("routing expert count does not match tensor expert dimension");
    }
    if (tg->dims[0] > UINT32_MAX || tg->dims[1] > UINT32_MAX || td->dims[0] > UINT32_MAX || td->dims[1] > UINT32_MAX) {
        throw std::runtime_error("tensor dims too large for replay");
    }
    const unsigned Kgate = (unsigned)tg->dims[0];
    const unsigned Ngate = (unsigned)tg->dims[1];
    const unsigned Kdown = (unsigned)td->dims[0];
    const unsigned Ndown = (unsigned)td->dims[1];
    if (tg->dims[0] != tu->dims[0] || tg->dims[1] != tu->dims[1] || tg->dims[2] != tu->dims[2] || Kdown != Ngate) {
        throw std::runtime_error("unexpected gate/up/down tensor geometry");
    }
    if ((Kgate % QK_K) != 0 || (Kdown % QK_K) != 0 || (Ngate % 16u) != 0 || (Ndown % 16u) != 0) {
        throw std::runtime_error("unsupported replay tensor alignment");
    }
    const uint64_t gate_row_bytes = (uint64_t)(Kgate / QK_K) * Q2K_BYTES;
    const uint64_t down_row_bytes = (uint64_t)(Kdown / QK_K) * Q2K_BYTES;
    const uint64_t gate_expert_bytes = (uint64_t)Ngate * gate_row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)Ndown * down_row_bytes;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    std::printf("Q2_K whole-layer routing replay: layer=%u dump=%u tokens=%u pairs=%u active=%u iters=%d\n",
                layer, chosen->idx, tokens, pair_stride, active, iters);
    std::printf("gate tensor: "); print_gguf_tensor(*tg);
    std::printf("up tensor:   "); print_gguf_tensor(*tu);
    std::printf("down tensor: "); print_gguf_tensor(*td);
    std::printf("hybrid thresholds: gate>=%u experts=%u work=%.2f%%, down>=%u experts=%u work=%.2f%%\n",
                gate_hot_threshold, hot_gate, 100.0 * (double)gate_hot_work / (double)pair_stride64,
                down_hot_threshold, hot_down, 100.0 * (double)down_hot_work / (double)pair_stride64);

    std::vector<int> hCounts(chosen->counts.begin(), chosen->counts.end());
    std::vector<int> hBuckets(pair_stride);
    std::vector<int> hBucketsStride((uint64_t)chosen->counts.size() * pair_stride, 0);
    std::vector<unsigned> offsets(chosen->counts.size());
    std::vector<int> ids(pair_stride);
    for (unsigned i = 0; i < pair_stride; ++i) ids[i] = (int)i;
    std::mt19937 rng(1234);
    std::shuffle(ids.begin(), ids.end(), rng);
    unsigned p = 0;
    std::vector<int> hGateHot;
    std::vector<int> hDownHot;
    unsigned max_gate_hot_count = 0;
    unsigned max_down_hot_count = 0;
    for (unsigned e = 0; e < chosen->counts.size(); ++e) {
        offsets[e] = p;
        const unsigned c = chosen->counts[e];
        for (unsigned j = 0; j < c; ++j) hBuckets[p + j] = ids[p + j];
        std::sort(hBuckets.begin() + p, hBuckets.begin() + p + c);
        for (unsigned j = 0; j < c; ++j) hBucketsStride[(uint64_t)e * pair_stride + j] = hBuckets[p + j];
        if (c >= gate_hot_threshold && c != 0u) {
            hGateHot.push_back((int)e);
            max_gate_hot_count = std::max(max_gate_hot_count, c);
        }
        if (c >= down_hot_threshold && c != 0u) {
            hDownHot.push_back((int)e);
            max_down_hot_count = std::max(max_down_hot_count, c);
        }
        p += c;
    }

    std::vector<float> hX((uint64_t)tokens * Kgate);
    std::vector<float> hDownIn((uint64_t)pair_stride * Kdown);
    fill_random(hX, 7);
    fill_random(hDownIn, 11);

    unsigned char *dGate = nullptr, *dUp = nullptr, *dDown = nullptr;
    int *dCounts = nullptr, *dBuckets = nullptr, *dBucketsStride = nullptr, *dGateHot = nullptr, *dDownHot = nullptr;
    float *dX = nullptr, *dMidCur = nullptr, *dMidHybrid = nullptr, *dDownIn = nullptr, *dOutCur = nullptr, *dOutHybrid = nullptr;
    HIP_CHECK(hipMalloc(&dGate, tg->bytes));
    HIP_CHECK(hipMalloc(&dUp, tu->bytes));
    HIP_CHECK(hipMalloc(&dDown, td->bytes));
    HIP_CHECK(hipMalloc(&dCounts, hCounts.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dBuckets, hBuckets.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dBucketsStride, hBucketsStride.size() * sizeof(int)));
    if (!hGateHot.empty()) HIP_CHECK(hipMalloc(&dGateHot, hGateHot.size() * sizeof(int)));
    if (!hDownHot.empty()) HIP_CHECK(hipMalloc(&dDownHot, hDownHot.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dMidCur, (uint64_t)pair_stride * Ngate * sizeof(float)));
    HIP_CHECK(hipMalloc(&dMidHybrid, (uint64_t)pair_stride * Ngate * sizeof(float)));
    HIP_CHECK(hipMalloc(&dDownIn, hDownIn.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOutCur, (uint64_t)pair_stride * Ndown * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOutHybrid, (uint64_t)pair_stride * Ndown * sizeof(float)));
    HIP_CHECK(hipMemcpy(dGate, gguf.data() + tg->abs_offset, tg->bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dUp, gguf.data() + tu->abs_offset, tu->bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dDown, gguf.data() + td->abs_offset, td->bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCounts, hCounts.data(), hCounts.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBuckets, hBuckets.data(), hBuckets.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBucketsStride, hBucketsStride.data(), hBucketsStride.size() * sizeof(int), hipMemcpyHostToDevice));
    if (dGateHot) HIP_CHECK(hipMemcpy(dGateHot, hGateHot.data(), hGateHot.size() * sizeof(int), hipMemcpyHostToDevice));
    if (dDownHot) HIP_CHECK(hipMemcpy(dDownHot, hDownHot.data(), hDownHot.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dDownIn, hDownIn.data(), hDownIn.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dMidCur, 0, (uint64_t)pair_stride * Ngate * sizeof(float)));
    HIP_CHECK(hipMemset(dMidHybrid, 0, (uint64_t)pair_stride * Ngate * sizeof(float)));
    HIP_CHECK(hipMemset(dOutCur, 0, (uint64_t)pair_stride * Ndown * sizeof(float)));
    HIP_CHECK(hipMemset(dOutHybrid, 0, (uint64_t)pair_stride * Ndown * sizeof(float)));

    constexpr unsigned PAIR_TILE = 8;
    const dim3 gate_cur_block(32 * 16, 1, 1);
    const dim3 gate_cur_grid((Ngate + 15u) / 16u, 256u, 1);
    const size_t gate_cur_shmem = PAIR_TILE * QK_K * sizeof(float);
    const size_t gate_m8_shmem = (8 * 16 * 16 + 2 * 16 * 16) * sizeof(half) + (2 * 8 * 16 * 16) * sizeof(float);
    auto launch_gate_current = [&](float* dst) {
        hipLaunchKernelGGL((q2_gate_up_all_current_sharedx_kernel<PAIR_TILE>), gate_cur_grid, gate_cur_block, gate_cur_shmem, 0,
                           dst, dGate, dUp, dX, dCounts, dBucketsStride, pair_stride,
                           1u, 0u, Kgate, Ngate, (unsigned)gate_row_bytes, gate_expert_bytes);
    };
    auto launch_gate_cold = [&](float* dst) {
        if (gate_hot_threshold == 0u) return;
        hipLaunchKernelGGL((q2_gate_up_all_current_sharedx_kernel<PAIR_TILE>), gate_cur_grid, gate_cur_block, gate_cur_shmem, 0,
                           dst, dGate, dUp, dX, dCounts, dBucketsStride, pair_stride,
                           1u, gate_hot_threshold, Kgate, Ngate, (unsigned)gate_row_bytes, gate_expert_bytes);
    };
    auto launch_gate_hybrid = [&](float* dst) {
        launch_gate_cold(dst);
        if (dGateHot) {
            dim3 block(32 * 8, 1, 1);
            dim3 grid(Ngate / 16u, (max_gate_hot_count + 8u * 16u - 1u) / (8u * 16u), (unsigned)hGateHot.size());
            hipLaunchKernelGGL((q2_gate_up_hotlist_wmma_multim_kernel<8,16,16,16>), grid, block, gate_m8_shmem, 0,
                               dst, dGate, dUp, dX, dCounts, dBucketsStride, dGateHot, pair_stride,
                               Kgate, Ngate, (unsigned)gate_row_bytes, gate_expert_bytes);
        }
    };

    const dim3 down_cur_block(32 * 16, 1, 1);
    const dim3 down_cur_grid((Ndown + 15u) / 16u, 256u, 1);
    const size_t down_cur_shmem = PAIR_TILE * QK_K * sizeof(float);
    const size_t down_m8_shmem = (8 * 16 * 16 + 16 * 16) * sizeof(half) + (8 * 16 * 16) * sizeof(float);
    auto launch_down_current = [&](float* dst) {
        hipLaunchKernelGGL((q2_down_all_current_sharedmid_kernel<PAIR_TILE>), down_cur_grid, down_cur_block, down_cur_shmem, 0,
                           dst, dDown, dDownIn, dCounts, dBucketsStride, pair_stride,
                           1u, 0u, Kdown, Ndown, (unsigned)down_row_bytes, down_expert_bytes);
    };
    auto launch_down_cold = [&](float* dst) {
        if (down_hot_threshold == 0u) return;
        hipLaunchKernelGGL((q2_down_all_current_sharedmid_kernel<PAIR_TILE>), down_cur_grid, down_cur_block, down_cur_shmem, 0,
                           dst, dDown, dDownIn, dCounts, dBucketsStride, pair_stride,
                           1u, down_hot_threshold, Kdown, Ndown, (unsigned)down_row_bytes, down_expert_bytes);
    };
    auto launch_down_hybrid = [&](float* dst) {
        launch_down_cold(dst);
        if (dDownHot) {
            dim3 block(32 * 8, 1, 1);
            dim3 grid(Ndown / 16u, (max_down_hot_count + 8u * 16u - 1u) / (8u * 16u), (unsigned)hDownHot.size());
            hipLaunchKernelGGL((q2_down_hotlist_wmma_multim_kernel<8,16,16,16>), grid, block, down_m8_shmem, 0,
                               dst, dDown, dDownIn, dCounts, dBucketsStride, dDownHot, pair_stride,
                               Kdown, Ndown, (unsigned)down_row_bytes, down_expert_bytes);
        }
    };

    launch_gate_current(dMidCur); HIP_CHECK(hipGetLastError());
    launch_gate_hybrid(dMidHybrid); HIP_CHECK(hipGetLastError());
    launch_down_current(dOutCur); HIP_CHECK(hipGetLastError());
    launch_down_hybrid(dOutHybrid); HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    compare_device_vectors("replay_gate_hybrid", dMidCur, dMidHybrid, (uint64_t)pair_stride * Ngate);
    compare_device_vectors("replay_down_hybrid", dOutCur, dOutHybrid, (uint64_t)pair_stride * Ndown);

    const float gate_cur_ms = time_kernel([&]() { launch_gate_current(dMidCur); }, iters);
    HIP_CHECK(hipGetLastError());
    const float gate_hybrid_ms = time_kernel([&]() { launch_gate_hybrid(dMidHybrid); }, iters);
    HIP_CHECK(hipGetLastError());
    const float down_cur_ms = time_kernel([&]() { launch_down_current(dOutCur); }, iters);
    HIP_CHECK(hipGetLastError());
    const float down_hybrid_ms = time_kernel([&]() { launch_down_hybrid(dOutHybrid); }, iters);
    HIP_CHECK(hipGetLastError());
    const double gate_flops = 4.0 * (double)pair_stride * (double)Kgate * (double)Ngate;
    const double down_flops = 2.0 * (double)pair_stride * (double)Kdown * (double)Ndown;
    std::printf("replay gate current: %.4f ms  %.2f logical TFLOP/s\n", gate_cur_ms, gate_flops / (gate_cur_ms * 1.0e-3) / 1.0e12);
    std::printf("replay gate hybrid:  %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", gate_hybrid_ms, gate_flops / (gate_hybrid_ms * 1.0e-3) / 1.0e12, gate_cur_ms / gate_hybrid_ms);
    std::printf("replay down current: %.4f ms  %.2f logical TFLOP/s\n", down_cur_ms, down_flops / (down_cur_ms * 1.0e-3) / 1.0e12);
    std::printf("replay down hybrid:  %.4f ms  %.2f logical TFLOP/s  speedup %.3fx\n", down_hybrid_ms, down_flops / (down_hybrid_ms * 1.0e-3) / 1.0e12, down_cur_ms / down_hybrid_ms);
    std::printf("replay routed total: current %.4f ms, hybrid %.4f ms, speedup %.3fx, save %.4f ms\n",
                gate_cur_ms + down_cur_ms, gate_hybrid_ms + down_hybrid_ms,
                (gate_cur_ms + down_cur_ms) / (gate_hybrid_ms + down_hybrid_ms),
                (gate_cur_ms + down_cur_ms) - (gate_hybrid_ms + down_hybrid_ms));

    HIP_CHECK(hipFree(dGate));
    HIP_CHECK(hipFree(dUp));
    HIP_CHECK(hipFree(dDown));
    HIP_CHECK(hipFree(dCounts));
    HIP_CHECK(hipFree(dBuckets));
    HIP_CHECK(hipFree(dBucketsStride));
    if (dGateHot) HIP_CHECK(hipFree(dGateHot));
    if (dDownHot) HIP_CHECK(hipFree(dDownHot));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dMidCur));
    HIP_CHECK(hipFree(dMidHybrid));
    HIP_CHECK(hipFree(dDownIn));
    HIP_CHECK(hipFree(dOutCur));
    HIP_CHECK(hipFree(dOutHybrid));
    return 0;
}

int main(int argc, char** argv) {
    unsigned M = 64;    // routed (token,slot) pairs for one expert bucket
    unsigned K = 2048;  // input dim for the selected expert matrix
    unsigned N = 4096;  // output rows for the selected expert matrix
    int iters = 50;
    unsigned pair_stride = 0;
    std::vector<unsigned char> hW;
    std::vector<unsigned char> hW2;
    std::string weight_label = "synthetic random Q2_K";
    bool gate_up_mode = false;
    bool gate_up_bucket_mode = false;
    bool down_bucket_mode = false;

    try {
        if (argc >= 2 && std::strcmp(argv[1], "--gguf-replay-routing") == 0) {
            if (argc < 5) {
                std::fprintf(stderr,
                             "usage: %s --gguf-replay-routing MODEL.gguf LAYER ROUTING_LOG [DUMP_INDEX=1] [iters=3] [GATE_HOT=64] [DOWN_HOT=32]\n",
                             argv[0]);
                return 2;
            }
            const char* path = argv[2];
            const unsigned layer = (unsigned)std::strtoul(argv[3], nullptr, 10);
            const char* routing = argv[4];
            const unsigned dump_idx = (argc >= 6) ? (unsigned)std::strtoul(argv[5], nullptr, 10) : 1u;
            const int replay_iters = (argc >= 7) ? std::atoi(argv[6]) : 3;
            const unsigned gate_hot = (argc >= 8) ? (unsigned)std::strtoul(argv[7], nullptr, 10) : 64u;
            const unsigned down_hot = (argc >= 9) ? (unsigned)std::strtoul(argv[8], nullptr, 10) : 32u;
            return run_replay_routing_layer(path, layer, routing, dump_idx, replay_iters, gate_hot, down_hot);
        }

        if (argc >= 2 && std::strcmp(argv[1], "--gguf-find") == 0) {
            if (argc < 4) {
                std::fprintf(stderr, "usage: %s --gguf-find MODEL.gguf SUBSTRING\n", argv[0]);
                return 2;
            }
            GgufView gguf(argv[2]);
            const std::string needle = argv[3];
            std::printf("GGUF mapped: size=%.2f GiB alignment=%llu tensor_data=0x%llx tensors=%zu\n",
                        (double)gguf.size() / 1073741824.0,
                        (unsigned long long)gguf.alignment(),
                        (unsigned long long)gguf.tensor_data_pos(),
                        gguf.tensors().size());
            size_t count = 0;
            for (const auto& t : gguf.tensors()) {
                if (t.name.find(needle) != std::string::npos) {
                    print_gguf_tensor(t);
                    ++count;
                }
            }
            std::printf("matches: %zu\n", count);
            return 0;
        }

        if (argc >= 2 && (std::strcmp(argv[1], "--gguf-gate-up") == 0 || std::strcmp(argv[1], "--gguf-gate-up-bucket") == 0)) {
            const bool bucket_arg = std::strcmp(argv[1], "--gguf-gate-up-bucket") == 0;
            if ((!bucket_arg && argc < 6) || (bucket_arg && argc < 8)) {
                std::fprintf(stderr,
                             "usage: %s --gguf-gate-up MODEL.gguf GATE_TENSOR UP_TENSOR EXPERT [M_pairs=64] [iters=50]\n"
                             "       %s --gguf-gate-up-bucket MODEL.gguf GATE_TENSOR UP_TENSOR EXPERT M_pairs PAIR_STRIDE [iters=50]\n"
                             "example: %s --gguf-gate-up model.gguf blk.20.ffn_gate_exps.weight blk.20.ffn_up_exps.weight 42 128 20\n",
                             argv[0], argv[0], argv[0]);
                return 2;
            }
            const char* path = argv[2];
            const std::string gate_name = argv[3];
            const std::string up_name = argv[4];
            const unsigned expert = (unsigned)std::strtoul(argv[5], nullptr, 10);
            if (bucket_arg) {
                M = (unsigned)std::strtoul(argv[6], nullptr, 10);
                pair_stride = (unsigned)std::strtoul(argv[7], nullptr, 10);
                if (argc >= 9) iters = std::atoi(argv[8]);
            } else {
                if (argc >= 7) M = (unsigned)std::strtoul(argv[6], nullptr, 10);
                if (argc >= 8) iters = std::atoi(argv[7]);
            }

            GgufView gguf(path);
            const GgufView::Tensor* tg = gguf.find(gate_name);
            const GgufView::Tensor* tu = gguf.find(up_name);
            if (!tg || !tu) {
                std::fprintf(stderr, "gate/up tensor not found: %s / %s\n", gate_name.c_str(), up_name.c_str());
                return 2;
            }
            std::printf("GGUF mapped: size=%.2f GiB alignment=%llu tensor_data=0x%llx\n",
                        (double)gguf.size() / 1073741824.0,
                        (unsigned long long)gguf.alignment(),
                        (unsigned long long)gguf.tensor_data_pos());
            print_gguf_tensor(*tg);
            print_gguf_tensor(*tu);
            if (tg->type != 10u || tu->type != 10u || tg->ndim != 3 || tu->ndim != 3 ||
                tg->dims[0] != tu->dims[0] || tg->dims[1] != tu->dims[1] || tg->dims[2] != tu->dims[2]) {
                std::fprintf(stderr, "gate/up microbench expects matching 3D Q2_K tensors\n");
                return 2;
            }
            if (expert >= tg->dims[2] || tg->dims[0] > UINT32_MAX || tg->dims[1] > UINT32_MAX) {
                std::fprintf(stderr, "bad expert or dims for gate/up tensor\n");
                return 2;
            }
            K = (unsigned)tg->dims[0];
            N = (unsigned)tg->dims[1];
            const uint64_t row_bytes64 = (K / QK_K) * (uint64_t)Q2K_BYTES;
            const uint64_t expert_bytes = (uint64_t)N * row_bytes64;
            const uint64_t gate_off = tg->abs_offset + (uint64_t)expert * expert_bytes;
            const uint64_t up_off = tu->abs_offset + (uint64_t)expert * expert_bytes;
            if ((K % QK_K) != 0 || gate_off > gguf.size() || expert_bytes > gguf.size() - gate_off ||
                up_off > gguf.size() || expert_bytes > gguf.size() - up_off) {
                std::fprintf(stderr, "bad gate/up expert slice geometry\n");
                return 2;
            }
            hW.resize((size_t)expert_bytes);
            hW2.resize((size_t)expert_bytes);
            std::memcpy(hW.data(), gguf.data() + gate_off, (size_t)expert_bytes);
            std::memcpy(hW2.data(), gguf.data() + up_off, (size_t)expert_bytes);
            const auto* first = reinterpret_cast<const block_q2_K_host*>(hW.data());
            std::printf("selected expert=%u K=%u N=%u row_bytes=%llu expert_bytes=%.3f MiB gate_off=0x%llx up_off=0x%llx\n",
                        expert, K, N, (unsigned long long)row_bytes64,
                        (double)expert_bytes / 1048576.0,
                        (unsigned long long)gate_off, (unsigned long long)up_off);
            std::printf("gate first block: d=%.6g dmin=%.6g scales[0..3]=%02x %02x %02x %02x qs[0..3]=%02x %02x %02x %02x\n",
                        f16_bits_to_f32(first->d), f16_bits_to_f32(first->dmin),
                        first->scales[0], first->scales[1], first->scales[2], first->scales[3],
                        first->qs[0], first->qs[1], first->qs[2], first->qs[3]);
            weight_label = gate_name + "+" + up_name + "/expert." + std::to_string(expert);
            gate_up_mode = !bucket_arg;
            gate_up_bucket_mode = bucket_arg;
        } else if (argc >= 2 && (std::strcmp(argv[1], "--gguf") == 0 || std::strcmp(argv[1], "--gguf-bucket") == 0)) {
            const bool bucket_arg = std::strcmp(argv[1], "--gguf-bucket") == 0;
            if ((!bucket_arg && argc < 5) || (bucket_arg && argc < 7)) {
                std::fprintf(stderr,
                             "usage: %s --gguf MODEL.gguf TENSOR_NAME EXPERT [M_pairs=64] [iters=50]\n"
                             "       %s --gguf-bucket MODEL.gguf TENSOR_NAME EXPERT M_pairs PAIR_STRIDE [iters=50]\n"
                             "example: %s --gguf model.gguf blk.20.ffn_down_exps.weight 42 64 20\n",
                             argv[0], argv[0], argv[0]);
                return 2;
            }
            const char* path = argv[2];
            const std::string tensor_name = argv[3];
            const unsigned expert = (unsigned)std::strtoul(argv[4], nullptr, 10);
            if (bucket_arg) {
                M = (unsigned)std::strtoul(argv[5], nullptr, 10);
                pair_stride = (unsigned)std::strtoul(argv[6], nullptr, 10);
                if (argc >= 8) iters = std::atoi(argv[7]);
            } else {
                if (argc >= 6) M = (unsigned)std::strtoul(argv[5], nullptr, 10);
                if (argc >= 7) iters = std::atoi(argv[6]);
            }

            GgufView gguf(path);
            const GgufView::Tensor* t = gguf.find(tensor_name);
            if (!t) {
                std::fprintf(stderr, "tensor not found: %s\n", tensor_name.c_str());
                std::fprintf(stderr, "try: %s --gguf-find %s ffn_down_exps\n", argv[0], path);
                return 2;
            }
            std::printf("GGUF mapped: size=%.2f GiB alignment=%llu tensor_data=0x%llx\n",
                        (double)gguf.size() / 1073741824.0,
                        (unsigned long long)gguf.alignment(),
                        (unsigned long long)gguf.tensor_data_pos());
            print_gguf_tensor(*t);
            if (t->type != 10u || t->ndim != 3) {
                std::fprintf(stderr, "this microbench currently expects one 3D Q2_K routed expert tensor; got type=%s rank=%u\n",
                             GgufView::type_name(t->type), t->ndim);
                return 2;
            }
            if (expert >= t->dims[2]) {
                std::fprintf(stderr, "expert %u is outside tensor expert dimension %llu\n",
                             expert, (unsigned long long)t->dims[2]);
                return 2;
            }
            if (t->dims[0] > UINT32_MAX || t->dims[1] > UINT32_MAX) {
                std::fprintf(stderr, "tensor dims are too large for this microbench\n");
                return 2;
            }
            K = (unsigned)t->dims[0];
            N = (unsigned)t->dims[1];
            const uint64_t row_bytes64 = (K / QK_K) * (uint64_t)Q2K_BYTES;
            const uint64_t expert_bytes = (uint64_t)N * row_bytes64;
            const uint64_t expert_offset = t->abs_offset + (uint64_t)expert * expert_bytes;
            if ((K % QK_K) != 0 || expert_offset > gguf.size() || expert_bytes > gguf.size() - expert_offset) {
                std::fprintf(stderr, "bad expert slice geometry: K=%u N=%u row_bytes=%llu expert_bytes=%llu\n",
                             K, N, (unsigned long long)row_bytes64, (unsigned long long)expert_bytes);
                return 2;
            }
            hW.resize((size_t)expert_bytes);
            std::memcpy(hW.data(), gguf.data() + expert_offset, (size_t)expert_bytes);
            const auto* first = reinterpret_cast<const block_q2_K_host*>(hW.data());
            std::printf("selected expert=%u K=%u N=%u row_bytes=%llu expert_bytes=%.3f MiB file_off=0x%llx\n",
                        expert, K, N, (unsigned long long)row_bytes64,
                        (double)expert_bytes / 1048576.0, (unsigned long long)expert_offset);
            std::printf("first block: d=%.6g dmin=%.6g scales[0..3]=%02x %02x %02x %02x qs[0..3]=%02x %02x %02x %02x\n",
                        f16_bits_to_f32(first->d), f16_bits_to_f32(first->dmin),
                        first->scales[0], first->scales[1], first->scales[2], first->scales[3],
                        first->qs[0], first->qs[1], first->qs[2], first->qs[3]);
            weight_label = tensor_name + "/expert." + std::to_string(expert);
            down_bucket_mode = bucket_arg;
        } else {
            if (argc >= 4) {
                M = (unsigned)std::strtoul(argv[1], nullptr, 10);
                K = (unsigned)std::strtoul(argv[2], nullptr, 10);
                N = (unsigned)std::strtoul(argv[3], nullptr, 10);
            }
            if (argc >= 5) iters = std::atoi(argv[4]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GGUF parse/map failed: %s\n", e.what());
        return 1;
    }

    if ((M % 16) || (K % 256) || (N % 16)) {
        std::fprintf(stderr,
                     "usage: %s [M K N iters] or %s --gguf MODEL.gguf TENSOR EXPERT [M_pairs] [iters], with M%%16=0 K%%256=0 N%%16=0\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (iters <= 0) iters = 1;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    if (gate_up_mode) {
        return run_gate_up_bench(hW, hW2, M, K, N, iters, weight_label);
    }
    if (gate_up_bucket_mode) {
        return run_gate_up_bucket_bench(hW, hW2, M, K, N, pair_stride, iters, weight_label);
    }
    if (down_bucket_mode) {
        return run_down_bucket_bench(hW, M, K, N, pair_stride, iters, weight_label);
    }
    std::printf("Q2_K expert slice: %s\n", weight_label.c_str());
    std::printf("Q2_K down/GEMM shape: M=%u pairs K=%u N=%u iters=%d\n", M, K, N, iters);

    const unsigned row_bytes = (K / QK_K) * Q2K_BYTES;
    if (hW.empty()) {
        hW.resize((uint64_t)N * row_bytes);
        fill_q2_weights(hW, K, N);
    }
    std::vector<float> hMid((uint64_t)M * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : hMid) x = dist(rng);

    unsigned char* dW = nullptr;
    half* dBhalf = nullptr;
    block_q8_K_host* dMidQ = nullptr;
    float *dMid = nullptr, *dCur = nullptr, *dQ8K = nullptr, *dWmma = nullptr, *dWmmaMulti4 = nullptr, *dWmmaMulti8 = nullptr, *dWmmaPacked = nullptr;
    HIP_CHECK(hipMalloc(&dW, hW.size()));
    HIP_CHECK(hipMalloc(&dBhalf, (uint64_t)K * N * sizeof(half)));
    HIP_CHECK(hipMalloc(&dMidQ, (uint64_t)M * (K / QK_K) * sizeof(block_q8_K_host)));
    HIP_CHECK(hipMalloc(&dMid, hMid.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dCur, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dQ8K, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmma, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmmaMulti4, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmmaMulti8, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWmmaPacked, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dMid, hMid.data(), hMid.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dCur, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dQ8K, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmma, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmmaMulti4, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmmaMulti8, 0, (uint64_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dWmmaPacked, 0, (uint64_t)M * N * sizeof(float)));

    auto launch_repack = [&]() {
        hipLaunchKernelGGL(q2_repack_half_kn_kernel, dim3((unsigned)(((uint64_t)K * N + 255u) / 256u)), dim3(256), 0, 0,
                           dBhalf, dW, K, N, row_bytes);
    };
    auto launch_quant_q8k = [&]() {
        hipLaunchKernelGGL(quantize_q8k_kernel, dim3(M, K / QK_K), dim3(256), 0, 0,
                           dMidQ, dMid, M, K);
    };
    launch_repack();
    HIP_CHECK(hipGetLastError());
    launch_quant_q8k();
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
    auto launch_q8k = [&]() {
        hipLaunchKernelGGL((q2_down_q8k_dot4_kernel<PAIR_TILE>), cur_grid, cur_block, 0, 0,
                           dQ8K, dW, dMidQ, M, K, N, row_bytes);
    };

    dim3 wmma_block(32, 1, 1);
    dim3 wmma_grid(N / 16, M / 16, 1);
    size_t wmma_shmem = (16 * 16 + 16 * 16) * sizeof(half);
    dim3 wmma_multi4_block(32 * 4, 1, 1);
    dim3 wmma_multi4_grid(N / 16, (M + 4 * 16 - 1) / (4 * 16), 1);
    size_t wmma_multi4_shmem = (4 * 16 * 16 + 16 * 16) * sizeof(half);
    dim3 wmma_multi8_block(32 * 8, 1, 1);
    dim3 wmma_multi8_grid(N / 16, (M + 8 * 16 - 1) / (8 * 16), 1);
    size_t wmma_multi8_shmem = (8 * 16 * 16 + 16 * 16) * sizeof(half);
    auto launch_wmma = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dWmma, dW, dMid, M, K, N, row_bytes);
    };
    auto launch_wmma_multi4 = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_multim_kernel<4,16,16,16>), wmma_multi4_grid, wmma_multi4_block, wmma_multi4_shmem, 0,
                           dWmmaMulti4, dW, dMid, M, K, N, row_bytes);
    };
    auto launch_wmma_multi8 = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_multim_kernel<8,16,16,16>), wmma_multi8_grid, wmma_multi8_block, wmma_multi8_shmem, 0,
                           dWmmaMulti8, dW, dMid, M, K, N, row_bytes);
    };
    auto launch_wmma_packed = [&]() {
        hipLaunchKernelGGL((q2_down_wmma_repacked_kernel<16,16,16>), wmma_grid, wmma_block, wmma_shmem, 0,
                           dWmmaPacked, dBhalf, dMid, M, K, N);
    };

    launch_cur();
    HIP_CHECK(hipGetLastError());
    launch_q8k();
    HIP_CHECK(hipGetLastError());
    launch_wmma();
    HIP_CHECK(hipGetLastError());
    launch_wmma_multi4();
    HIP_CHECK(hipGetLastError());
    launch_wmma_multi8();
    HIP_CHECK(hipGetLastError());
    launch_wmma_packed();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hCur((uint64_t)M * N), hQ8K((uint64_t)M * N), hWmma((uint64_t)M * N), hWmmaMulti4((uint64_t)M * N), hWmmaMulti8((uint64_t)M * N), hWmmaPacked((uint64_t)M * N);
    HIP_CHECK(hipMemcpy(hCur.data(), dCur, hCur.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hQ8K.data(), dQ8K, hQ8K.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hWmma.data(), dWmma, hWmma.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hWmmaMulti4.data(), dWmmaMulti4, hWmmaMulti4.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hWmmaMulti8.data(), dWmmaMulti8, hWmmaMulti8.size() * sizeof(float), hipMemcpyDeviceToHost));
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
    compare("q8k_dot4", hQ8K);
    compare("wmma_direct", hWmma);
    compare("wmma_multi4", hWmmaMulti4);
    compare("wmma_multi8", hWmmaMulti8);
    compare("wmma_packed", hWmmaPacked);

    float repack_ms = time_kernel(launch_repack, std::max(1, std::min(iters, 20)));
    float quant_q8k_ms = time_kernel(launch_quant_q8k, iters);
    float cur_ms = time_kernel(launch_cur, iters);
    float q8k_ms = time_kernel(launch_q8k, iters);
    float wmma_ms = time_kernel(launch_wmma, iters);
    float wmma_multi4_ms = time_kernel(launch_wmma_multi4, iters);
    float wmma_multi8_ms = time_kernel(launch_wmma_multi8, iters);
    float wmma_packed_ms = time_kernel(launch_wmma_packed, iters);
    double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("repack Q2_K->KxN half: %.4f ms  %.2f GiB/s output\n", repack_ms,
                ((double)K * N * sizeof(half) / 1073741824.0) / (repack_ms * 1.0e-3));
    std::printf("quant mid->Q8_K:    %.4f ms  %.2f GiB/s input\n", quant_q8k_ms,
                ((double)M * K * sizeof(float) / 1073741824.0) / (quant_q8k_ms * 1.0e-3));
    std::printf("current-like:       %.4f ms  %.2f TFLOP/s\n", cur_ms, flops / (cur_ms * 1.0e-3) / 1.0e12);
    std::printf("q8k dot4 proto:     %.4f ms  %.2f logical TFLOP/s  speedup %.3fx  incl_quant %.3fx\n",
                q8k_ms, flops / (q8k_ms * 1.0e-3) / 1.0e12, cur_ms / q8k_ms, cur_ms / (q8k_ms + quant_q8k_ms));
    std::printf("wmma direct proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_ms, flops / (wmma_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_ms);
    std::printf("wmma multi4 proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_multi4_ms, flops / (wmma_multi4_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_multi4_ms);
    std::printf("wmma multi8 proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_multi8_ms, flops / (wmma_multi8_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_multi8_ms);
    std::printf("wmma packed proto:  %.4f ms  %.2f TFLOP/s  speedup %.3fx\n", wmma_packed_ms, flops / (wmma_packed_ms * 1.0e-3) / 1.0e12, cur_ms / wmma_packed_ms);

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dBhalf));
    HIP_CHECK(hipFree(dMidQ));
    HIP_CHECK(hipFree(dMid));
    HIP_CHECK(hipFree(dCur));
    HIP_CHECK(hipFree(dQ8K));
    HIP_CHECK(hipFree(dWmma));
    HIP_CHECK(hipFree(dWmmaMulti4));
    HIP_CHECK(hipFree(dWmmaMulti8));
    HIP_CHECK(hipFree(dWmmaPacked));
    return 0;
}
