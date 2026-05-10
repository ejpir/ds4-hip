// Tiny rocWMMA smoke + GEMM microbench for gfx1151 / wave32.
// Build with: make hip-rocwmma-smoke
// Or directly:
//   hipcc --offload-arch=gfx1151 -O3 -std=c++17 tools/hip_rocwmma_smoke.cpp -o hip-rocwmma-smoke

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma-version.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { \
    hipError_t err__ = (x); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

template <int BM=16, int BN=16, int BK=16>
__global__ void rocwmma_gemm_16x16_kernel(const half* __restrict__ A,
                                          const half* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K)
{
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    int row = tile_m * BM;
    int col = tile_n * BN;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;

    frag_a a;
    frag_b b;
    frag_c acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        rocwmma::load_matrix_sync(a, A + row * K + k0, K);
        rocwmma::load_matrix_sync(b, B + k0 * N + col, N);
        rocwmma::mma_sync(acc, a, b, acc);
    }

    rocwmma::store_matrix_sync(C + row * N + col, acc, N, rocwmma::mem_row_major);
}

__global__ void naive_gemm_kernel(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) {
        s += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = s;
}

static void cpu_ref(const std::vector<half>& A, const std::vector<half>& B, std::vector<float>& C,
                    int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = s;
        }
    }
}

template <typename LaunchFn>
static float time_kernel(LaunchFn launch, int iters)
{
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

int main(int argc, char** argv)
{
    int M = 1024, N = 1024, K = 1024;
    int iters = 100;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) iters = std::atoi(argv[4]);
    if (M % 16 || N % 16 || K % 16) {
        std::fprintf(stderr, "M/N/K must be multiples of 16\n");
        return 2;
    }

    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("rocWMMA version: %s\n", rocwmma_get_version().c_str());
    std::printf("device: %s arch=%s warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);
    std::printf("GEMM: M=%d N=%d K=%d iters=%d\n", M, N, K, iters);

    std::vector<half> hA((size_t)M * K), hB((size_t)K * N);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = __float2half(dist(rng));
    for (auto& x : hB) x = __float2half(dist(rng));

    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr, *dCnaive = nullptr;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dCnaive, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(dCnaive, 0, (size_t)M * N * sizeof(float)));

    dim3 block_wmma(32, 1, 1); // one wave32 per 16x16 tile
    dim3 grid_wmma(N / 16, M / 16, 1);
    auto launch_wmma = [&]() {
        hipLaunchKernelGGL((rocwmma_gemm_16x16_kernel<16,16,16>), grid_wmma, block_wmma, 0, 0, dA, dB, dC, M, N, K);
    };
    launch_wmma();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    dim3 block_naive(16, 16, 1);
    dim3 grid_naive((N + 15) / 16, (M + 15) / 16, 1);
    auto launch_naive = [&]() {
        hipLaunchKernelGGL(naive_gemm_kernel, grid_naive, block_naive, 0, 0, dA, dB, dCnaive, M, N, K);
    };
    launch_naive();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hC((size_t)M * N), hCnaive((size_t)M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hCnaive.data(), dCnaive, hCnaive.size() * sizeof(float), hipMemcpyDeviceToHost));

    double max_abs = 0.0, rms = 0.0, max_abs_vs_cpu = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
        double e = double(hC[i]) - double(hCnaive[i]);
        max_abs = std::max(max_abs, std::abs(e));
        rms += e * e;
    }
    rms = std::sqrt(rms / std::max<size_t>(1, hC.size()));

    // CPU reference only for small problems to keep smoke fast.
    if ((long long)M * N * K <= 128LL * 128 * 128) {
        std::vector<float> hRef((size_t)M * N);
        cpu_ref(hA, hB, hRef, M, N, K);
        for (size_t i = 0; i < hC.size(); ++i) {
            max_abs_vs_cpu = std::max(max_abs_vs_cpu, std::abs(double(hC[i]) - double(hRef[i])));
        }
        std::printf("max_abs_vs_cpu=%.6g\n", max_abs_vs_cpu);
    }
    std::printf("max_abs_vs_naive=%.6g rms_vs_naive=%.6g\n", max_abs, rms);

    float ms_wmma = time_kernel(launch_wmma, iters);
    // Fewer naive iters because it is intentionally slow.
    int naive_iters = std::max(1, std::min(iters, 10));
    float ms_naive = time_kernel(launch_naive, naive_iters);
    double flops = 2.0 * double(M) * double(N) * double(K);
    std::printf("rocWMMA: %.4f ms  %.2f TFLOP/s\n", ms_wmma, flops / (ms_wmma * 1.0e-3) / 1.0e12);
    std::printf("naive:   %.4f ms  %.2f TFLOP/s\n", ms_naive, flops / (ms_naive * 1.0e-3) / 1.0e12);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(dCnaive));
    return (max_abs > 2.0e-2 || rms > 1.0e-3) ? 1 : 0;
}
