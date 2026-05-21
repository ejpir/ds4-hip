// DS4 ROCm hierarchical-composition (HC) split/sum/expand kernels.
//
// Included from ds4_cuda.cu in the same translation unit to preserve current
// static helper visibility and launch behavior.

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand_half_kernel(
        float *out_hc,
        const __half *block_out,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    const float block_v = __half2float(block_out[(uint64_t)t * n_embd + d]);
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand_add_half_kernel(
        float *out_hc,
        const float *block_out,
        const __half *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    const float block_v = block_out[(uint64_t)t * n_embd + d] +
                          __half2float(block_add[(uint64_t)t * n_embd + d]);
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_expand4_half_kernel(
        float *out_hc,
        const __half *block_out,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = __half2float(block_out[td]);
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 20u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

__global__ static void hc_expand4_add_half_kernel(
        float *out_hc,
        const float *block_out,
        const __half *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t t = gid / n_embd;
    const uint64_t td = (uint64_t)t * n_embd + d;
    const uint64_t hc_base = (uint64_t)t * 4u * n_embd + d;
    const float bv = block_out[td] + __half2float(block_add[td]);
    const float r0 = residual_hc[hc_base + 0u * (uint64_t)n_embd];
    const float r1 = residual_hc[hc_base + 1u * (uint64_t)n_embd];
    const float r2 = residual_hc[hc_base + 2u * (uint64_t)n_embd];
    const float r3 = residual_hc[hc_base + 3u * (uint64_t)n_embd];
    const float *sp = split + (uint64_t)t * 20u;
    const float *post = sp + 4u;
    const float *comb = sp + 8u;
#pragma unroll
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

__device__ static float hc_warp_sum_oldhip_w32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float hc_block_sum_oldhip_w32(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wid = tid >> 5u;
    const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
    v = hc_warp_sum_oldhip_w32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : 0.0f;
    if (wid == 0u) v = hc_warp_sum_oldhip_w32(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__global__ static void hc_split4_weighted_sum_norm_oldhip_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    __shared__ float s[24];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows) return;
    const float *mix_row = mix + (uint64_t)row * 24u;
    float *split_row = split + (uint64_t)row * 24u;
    if (tid == 0u) {
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        for (uint32_t i = 0; i < 4u; i++) s[i] = 1.0f / (1.0f + expf(-(mix_row[i] * pre_scale + base[i]))) + epsv;
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t off = 4u + i;
            s[off] = 2.0f / (1.0f + expf(-(mix_row[off] * post_scale + base[off])));
        }
        float c[16];
        for (uint32_t dst = 0; dst < 4u; dst++) {
            float row_max = -1.0e30f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const uint32_t off = 8u + idx;
                const float v = mix_row[off] * comb_scale + base[off];
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
                c[idx] = c[idx] * inv + epsv;
            }
        }
        for (uint32_t src = 0; src < 4u; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
            const float inv = 1.0f / (sum + epsv);
            for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
        }
        for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
            for (uint32_t dst = 0; dst < 4u; dst++) {
                float sum = 0.0f;
                for (uint32_t src = 0; src < 4u; src++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + epsv);
                for (uint32_t src = 0; src < 4u; src++) c[src + dst * 4u] *= inv;
            }
            for (uint32_t src = 0; src < 4u; src++) {
                float sum = 0.0f;
                for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + epsv);
                for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
            }
        }
        for (uint32_t i = 0; i < 16u; i++) s[8u + i] = c[i];
        for (uint32_t i = 0; i < 24u; i++) split_row[i] = s[i];
    }
    __syncthreads();

    const float *res = residual_hc + (uint64_t)row * 4u * n_embd;
    float *out_row = out + (uint64_t)row * n_embd;
    float *norm_row = norm_out + (uint64_t)row * n_embd;
    float ss = 0.0f;
    for (uint32_t d = tid; d < n_embd; d += blockDim.x) {
        const float v = res[d] * s[0] +
                        res[(uint64_t)n_embd + d] * s[1] +
                        res[(uint64_t)2u * n_embd + d] * s[2] +
                        res[(uint64_t)3u * n_embd + d] * s[3];
        out_row[d] = v;
        ss += v * v;
    }
    const float total = hc_block_sum_oldhip_w32(ss);
    const float inv = rsqrtf(total / (float)n_embd + norm_eps);
    for (uint32_t d = tid; d < n_embd; d += blockDim.x) norm_row[d] = out_row[d] * inv * norm_w[d];
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

