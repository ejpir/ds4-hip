static uint32_t cuda_q8_tile_env(const char *primary, const char *alias, uint32_t def) {
    uint32_t tile = cuda_parse_u32_env_alias(primary, alias, def, 2u, 32u);
    if (tile != 2u && tile != 4u && tile != 8u && tile != 16u && tile != 32u) tile = def;
    return tile;
}

static uint32_t cuda_q8_block_tile_env(const char *primary, const char *alias, uint32_t def, uint32_t tile) {
    uint32_t block_tile = cuda_parse_u32_env_alias(primary, alias, def, 8u, 32u);
    if (block_tile != 8u && block_tile != 16u && block_tile != 32u) block_tile = def;
    while ((uint64_t)tile * block_tile * 32u * sizeof(float) > 65536u && block_tile > 8u) block_tile >>= 1u;
    return block_tile;
}

template <uint32_t BT>
static void cuda_launch_q8_batch_sharedx_bt(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 4u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 8u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 16u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    }
}

static void cuda_launch_q8_batch_sharedx(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    (n_tok + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_q8_batch_sharedx_bt<8u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_q8_batch_sharedx_bt<32u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_q8_batch_sharedx_bt<16u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void cuda_launch_grouped_q8_a_sharedx_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    }
}

static void cuda_launch_grouped_q8_a_sharedx(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_grouped_q8_a_sharedx_bt<8u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_grouped_q8_a_sharedx_bt<32u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_grouped_q8_a_sharedx_bt<16u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    }
}

static int cuda_matmul_q8_0_tensor_f16_gemm(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     xh,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out->ptr,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

static int cuda_matmul_q8_0_tensor_f16_gemm_out_half(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out_h || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(__half), &out_bytes) ||
        x->bytes < x_bytes || out_h->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16-out gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16-out activation convert launch")) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     xh,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out_h->ptr,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16-out matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16-out matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_f16_gemm_out_half(out_h, model_map, model_size,
                                                     weight_offset, in_dim, out_dim,
                                                     x, n_tok, "q8_f16_out");
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const int attn_q_b_shape = (in_dim == 1024u && out_dim == 32768u);
    if (n_tok > 1 &&
        ((label && strstr(label, "attn_q_b") != NULL) || attn_q_b_shape) &&
        !g_quality_mode &&
        (cuda_env_flag_any3("DS4_CUDA_ATTN_Q_B_CUBLAS", "DS4_HIP_ATTN_Q_B_CUBLAS", NULL) ||
         cuda_env_flag_any3("DS4_CUDA_Q_PATH_CUBLAS", "DS4_HIP_Q_PATH_CUBLAS", NULL)) &&
        cuda_matmul_q8_0_tensor_f16_gemm(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok, label ? label : "attn_q_b")) {
        return 1;
    }
    if (n_tok > 1 && !g_quality_mode &&
        ((cuda_runtime_config()->shared_expert_cublas &&
          ((in_dim == 4096u && out_dim == 2048u) || (in_dim == 2048u && out_dim == 4096u))) ||
         (cuda_runtime_config()->shared_down_cublas && in_dim == 2048u && out_dim == 4096u)) &&
        cuda_matmul_q8_0_tensor_f16_gemm(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok, label ? label : "shared_expert")) {
        return 1;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (n_tok == 1 && !cuda_runtime_config()->q8_prequant_decode) {
        if (cuda_env_present("DS4_CUDA_OLDHIP_Q8_SMALL_DECODE_BLOCK") &&
            !cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_SMALL_DECODE_BLOCK") &&
            cuda_offset_in_env_range(weight_offset,
                                     "DS4_CUDA_OLDHIP_Q8_SMALL_DECODE_OFFSETS",
                                     "DS4_CUDA_OLDHIP_Q8_SMALL_DECODE_MIN_OFFSET",
                                     "DS4_CUDA_OLDHIP_Q8_SMALL_DECODE_MAX_OFFSET") &&
            (in_dim & 31u) == 0u && out_dim < 1024u) {
            const unsigned threads = in_dim >= 8192u ? 1024u : 256u;
            matmul_q8_0_f32_small_block_w32_kernel<<<(unsigned)out_dim, threads>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 small-block launch");
        }
        if (!cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_DECODE_SHAREDX") &&
            (in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 sharedx launch");
        }
        matmul_q8_0_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 warp launch");
    }
    if (n_tok > 1 && !cuda_runtime_config()->q8_prequant_batch) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (cuda_env_flag_any3("DS4_CUDA_Q8_WMMA_4W", "DS4_HIP_Q8_WMMA_4W", NULL) &&
            !g_quality_mode && (in_dim % 32u) == 0u &&
            out_dim >= cuda_parse_u32_env_alias("DS4_CUDA_Q8_WMMA_4W_MIN_OUT", "DS4_HIP_Q8_WMMA_4W_MIN_OUT", 1024u, 16u, UINT32_MAX) &&
            n_tok >= cuda_parse_u32_env_alias("DS4_CUDA_Q8_WMMA_4W_MIN_TOKENS", "DS4_HIP_Q8_WMMA_4W_MIN_TOKENS", 256u, 1u, 65535u) &&
            in_dim <= UINT32_MAX && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            const dim3 grid((uint32_t)((out_dim + 63u) / 64u),
                            (uint32_t)((n_tok + 63u) / 64u),
                            1u);
            matmul_q8_0_f32_batch_wmma_4w_kernel<<<grid, 128u>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)n_tok,
                    (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch wmma 4w launch");
        }
        if ((cuda_env_present("DS4_CUDA_Q8_WMMA_ONFLY") || cuda_env_present("DS4_CUDA_Q8_WMMA_FAST")) &&
            !g_quality_mode && (in_dim % 16u) == 0u && (out_dim % 16u) == 0u &&
            n_tok >= cuda_parse_u32_env_alias("DS4_CUDA_Q8_WMMA_MIN_TOKENS", "DS4_HIP_Q8_WMMA_MIN_TOKENS", 2u, 1u, 65535u) &&
            in_dim <= UINT32_MAX && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
            const uint32_t tiles_n_default = (out_dim >= 8192u) ? 32u : 16u;
            uint32_t tiles_n = cuda_parse_u32_env_alias("DS4_CUDA_Q8_WMMA_TILES_N", "DS4_HIP_Q8_WMMA_TILES_N", tiles_n_default, 4u, 32u);
            if (tiles_n != 4u && tiles_n != 8u && tiles_n != 16u && tiles_n != 32u) tiles_n = tiles_n_default;
            const dim3 grid((uint32_t)((out_dim + tiles_n * bn - 1u) / (tiles_n * bn)),
                            (uint32_t)((n_tok + bm - 1u) / bm),
                            1u);
            const size_t shmem = (bm * bk + tiles_n * bk * bn) * sizeof(half) +
                                 (tiles_n * bm * bn) * sizeof(float);
            if (tiles_n == 4u) {
                matmul_q8_0_f32_batch_wmma_onthefly_kernel<4,16,16,16><<<grid, 128u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
            } else if (tiles_n == 16u) {
                matmul_q8_0_f32_batch_wmma_onthefly_kernel<16,16,16,16><<<grid, 512u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
            } else if (tiles_n == 32u) {
                matmul_q8_0_f32_batch_wmma_onthefly_kernel<32,16,16,16><<<grid, 1024u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
            } else {
                matmul_q8_0_f32_batch_wmma_onthefly_kernel<8,16,16,16><<<grid, 256u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
            }
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch wmma onfly launch");
        }
#endif
        if (!cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_BATCH_SHAREDX") &&
            (in_dim & 31u) == 0u && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            uint32_t rows_per_block = cuda_parse_u32_env_alias("DS4_CUDA_Q8_BATCH_RPB", "DS4_HIP_Q8_BATCH_RPB", 32u, 1u, 32u);
            if (rows_per_block == 0u) rows_per_block = 32u;
            const uint32_t tile = cuda_q8_tile_env("DS4_CUDA_Q8_BATCH_TILE", "DS4_HIP_Q8_BATCH_TILE", 32u);
            const uint32_t block_tile = cuda_q8_block_tile_env("DS4_CUDA_Q8_BATCH_SHARED_X_BLOCKS", "DS4_HIP_Q8_BATCH_SHARED_X_BLOCKS", 16u, tile);
            cuda_launch_q8_batch_sharedx((float *)out->ptr,
                                         reinterpret_cast<const unsigned char *>(wptr),
                                         (const float *)x->ptr,
                                         (uint32_t)blocks,
                                         (uint32_t)out_dim,
                                         (uint32_t)n_tok,
                                         blocks * 34u,
                                         rows_per_block,
                                         tile,
                                         block_tile);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch sharedx launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_f32_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch warp launch");
    }
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    const int use_dp4a = cfg->q8_use_dp4a;
    const int profile_decode = (n_tok == 1u && cfg->q8_decode_profile);
    cudaEvent_t prof0 = NULL, prof1 = NULL, prof2 = NULL;
    if (profile_decode) {
        if (cudaEventCreate(&prof0) != cudaSuccess ||
            cudaEventCreate(&prof1) != cudaSuccess ||
            cudaEventCreate(&prof2) != cudaSuccess) {
            if (prof0) (void)cudaEventDestroy(prof0);
            if (prof1) (void)cudaEventDestroy(prof1);
            if (prof2) (void)cudaEventDestroy(prof2);
            prof0 = prof1 = prof2 = NULL;
        } else {
            (void)cudaEventRecord(prof0, 0);
        }
    }
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) {
        if (prof0) (void)cudaEventDestroy(prof0);
        if (prof1) (void)cudaEventDestroy(prof1);
        if (prof2) (void)cudaEventDestroy(prof2);
        return 0;
    }
    if (prof1) (void)cudaEventRecord(prof1, 0);
    if (n_tok == 1) {
        uint32_t rows_per_block = cfg->q8_decode_rpb;
        matmul_q8_0_preq_rows_w32_kernel<<<((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                                            rows_per_block * 32u>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                rows_per_block,
                use_dp4a);
        const int ok_rows = cuda_ok(cudaGetLastError(), "matmul_q8_0 rows launch");
        if (prof2) {
            (void)cudaEventRecord(prof2, 0);
            if (cudaEventSynchronize(prof2) == cudaSuccess) {
                float ms_q = 0.0f, ms_mm = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_q, prof0, prof1);
                (void)cudaEventElapsedTime(&ms_mm, prof1, prof2);
                (void)cudaEventElapsedTime(&ms_total, prof0, prof2);
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "q8 decode profile label=%s in=%llu out=%llu blocks=%llu rpb=%u quant=%.3f matmul=%.3f total=%.3f ms\n",
                        label ? label : "q8_0",
                        (unsigned long long)in_dim,
                        (unsigned long long)out_dim,
                        (unsigned long long)blocks,
                        rows_per_block,
                        ms_q,
                        ms_mm,
                        ms_total);
            }
        }
        if (prof0) (void)cudaEventDestroy(prof0);
        if (prof1) (void)cudaEventDestroy(prof1);
        if (prof2) (void)cudaEventDestroy(prof2);
        return ok_rows;
    }
    if (!cuda_env_present("DS4_CUDA_NO_Q8_BATCH_WARP") && blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map ||
        in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight0_bytes = 0, weight1_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out0_dim, row_bytes, &weight0_bytes) ||
        !cuda_u64_mul_checked(out1_dim, row_bytes, &weight1_bytes)) {
        return 0;
    }
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;
    if (!cuda_runtime_config()->q8_prequant_decode) {
        const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
        if (!cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_DECODE_SHAREDX") &&
            !cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_PAIR_DECODE_SHAREDX") &&
            (in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((max_out + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr,
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    reinterpret_cast<const unsigned char *>(w1),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out0_dim,
                    out1_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 sharedx launch");
        }
        matmul_q8_0_pair_f32_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                reinterpret_cast<const unsigned char *>(w1),
                (const float *)x->ptr,
                in_dim,
                out0_dim,
                out1_dim,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 warp launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_runtime_config()->q8_use_dp4a;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;
    if (!cuda_runtime_config()->q8_prequant_decode) {
        /* Production HIP uses split-K16 accumulation for the decode HC-expand
         * Q8 projections whose reductions are most sensitive to FP8/router
         * near-ties.  Mirror that shape before falling back to the full-row
         * fused kernel. */
        const int store_block_out = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        if (!block_add &&
            cuda_env_present("DS4_CUDA_SPLITK_ATTN_OUT_B") &&
            !cuda_env_present("DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_B") &&
            cuda_offset_in_env_range(weight_offset,
                                     "DS4_CUDA_SPLITK_ATTN_OUT_B_OFFSETS",
                                     "DS4_CUDA_SPLITK_ATTN_OUT_B_MIN_OFFSET",
                                     "DS4_CUDA_SPLITK_ATTN_OUT_B_MAX_OFFSET") &&
            in_dim == 8192u && out_dim == 4096u && n_hc == 4u && blocks == 256u) {
            int have_splits = 0;
            uint32_t n_splits = (uint32_t)cuda_parse_u64_env("DS4_CUDA_SPLITK_ATTN_OUT_B_SPLITS", &have_splits);
            if (!have_splits || n_splits == 0u) n_splits = 16u;
            if (n_splits > blocks) n_splits = (uint32_t)blocks;
            float *partial = (float *)cuda_tmp_alloc((uint64_t)n_splits * out_dim * sizeof(float), "q8 hc expand attn_out_b splitk");
            if (!partial) return 0;
            if (n_splits == 16u) {
                matmul_q8_0_hc_partial16_w32_kernel<<<dim3((unsigned)((out_dim + 31u) / 32u), 16u),
                                                       1024u, 512u * sizeof(float)>>>(
                        partial,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)out_dim,
                        blocks * 34u);
                if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand attn_out_b splitk16 partial launch")) return 0;
                hc_expand_partial16_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out_hc->ptr,
                        (float *)block_out->ptr,
                        partial,
                        (const float *)residual_hc->ptr,
                        (const float *)split->ptr,
                        (uint32_t)out_dim,
                        n_hc,
                        store_block_out);
            } else {
                const uint32_t chunk = ((uint32_t)blocks + n_splits - 1u) / n_splits;
                matmul_q8_0_hc_partial_w32_kernel<<<dim3((unsigned)((out_dim + 31u) / 32u), n_splits),
                                                     1024u, (size_t)(chunk << 5) * sizeof(float)>>>(
                        partial,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)blocks,
                        (uint32_t)out_dim,
                        blocks * 34u,
                        n_splits);
                if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand attn_out_b splitk partial launch")) return 0;
                hc_expand_partial_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out_hc->ptr,
                        (float *)block_out->ptr,
                        partial,
                        (const float *)residual_hc->ptr,
                        (const float *)split->ptr,
                        (uint32_t)out_dim,
                        n_hc,
                        n_splits,
                        store_block_out);
            }
            return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand attn_out_b splitk expand launch");
        }
        if (block_add &&
            cuda_env_present("DS4_CUDA_SPLITK_SHARED_DOWN") &&
            !cuda_env_present("DS4_CUDA_DISABLE_SPLITK_SHARED_DOWN") &&
            cuda_offset_in_env_range(weight_offset,
                                     "DS4_CUDA_SPLITK_SHARED_DOWN_OFFSETS",
                                     "DS4_CUDA_SPLITK_SHARED_DOWN_MIN_OFFSET",
                                     "DS4_CUDA_SPLITK_SHARED_DOWN_MAX_OFFSET") &&
            in_dim == 2048u && out_dim == 4096u && n_hc == 4u && blocks == 64u) {
            int have_splits = 0;
            uint32_t n_splits = (uint32_t)cuda_parse_u64_env("DS4_CUDA_SPLITK_SHARED_DOWN_SPLITS", &have_splits);
            if (!have_splits || n_splits == 0u) n_splits = 4u;
            if (n_splits > blocks) n_splits = (uint32_t)blocks;
            float *partial = (float *)cuda_tmp_alloc((uint64_t)n_splits * out_dim * sizeof(float), "q8 hc expand shared_down splitk");
            if (!partial) return 0;
            if (n_splits == 4u) {
                matmul_q8_0_hc_partial16_w32_kernel<<<dim3((unsigned)((out_dim + 31u) / 32u), 4u),
                                                       1024u, 512u * sizeof(float)>>>(
                        partial,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)out_dim,
                        blocks * 34u);
                if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand shared_down splitk4 partial launch")) return 0;
                hc_expand_add_partial4_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out_hc->ptr,
                        (float *)block_out->ptr,
                        partial,
                        (const float *)block_add->ptr,
                        (const float *)residual_hc->ptr,
                        (const float *)split->ptr,
                        (uint32_t)out_dim,
                        n_hc,
                        store_block_out);
            } else {
                const uint32_t chunk = ((uint32_t)blocks + n_splits - 1u) / n_splits;
                matmul_q8_0_hc_partial_w32_kernel<<<dim3((unsigned)((out_dim + 31u) / 32u), n_splits),
                                                     1024u, (size_t)(chunk << 5) * sizeof(float)>>>(
                        partial,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)blocks,
                        (uint32_t)out_dim,
                        blocks * 34u,
                        n_splits);
                if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand shared_down splitk partial launch")) return 0;
                hc_expand_add_partial_kernel<<<(unsigned)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out_hc->ptr,
                        (float *)block_out->ptr,
                        partial,
                        (const float *)block_add->ptr,
                        (const float *)residual_hc->ptr,
                        (const float *)split->ptr,
                        (uint32_t)out_dim,
                        n_hc,
                        n_splits,
                        store_block_out);
            }
            return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand shared_down splitk expand launch");
        }
        if (!cuda_env_present("DS4_CUDA_NO_OLDHIP_Q8_HC_EXPAND_SHAREDX") &&
            (in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_hc_expand_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out_hc->ptr,
                    (float *)block_out->ptr,
                    block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                    (const float *)residual_hc->ptr,
                    (const float *)split->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u,
                    n_embd,
                    n_hc,
                    block_add ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 sharedx launch");
        }
        matmul_q8_0_hc_expand_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                blocks,
                block_add ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    const int use_dp4a = cfg->q8_use_dp4a;
    const int profile_decode = cfg->q8_decode_profile;
    cudaEvent_t prof0 = NULL, prof1 = NULL, prof2 = NULL;
    if (profile_decode) {
        if (cudaEventCreate(&prof0) != cudaSuccess ||
            cudaEventCreate(&prof1) != cudaSuccess ||
            cudaEventCreate(&prof2) != cudaSuccess) {
            if (prof0) (void)cudaEventDestroy(prof0);
            if (prof1) (void)cudaEventDestroy(prof1);
            if (prof2) (void)cudaEventDestroy(prof2);
            prof0 = prof1 = prof2 = NULL;
        } else {
            (void)cudaEventRecord(prof0, 0);
        }
    }
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) {
        if (prof0) (void)cudaEventDestroy(prof0);
        if (prof1) (void)cudaEventDestroy(prof1);
        if (prof2) (void)cudaEventDestroy(prof2);
        return 0;
    }
    if (prof1) (void)cudaEventRecord(prof1, 0);
    uint32_t rows_per_block = cfg->q8_hc_decode_rpb;
    matmul_q8_0_hc_expand_preq_rows_w32_kernel<<<((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                                                  rows_per_block * 32u>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            rows_per_block,
            block_add ? 1 : 0,
            use_dp4a);
    const int ok_rows = cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand rows launch");
    if (prof2) {
        (void)cudaEventRecord(prof2, 0);
        if (cudaEventSynchronize(prof2) == cudaSuccess) {
            float ms_q = 0.0f, ms_mm = 0.0f, ms_total = 0.0f;
            (void)cudaEventElapsedTime(&ms_q, prof0, prof1);
            (void)cudaEventElapsedTime(&ms_mm, prof1, prof2);
            (void)cudaEventElapsedTime(&ms_total, prof0, prof2);
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "q8 hc decode profile label=%s in=%llu out=%llu blocks=%llu rpb=%u add=%u quant=%.3f matmul_hc=%.3f total=%.3f ms\n",
                    label ? label : "q8_0_hc_expand",
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim,
                    (unsigned long long)blocks,
                    rows_per_block,
                    block_add ? 1u : 0u,
                    ms_q,
                    ms_mm,
                    ms_total);
        }
    }
    if (prof0) (void)cudaEventDestroy(prof0);
    if (prof1) (void)cudaEventDestroy(prof1);
    if (prof2) (void)cudaEventDestroy(prof2);
    return ok_rows;
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int ordered_router =
        n_tok == 1u &&
        !cuda_env_present("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    if (g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    /* The 4096x256 F16 router projection is latency-bound and the ordered
     * 32-thread row kernel is at least as fast on gfx1151; keep shared-X for
     * compressor/indexer F16 decode where reusing x across rows is the win. */
    const bool f16_decode_router_shape = (in_dim == 4096u && out_dim == 256u);
    const bool f16_decode_force_router_sharedx =
        cuda_env_present("DS4_CUDA_F16_DECODE_SHAREDX_ROUTER") ||
        cuda_env_present("DS4_HIP_F16_DECODE_SHAREDX_ROUTER");
    if (n_tok == 1u && !g_quality_mode && !cuda_runtime_config()->graph_dump &&
        (!f16_decode_router_shape || f16_decode_force_router_sharedx) &&
        !cuda_env_present("DS4_CUDA_NO_F16_DECODE_SHAREDX") &&
        !cuda_env_present("DS4_HIP_NO_F16_DECODE_SHAREDX") &&
        !cuda_env_present("DS4_CUDA_NO_F16_DECODE_SHAREDX_SINGLE") &&
        !cuda_env_present("DS4_HIP_NO_F16_DECODE_SHAREDX_SINGLE")) {
        const uint32_t max_shared = cuda_parse_u32_env_alias("DS4_CUDA_F16_DECODE_SHAREDX_MAX",
                                                            "DS4_HIP_F16_DECODE_SHAREDX_MAX",
                                                            8192u, 256u, 16384u);
        if (in_dim <= max_shared && in_dim * sizeof(float) <= 65536u) {
            uint32_t rows_per_block = cuda_parse_u32_env_alias("DS4_CUDA_F16_DECODE_RPB",
                                                               "DS4_HIP_F16_DECODE_RPB",
                                                               16u, 1u, 32u);
            if (rows_per_block != 1u && rows_per_block != 2u && rows_per_block != 4u &&
                rows_per_block != 8u && rows_per_block != 16u && rows_per_block != 32u) {
                rows_per_block = 16u;
            }
            matmul_f16_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16 sharedx launch");
        }
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1 ||
        cuda_env_present("DS4_CUDA_NO_F16_PAIR_MATMUL") ||
        cuda_env_present("DS4_CUDA_NO_ORDERED_F16_MATMUL")) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    uint64_t weight_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (!g_quality_mode && !cuda_runtime_config()->graph_dump &&
        !cuda_env_present("DS4_CUDA_NO_F16_DECODE_SHAREDX") &&
        !cuda_env_present("DS4_HIP_NO_F16_DECODE_SHAREDX")) {
        const uint32_t max_shared = cuda_parse_u32_env_alias("DS4_CUDA_F16_DECODE_SHAREDX_MAX",
                                                            "DS4_HIP_F16_DECODE_SHAREDX_MAX",
                                                            8192u, 256u, 16384u);
        if (in_dim <= max_shared && in_dim * sizeof(float) <= 65536u) {
            uint32_t rows_per_block = cuda_parse_u32_env_alias("DS4_CUDA_F16_DECODE_RPB",
                                                               "DS4_HIP_F16_DECODE_RPB",
                                                               16u, 1u, 32u);
            if (rows_per_block != 1u && rows_per_block != 2u && rows_per_block != 4u &&
                rows_per_block != 8u && rows_per_block != 16u && rows_per_block != 32u) {
                rows_per_block = 16u;
            }
            matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16_pair sharedx launch");
        }
    }
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(float), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}
