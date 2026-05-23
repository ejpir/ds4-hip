extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        gate_offset <= model_size && up_offset <= model_size &&
        weight_bytes <= model_size - gate_offset &&
        weight_bytes <= model_size - up_offset &&
        !cuda_runtime_config()->disable_shared_gate_up_fused_w32) {
        const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8");
        const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8");
        if (!wg || !wu) return 0;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        if (getenv("DS4_CUDA_NO_OLDHIP_SHARED_GATE_UP_ROWS") == NULL) {
            const unsigned rows_per_block = 32u;
            shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    rows_per_block * 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    reinterpret_cast<const unsigned char *>(wg),
                    reinterpret_cast<const unsigned char *>(wu),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    row_bytes,
                    store_gate_up);
        } else {
            shared_gate_up_swiglu_q8_0_w32_kernel<<<(unsigned)out_dim, 32>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    reinterpret_cast<const unsigned char *>(wg),
                    reinterpret_cast<const unsigned char *>(wu),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    row_bytes,
                    store_gate_up);
        }
        return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 launch");
    }
    if (!cuda_runtime_config()->disable_shared_gate_up_pair) {
        if (cuda_runtime_config()->q8_prequant_decode &&
            !cuda_runtime_config()->disable_shared_gate_up_pair_swiglu &&
            gate && up && mid && x && out_dim <= UINT32_MAX &&
            gate_offset <= model_size && up_offset <= model_size &&
            weight_bytes <= model_size - gate_offset &&
            weight_bytes <= model_size - up_offset &&
            x->bytes >= in_dim * sizeof(float) &&
            gate->bytes >= out_dim * sizeof(float) &&
            up->bytes >= out_dim * sizeof(float) &&
            mid->bytes >= out_dim * sizeof(float)) {
            const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_pair");
            const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_pair");
            if (!wg || !wu) return 0;
            const uint64_t xq_bytes = blocks * 32u;
            const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
            const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
            void *tmp = cuda_tmp_alloc(tmp_bytes, "shared gate/up pair prequant swiglu");
            if (!tmp) return 0;
            int8_t *xq = (int8_t *)tmp;
            float *xscale = (float *)((char *)tmp + scale_offset);
            const int use_dp4a = cuda_q8_use_dp4a();
            dim3 qgrid((unsigned)blocks, 1, 1);
            quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
            if (!cuda_ok(cudaGetLastError(), "shared gate/up pair quantize launch")) return 0;
            const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
            shared_gate_up_swiglu_q8_0_pair_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    reinterpret_cast<const unsigned char *>(wg),
                    reinterpret_cast<const unsigned char *>(wu),
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    blocks,
                    use_dp4a,
                    store_gate_up);
            return cuda_ok(cudaGetLastError(), "shared gate/up pair prequant swiglu launch");
        }
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, 0.0f, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, 0.0f, 1.0f);
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        gate->bytes < n_tok * out_dim * sizeof(float) ||
        up->bytes < n_tok * out_dim * sizeof(float) ||
        mid->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

    uint32_t rows_per_block = cuda_parse_u32_env_alias("DS4_CUDA_SHARED_GATE_UP_BATCH_RPB", "DS4_HIP_SHARED_GATE_UP_BATCH_RPB", 32u, 1u, 32u);
    if (rows_per_block == 0u) rows_per_block = 32u;
    const uint32_t tile = cuda_q8_tile_env("DS4_CUDA_SHARED_GATE_UP_BATCH_TILE", "DS4_HIP_SHARED_GATE_UP_BATCH_TILE", 16u);
    const uint32_t block_tile = cuda_q8_block_tile_env("DS4_CUDA_SHARED_GATE_UP_BATCH_SHARED_X_BLOCKS", "DS4_HIP_SHARED_GATE_UP_BATCH_SHARED_X_BLOCKS", 16u, tile);
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up)
    if (tile == 8u) {
        if (block_tile == 8u) { DS4_LAUNCH_SHARED_GU_BATCH(8u, 8u); }
        else if (block_tile == 32u) { DS4_LAUNCH_SHARED_GU_BATCH(8u, 32u); }
        else { DS4_LAUNCH_SHARED_GU_BATCH(8u, 16u); }
    } else if (tile == 32u) {
        if (block_tile == 8u) { DS4_LAUNCH_SHARED_GU_BATCH(32u, 8u); }
        else if (block_tile == 32u) { DS4_LAUNCH_SHARED_GU_BATCH(32u, 32u); }
        else { DS4_LAUNCH_SHARED_GU_BATCH(32u, 16u); }
    } else {
        if (block_tile == 8u) { DS4_LAUNCH_SHARED_GU_BATCH(16u, 8u); }
        else if (block_tile == 32u) { DS4_LAUNCH_SHARED_GU_BATCH(16u, 32u); }
        else { DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u); }
    }
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 batch launch");
}
