extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
