/* GPU backend environment aliases.
 *
 * ds4.c keeps the graph decisions backend-neutral.  CUDA-named knobs remain the
 * canonical names because the GPU ABI is CUDA-shaped; HIP/ROCm aliases are
 * collected here so ROCm-specific spelling does not leak through the graph
 * implementation.
 */

static DS4_MAYBE_UNUSED bool ds4_gpu_env_present(const char *name) {
    const char *s = getenv(name);
    return s && s[0];
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_flag_enabled(const char *name) {
    const char *s = getenv(name);
    if (!s || !s[0]) return false;
    return strcmp(s, "0") != 0 && strcmp(s, "false") != 0 &&
           strcmp(s, "off") != 0 && strcmp(s, "no") != 0;
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_flag_enabled2(const char *name0, const char *name1) {
    return ds4_gpu_env_flag_enabled(name0) || ds4_gpu_env_flag_enabled(name1);
}

static DS4_MAYBE_UNUSED uint32_t ds4_gpu_env_u32_default2(const char *name0, const char *name1, uint32_t def) {
    const char *s = getenv(name0);
    if ((!s || !s[0]) && name1) s = getenv(name1);
    if (!s || !s[0]) return def;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s) return def;
    if (v > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)v;
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_no_q8_token_embed(void) {
    return ds4_gpu_env_present("DS4_CUDA_NO_Q8_TOKEN_EMBED") ||
           ds4_gpu_env_present("DS4_HIP_NO_Q8_TOKEN_EMBED");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_q_head_rope_fused(void) {
    static int cache = -1;
    if (cache < 0) {
        cache = !ds4_gpu_env_present("DS4_CUDA_NO_HEAD_RMS_ROPE_FUSED") &&
                !ds4_gpu_env_present("DS4_HIP_NO_HEAD_RMS_ROPE_FUSED") &&
                !ds4_gpu_env_present("DS4_METAL_GRAPH_DUMP_PREFIX");
    }
    return cache != 0;
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_attn_q_b_f16_out(void) {
    return ds4_gpu_env_present("DS4_CUDA_ATTN_Q_B_F16_OUT") ||
           ds4_gpu_env_present("DS4_HIP_ATTN_Q_B_F16_OUT");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_qkv_pair_decode(void) {
    static int cache = -1;
    if (cache < 0) {
        cache = (ds4_gpu_env_flag_enabled2("DS4_CUDA_QKV_PAIR_DECODE", "DS4_HIP_QKV_PAIR_DECODE") &&
                 !ds4_gpu_env_present("DS4_METAL_GRAPH_DUMP_PREFIX"));
    }
    return cache != 0;
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_attn_output_f16_out(uint32_t n_tokens) {
    const uint32_t min_tokens = ds4_gpu_env_u32_default2("DS4_CUDA_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS",
                                                        "DS4_HIP_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS",
                                                        0u);
    return n_tokens >= min_tokens &&
           ds4_gpu_env_flag_enabled2("DS4_CUDA_ATTENTION_OUTPUT_F16_OUT",
                                     "DS4_HIP_ATTENTION_OUTPUT_F16_OUT") &&
           !ds4_gpu_env_present("DS4_METAL_GRAPH_DUMP_PREFIX");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_shared_gate_up_batch_fused(void) {
    return ds4_gpu_env_present("DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_shared_down_f16_out(uint32_t n_tokens, bool keep_ffn_out) {
    const uint32_t min_tokens = ds4_gpu_env_u32_default2("DS4_CUDA_SHARED_DOWN_F16_OUT_MIN_TOKENS",
                                                        "DS4_HIP_SHARED_DOWN_F16_OUT_MIN_TOKENS",
                                                        0u);
    return n_tokens >= min_tokens &&
           ds4_gpu_env_flag_enabled2("DS4_CUDA_SHARED_DOWN_F16_OUT",
                                     "DS4_HIP_SHARED_DOWN_F16_OUT") &&
           !ds4_gpu_env_present("DS4_METAL_GRAPH_DUMP_PREFIX") &&
           !keep_ffn_out;
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_prefill_all_row_head(void) {
    return ds4_gpu_env_present("DS4_METAL_PREFILL_ALL_ROW_HEAD") ||
           ds4_gpu_env_present("DS4_CUDA_PREFILL_ALL_ROW_HEAD") ||
           ds4_gpu_env_present("DS4_HIP_PREFILL_ALL_ROW_HEAD");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_prefill_all_row_topk(void) {
    return ds4_gpu_env_present("DS4_METAL_PREFILL_ALL_ROW_TOPK") ||
           ds4_gpu_env_present("DS4_CUDA_PREFILL_ALL_ROW_TOPK") ||
           ds4_gpu_env_present("DS4_HIP_PREFILL_ALL_ROW_TOPK");
}

static DS4_MAYBE_UNUSED bool ds4_gpu_env_prefill_chunk_profile(void) {
    return ds4_gpu_env_present("DS4_METAL_GRAPH_PREFILL_CHUNK_PROFILE") ||
           ds4_gpu_env_present("DS4_HIP_PREFILL_CHUNK_PROFILE");
}
