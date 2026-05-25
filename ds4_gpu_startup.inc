/* GPU/ROCm startup diagnostics and optional cache preload helpers.
 * Included by ds4.c after GGUF/tensor helpers are defined so
 * backend-specific startup policy stays out of the main engine body. */

static bool ds4_str_has_cstr(ds4_str s, const char *needle) {
    if (!needle || !needle[0]) return true;
    const size_t n = strlen(needle);
    if (s.len < n) return false;
    for (uint64_t i = 0; i + n <= s.len; i++) {
        if (memcmp(s.ptr + i, needle, n) == 0) return true;
    }
    return false;
}

static uint64_t g_startup_model_file_bytes;
static uint64_t g_startup_tensor_bytes;
static uint64_t g_startup_q8_f16_original_bytes;
static uint64_t g_startup_q8_f16_expanded_bytes;

static void tensor_stat_add(uint64_t *count, uint64_t *bytes, const ds4_tensor *t) {
    if (count) (*count)++;
    if (bytes) *bytes += t ? t->bytes : 0;
}

static void model_startup_tensor_summary(const ds4_model *m) {
    if (!m) return;
    enum { TYPE_BUCKETS = 64 };
    uint64_t type_count[TYPE_BUCKETS] = {0};
    uint64_t type_bytes[TYPE_BUCKETS] = {0};
    uint64_t tensor_bytes = 0;
    uint64_t routed_gate_n = 0, routed_gate_b = 0;
    uint64_t routed_up_n = 0, routed_up_b = 0;
    uint64_t routed_down_n = 0, routed_down_b = 0;
    uint64_t shared_gate_n = 0, shared_gate_b = 0;
    uint64_t shared_up_n = 0, shared_up_b = 0;
    uint64_t shared_down_n = 0, shared_down_b = 0;
    uint64_t attn_n = 0, attn_b = 0;
    uint64_t indexer_n = 0, indexer_b = 0;
    uint64_t emb_out_n = 0, emb_out_b = 0;
    uint64_t hc_n = 0, hc_b = 0;
    uint64_t accounted_n = 0, accounted_b = 0;

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        tensor_bytes += t->bytes;
        if (t->type < TYPE_BUCKETS) {
            type_count[t->type]++;
            type_bytes[t->type] += t->bytes;
        }

        bool accounted = false;
        if (ds4_str_has_cstr(t->name, "ffn_gate_exps.weight")) {
            tensor_stat_add(&routed_gate_n, &routed_gate_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "ffn_up_exps.weight")) {
            tensor_stat_add(&routed_up_n, &routed_up_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "ffn_down_exps.weight")) {
            tensor_stat_add(&routed_down_n, &routed_down_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "ffn_gate_shexp.weight")) {
            tensor_stat_add(&shared_gate_n, &shared_gate_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "ffn_up_shexp.weight")) {
            tensor_stat_add(&shared_up_n, &shared_up_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "ffn_down_shexp.weight")) {
            tensor_stat_add(&shared_down_n, &shared_down_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "token_embd") ||
                   ds4_str_has_cstr(t->name, "output.weight") ||
                   ds4_str_has_cstr(t->name, "output_norm")) {
            tensor_stat_add(&emb_out_n, &emb_out_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "indexer_")) {
            tensor_stat_add(&indexer_n, &indexer_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "attn_")) {
            tensor_stat_add(&attn_n, &attn_b, t);
            accounted = true;
        } else if (ds4_str_has_cstr(t->name, "hc_") ||
                   ds4_str_has_cstr(t->name, "_hc_")) {
            tensor_stat_add(&hc_n, &hc_b, t);
            accounted = true;
        }
        if (accounted) tensor_stat_add(&accounted_n, &accounted_b, t);
    }

    g_startup_model_file_bytes = m->size;
    g_startup_tensor_bytes = tensor_bytes;

    fprintf(stderr, "ds4: GGUF tensor inventory: total=%" PRIu64 " tensors, ", m->n_tensors);
    fprint_size(stderr, tensor_bytes);
    fprintf(stderr, " described (file=");
    fprint_size(stderr, m->size);
    fprintf(stderr, ", tensor_data_offset=");
    fprint_size(stderr, m->tensor_data_pos);
    fprintf(stderr, ")\n");

    fprintf(stderr, "ds4: GGUF tensor types:");
    bool any_type = false;
    const uint32_t type_n = (uint32_t)(sizeof(gguf_types) / sizeof(gguf_types[0]));
    for (uint32_t type = 0; type < type_n && type < TYPE_BUCKETS; type++) {
        if (type_count[type] == 0) continue;
        fprintf(stderr, "%s %s=%" PRIu64 "/", any_type ? "," : "", tensor_type_name(type), type_count[type]);
        fprint_size(stderr, type_bytes[type]);
        any_type = true;
    }
    if (!any_type) fprintf(stderr, " none");
    fprintf(stderr, "\n");

    const uint64_t routed_n = routed_gate_n + routed_up_n + routed_down_n;
    const uint64_t routed_b = routed_gate_b + routed_up_b + routed_down_b;
    const uint64_t shared_n = shared_gate_n + shared_up_n + shared_down_n;
    const uint64_t shared_b = shared_gate_b + shared_up_b + shared_down_b;
    const uint64_t other_n = accounted_n <= m->n_tensors ? m->n_tensors - accounted_n : 0;
    const uint64_t other_b = accounted_b <= tensor_bytes ? tensor_bytes - accounted_b : 0;
    fprintf(stderr, "ds4: GGUF tensor groups: routed_moe=%" PRIu64 "/", routed_n);
    fprint_size(stderr, routed_b);
    fprintf(stderr, " (gate=%" PRIu64 "/", routed_gate_n);
    fprint_size(stderr, routed_gate_b);
    fprintf(stderr, ", up=%" PRIu64 "/", routed_up_n);
    fprint_size(stderr, routed_up_b);
    fprintf(stderr, ", down=%" PRIu64 "/", routed_down_n);
    fprint_size(stderr, routed_down_b);
    fprintf(stderr, "), shared=%" PRIu64 "/", shared_n);
    fprint_size(stderr, shared_b);
    fprintf(stderr, " (gate=%" PRIu64 "/", shared_gate_n);
    fprint_size(stderr, shared_gate_b);
    fprintf(stderr, ", up=%" PRIu64 "/", shared_up_n);
    fprint_size(stderr, shared_up_b);
    fprintf(stderr, ", down=%" PRIu64 "/", shared_down_n);
    fprint_size(stderr, shared_down_b);
    fprintf(stderr, "), attention=%" PRIu64 "/", attn_n);
    fprint_size(stderr, attn_b);
    fprintf(stderr, ", indexer=%" PRIu64 "/", indexer_n);
    fprint_size(stderr, indexer_b);
    fprintf(stderr, ", emb_output=%" PRIu64 "/", emb_out_n);
    fprint_size(stderr, emb_out_b);
    fprintf(stderr, ", hc=%" PRIu64 "/", hc_n);
    fprint_size(stderr, hc_b);
    fprintf(stderr, ", other=%" PRIu64 "/", other_n);
    fprint_size(stderr, other_b);
    fprintf(stderr, "\n");
}

static bool env_truthy(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static bool env_present_nonempty(const char *name) {
    const char *v = getenv(name);
    return v && v[0];
}

static const char *onoff(bool v) {
    return v ? "on" : "off";
}

static bool env_truthy2(const char *a, const char *b) {
    return env_truthy(a) || env_truthy(b);
}

static const char *env_first_nonempty(const char *a, const char *b, const char *fallback) {
    const char *v = a ? getenv(a) : NULL;
    if (v && v[0]) return v;
    v = b ? getenv(b) : NULL;
    if (v && v[0]) return v;
    return fallback;
}

static void gpu_fast_path_summary(bool quality) {
    const bool copy_model = env_truthy("DS4_CUDA_COPY_MODEL") ||
                            env_truthy("DS4_CUDA_COPY_MODEL_CHUNKED") ||
                            env_truthy("DS4_CUDA_COPY_MODEL_STAGED");
    const bool q8_prequant = env_truthy("DS4_CUDA_Q8_PREQUANT_DECODE");
    const bool splitk_attn_out_low = !env_truthy("DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW");
    const bool q_head_rope_fused = !env_present_nonempty("DS4_CUDA_NO_HEAD_RMS_ROPE_FUSED") &&
                                   !env_present_nonempty("DS4_HIP_NO_HEAD_RMS_ROPE_FUSED") &&
                                   !env_present_nonempty("DS4_METAL_GRAPH_DUMP_PREFIX");
    const bool qkv_pair_decode = env_truthy2("DS4_CUDA_QKV_PAIR_DECODE", "DS4_HIP_QKV_PAIR_DECODE") &&
                                 !env_present_nonempty("DS4_METAL_GRAPH_DUMP_PREFIX");
    const char *q8_decode_rpb = env_first_nonempty("DS4_CUDA_Q8_DECODE_RPB",
                                                   "DS4_HIP_Q8_DECODE_RPB",
                                                   "default");
    const bool oldhip_attention_decode = env_truthy2("DS4_CUDA_OLDHIP_ATTENTION_DECODE",
                                                     "DS4_HIP_OLDHIP_ATTENTION_DECODE");
    const bool oldhip_attention_vec4 = oldhip_attention_decode &&
                                       !env_truthy2("DS4_CUDA_NO_OLDHIP_ATTENTION_VEC4",
                                                    "DS4_HIP_NO_OLDHIP_ATTENTION_VEC4");
    const bool shared_gate_fused = !env_truthy("DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32");
    const bool shared_gate_pair_swiglu = q8_prequant && !shared_gate_fused &&
                                        !env_truthy("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") &&
                                        !env_truthy("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR_SWIGLU");
    const char *moe_decode_rpb = env_first_nonempty("DS4_CUDA_MOE_DECODE_RPB",
                                                    "DS4_HIP_MOE_DECODE_RPB",
                                                    "default");
    const bool attn_out_preload = env_truthy("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") ||
                                  env_truthy("DS4_CUDA_ATTENTION_OUTPUT_CUBLAS") ||
                                  env_truthy("DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL");
    const bool attn_q_b_preload = env_truthy("DS4_CUDA_ATTN_Q_B_PRELOAD") ||
                                  env_truthy("DS4_HIP_ATTN_Q_B_PRELOAD");
    const bool shared_down_preload = env_truthy("DS4_CUDA_SHARED_DOWN_PRELOAD") ||
                                     env_truthy("DS4_CUDA_SHARED_DOWN_CUBLAS");
    fprintf(stderr,
            "ds4: ROCm fast paths: quality=%s copy_model=%s q8_prequant_decode=%s "
            "q8_decode_rpb=%s q_head_rope_fused=%s qkv_pair_decode=%s "
            "oldhip_attention_decode=%s oldhip_attention_vec4=%s "
            "attn_out_low_splitk=%s attn_out_low_prequant=%s "
            "shared_gate_up_fused_w32=%s shared_gate_up_prequant_pair=%s "
            "shared_gate_up_pair_swiglu=%s moe_decode_rpb=%s "
            "q8_fp16_preload(attn_out=%s,attn_q_b=%s,shared_down=%s)\n",
            onoff(quality),
            onoff(copy_model),
            onoff(q8_prequant),
            q8_decode_rpb,
            onoff(q_head_rope_fused),
            onoff(qkv_pair_decode),
            onoff(oldhip_attention_decode),
            onoff(oldhip_attention_vec4),
            onoff(splitk_attn_out_low),
            onoff(q8_prequant && !splitk_attn_out_low),
            onoff(shared_gate_fused),
            onoff(q8_prequant && !shared_gate_fused),
            onoff(shared_gate_pair_swiglu),
            moe_decode_rpb,
            onoff(attn_out_preload),
            onoff(attn_q_b_preload),
            onoff(shared_down_preload));
    fprintf(stderr,
            "ds4: ROCm optional paths: prefill_raw_fast=%s prefill_mixed_fast=%s "
            "indexed_heads32=%s q8_batch_fast=%s q8_wmma_fast=%s "
            "moe_wmma_hot=%s attn_q_b_cublas=%s attn_output_cublas_all=%s "
            "shared_down_cublas=%s shared_gate_up_batch_fused=%s\n",
            onoff(env_truthy2("DS4_CUDA_PREFILL_RAW_FAST", "DS4_HIP_PREFILL_RAW_FAST")),
            onoff(env_truthy2("DS4_CUDA_PREFILL_MIXED_FAST", "DS4_HIP_PREFILL_MIXED_FAST")),
            onoff(env_truthy2("DS4_CUDA_INDEXED_HEADS32", "DS4_HIP_INDEXED_HEADS32")),
            onoff(env_truthy2("DS4_CUDA_Q8_BATCH_FAST", "DS4_HIP_Q8_BATCH_FAST")),
            onoff(env_truthy2("DS4_CUDA_Q8_WMMA_FAST", "DS4_HIP_Q8_WMMA_FAST")),
            onoff(env_truthy2("DS4_CUDA_MOE_WMMA_HOT", "DS4_HIP_MOE_WMMA_HOT")),
            onoff(env_truthy2("DS4_CUDA_ATTN_Q_B_CUBLAS", "DS4_HIP_ATTN_Q_B_CUBLAS")),
            onoff(env_truthy2("DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL", "DS4_HIP_ATTENTION_OUTPUT_CUBLAS_ALL")),
            onoff(env_truthy("DS4_CUDA_SHARED_DOWN_CUBLAS")),
            onoff(env_truthy("DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED")));
}

static void gpu_known_footprint_report(uint64_t context_bytes) {
    const bool copy_model = env_truthy("DS4_CUDA_COPY_MODEL") ||
                            env_truthy("DS4_CUDA_COPY_MODEL_CHUNKED") ||
                            env_truthy("DS4_CUDA_COPY_MODEL_STAGED");
    const uint64_t model_resident = copy_model ? g_startup_model_file_bytes : 0;
    const uint64_t total = model_resident + g_startup_q8_f16_expanded_bytes + context_bytes;
    fprintf(stderr, "ds4: CUDA known resident footprint: model_image=");
    fprint_size(stderr, model_resident);
    if (!copy_model) fprintf(stderr, " (not fully copied)");
    fprintf(stderr, " q8_f16_cache=");
    fprint_size(stderr, g_startup_q8_f16_expanded_bytes);
    fprintf(stderr, " context=");
    fprint_size(stderr, context_bytes);
    fprintf(stderr, " total_known=");
    fprint_size(stderr, total);
    fprintf(stderr, "\n");
}

#ifdef DS4_USE_GPU_API
static bool ds4_str_contains_cstr(ds4_str s, const char *needle) {
    if (!needle || !needle[0]) return true;
    const size_t n = strlen(needle);
    if (s.len < n) return false;
    for (uint64_t i = 0; i + n <= s.len; i++) {
        if (memcmp(s.ptr + i, needle, n) == 0) return true;
    }
    return false;
}

static bool accelerator_cuda_preload_q8_f16_tensors(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return false;
    const bool preload_all = getenv("DS4_CUDA_Q8_F16_PRELOAD") != NULL ||
                             getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL;
    const bool preload_attn_out = getenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") != NULL ||
                                  getenv("DS4_CUDA_ATTENTION_OUTPUT_CUBLAS") != NULL ||
                                  getenv("DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL") != NULL;
    const bool preload_q_b = getenv("DS4_CUDA_ATTN_Q_B_PRELOAD") != NULL ||
                             getenv("DS4_HIP_ATTN_Q_B_PRELOAD") != NULL;
    const bool preload_shared = getenv("DS4_CUDA_SHARED_EXPERT_CUBLAS") != NULL ||
                                getenv("DS4_CUDA_SHARED_EXPERT_PRELOAD") != NULL;
    const bool preload_shared_down = getenv("DS4_CUDA_SHARED_DOWN_CUBLAS") != NULL ||
                                     getenv("DS4_CUDA_SHARED_DOWN_PRELOAD") != NULL;
    if (!preload_all && !preload_attn_out && !preload_q_b && !preload_shared && !preload_shared_down) return true;

    const double t0 = now_sec();
    uint64_t requested = 0;
    uint64_t original_bytes = 0;
    uint64_t expanded_bytes = 0;
    uint64_t attn_out_n = 0, attn_out_original = 0, attn_out_expanded = 0;
    uint64_t q_b_n = 0, q_b_original = 0, q_b_expanded = 0;
    uint64_t shared_down_n = 0, shared_down_original = 0, shared_down_expanded = 0;
    uint64_t shared_gate_up_n = 0, shared_gate_up_original = 0, shared_gate_up_expanded = 0;
    uint64_t other_n = 0, other_original = 0, other_expanded = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->type != DS4_TENSOR_Q8_0 || t->ndim != 2 || t->bytes == 0) continue;
        const bool is_attn_out = ds4_str_contains_cstr(t->name, "attn_output_a.weight") ||
                                 ds4_str_contains_cstr(t->name, "attn_output_b.weight");
        const bool is_q_b = ds4_str_contains_cstr(t->name, "attn_q_b.weight");
        const bool is_shared_down = ds4_str_contains_cstr(t->name, "ffn_down_shexp.weight");
        const bool is_shared_gate_up = ds4_str_contains_cstr(t->name, "ffn_gate_shexp.weight") ||
                                       ds4_str_contains_cstr(t->name, "ffn_up_shexp.weight");
        const bool is_shared = is_shared_gate_up || is_shared_down;
        if (!preload_all && !(preload_attn_out && is_attn_out) && !(preload_q_b && is_q_b) &&
            !(preload_shared && is_shared) && !(preload_shared_down && is_shared_down)) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) return false;
        char label[160];
        snprintf(label, sizeof(label), "tensor:%.*s", (int)t->name.len, t->name.ptr);
        if (ds4_gpu_cache_q8_f16_range(m->map, m->size, t->abs_offset, t->bytes, t->dim[0], t->dim[1], label) == 0) {
            fprintf(stderr, "ds4: CUDA failed to preload dequantized Q8 tensor %.*s\n",
                    (int)t->name.len, t->name.ptr);
            return false;
        }
        requested++;
        original_bytes += t->bytes;
        uint64_t expanded = 0;
        if (t->dim[0] != 0 && t->dim[1] <= UINT64_MAX / t->dim[0] / sizeof(uint16_t)) {
            expanded = t->dim[0] * t->dim[1] * sizeof(uint16_t);
            expanded_bytes += expanded;
        }
        if (is_attn_out) {
            tensor_stat_add(&attn_out_n, &attn_out_original, t);
            attn_out_expanded += expanded;
        } else if (is_q_b) {
            tensor_stat_add(&q_b_n, &q_b_original, t);
            q_b_expanded += expanded;
        } else if (is_shared_down) {
            tensor_stat_add(&shared_down_n, &shared_down_original, t);
            shared_down_expanded += expanded;
        } else if (is_shared_gate_up) {
            tensor_stat_add(&shared_gate_up_n, &shared_gate_up_original, t);
            shared_gate_up_expanded += expanded;
        } else {
            tensor_stat_add(&other_n, &other_original, t);
            other_expanded += expanded;
        }
    }
    g_startup_q8_f16_original_bytes = original_bytes;
    g_startup_q8_f16_expanded_bytes = expanded_bytes;
    if (requested != 0 || getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) {
        fprintf(stderr,
                "ds4: CUDA requested Q8 fp16 preload tensors=%" PRIu64 " original=",
                requested);
        fprint_size(stderr, original_bytes);
        fprintf(stderr, " expanded=");
        fprint_size(stderr, expanded_bytes);
        fprintf(stderr, " in %.3fs\n", now_sec() - t0);
        if (requested != 0) {
            fprintf(stderr, "ds4: CUDA Q8 fp16 preload groups: attn_out=%" PRIu64 "/", attn_out_n);
            fprint_size(stderr, attn_out_original);
            fprintf(stderr, "->");
            fprint_size(stderr, attn_out_expanded);
            fprintf(stderr, ", attn_q_b=%" PRIu64 "/", q_b_n);
            fprint_size(stderr, q_b_original);
            fprintf(stderr, "->");
            fprint_size(stderr, q_b_expanded);
            fprintf(stderr, ", shared_down=%" PRIu64 "/", shared_down_n);
            fprint_size(stderr, shared_down_original);
            fprintf(stderr, "->");
            fprint_size(stderr, shared_down_expanded);
            fprintf(stderr, ", shared_gate_up=%" PRIu64 "/", shared_gate_up_n);
            fprint_size(stderr, shared_gate_up_original);
            fprintf(stderr, "->");
            fprint_size(stderr, shared_gate_up_expanded);
            fprintf(stderr, ", other=%" PRIu64 "/", other_n);
            fprint_size(stderr, other_original);
            fprintf(stderr, "->");
            fprint_size(stderr, other_expanded);
            fprintf(stderr, "\n");
        }
    }
    return true;
}
#endif

