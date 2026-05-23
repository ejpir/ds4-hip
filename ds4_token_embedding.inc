/* Token embedding row loaders.  Kept out of ds4.c so GGUF quantization
 * support for model variants does not conflict with upstream core edits. */

static void dequantize_q8_0_row(float *out, const uint8_t *row, uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const float d = f16_to_f32(scale_bits);
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t n_this = n - i0 < 32u ? n - i0 : 32u;
        for (uint64_t i = 0; i < n_this; i++) out[i0 + i] = d * (float)qs[i];
    }
}

/* Load one token embedding row and expand it to float activations. */
static void embed_token_dense(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    const uint64_t stride = te->dim[0];
    if (te->type == DS4_TENSOR_F16) {
        const uint16_t *base = tensor_data(m, te);
        const uint16_t *row = base + (uint64_t)token * stride;
        for (uint64_t i = 0; i < stride; i++) out[i] = f16_to_f32(row[i]);
        return;
    }
    if (te->type == DS4_TENSOR_Q8_0) {
        const uint64_t row_bytes = ((stride + 31u) / 32u) * 34u;
        const uint8_t *base = tensor_data(m, te);
        dequantize_q8_0_row(out, base + (uint64_t)token * row_bytes, stride);
        return;
    }
    if (te->type == DS4_TENSOR_F32) {
        const float *base = tensor_data(m, te);
        memcpy(out, base + (uint64_t)token * stride, (size_t)stride * sizeof(out[0]));
        return;
    }
    ds4_die("unsupported token embedding tensor type");
}

