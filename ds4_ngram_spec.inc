/* Optional n-gram speculative decode verifier.
 * Included from ds4.c next to the normal decode entrypoints so this
 * experimental/opt-in path can evolve without churning core decode. */

#ifndef DS4_NO_METAL
static int ds4_env_int_clamped(const char *name, int def, int lo, int hi) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int)v;
}

static int ds4_ngram_find_draft(const token_vec *hist,
                                int min_match,
                                int max_match,
                                int max_draft,
                                int eos_token,
                                int *drafts,
                                int *match_len_out,
                                int *source_pos_out) {
    if (!hist || !drafts || hist->len <= min_match || max_draft <= 0) return 0;
    if (max_match > hist->len - 1) max_match = hist->len - 1;
    if (max_match < min_match) return 0;

    for (int ml = max_match; ml >= min_match; ml--) {
        const int suffix = hist->len - ml;
        for (int pos = suffix - 1; pos >= 0; pos--) {
            if (pos + ml >= hist->len) continue;
            if (memcmp(hist->v + pos, hist->v + suffix, (size_t)ml * sizeof(hist->v[0])) != 0) continue;
            int n = hist->len - (pos + ml);
            if (n > max_draft) n = max_draft;
            if (n <= 0) continue;
            for (int i = 0; i < n; i++) {
                drafts[i] = hist->v[pos + ml + i];
                if (drafts[i] == eos_token) {
                    n = i + 1;
                    break;
                }
            }
            if (match_len_out) *match_len_out = ml;
            if (source_pos_out) *source_pos_out = pos;
            return n;
        }
    }
    return 0;
}
#endif

int ds4_session_eval_ngram_speculative_argmax(ds4_session *s, int first_token,
                                              int max_tokens, int eos_token,
                                              int *accepted, int accepted_cap,
                                              char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "Metal support is not compiled in");
    return -1;
#else
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    ds4_engine *e = s->engine;

    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap) return n_accept;

    /* --quality keeps the exact one-token target stream.  The n-gram path uses
     * the existing tiny-batch verifier/commit machinery; it is safe-stateful
     * (rejects restore the target frontier), but batch reductions can still
     * perturb near-tie greedy decisions, so keep it opt-in and non-quality. */
    if (e->quality || getenv("DS4_SPEC_NGRAM_STRICT") != NULL) return n_accept;
    if (s->ngram_spec_cooldown > 0) {
        s->ngram_spec_cooldown--;
        return n_accept;
    }
    bool have_spec_scratch = s->graph.spec_logits != NULL;
    for (uint32_t il = 0; have_spec_scratch && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        have_spec_scratch = s->graph.spec_attn_state_kv[il] &&
                            s->graph.spec_attn_state_score[il];
        if (have_spec_scratch && ratio == 4) {
            have_spec_scratch = s->graph.spec_index_state_kv[il] &&
                                s->graph.spec_index_state_score[il];
        }
    }
    if (!have_spec_scratch) return n_accept;

    int max_draft = ds4_env_int_clamped("DS4_SPEC_MAX_DRAFT", 16, 1, 16);
    max_draft = ds4_env_int_clamped("DS4_SPEC_NGRAM_MAX_DRAFT", max_draft, 1, 16);
    int min_draft = ds4_env_int_clamped("DS4_SPEC_MIN_DRAFT", 8, 1, 16);
    min_draft = ds4_env_int_clamped("DS4_SPEC_NGRAM_MIN_DRAFT", min_draft, 1, 16);
    int min_match = ds4_env_int_clamped("DS4_SPEC_NGRAM_MIN_MATCH", 4, 1, 128);
    int max_match = ds4_env_int_clamped("DS4_SPEC_NGRAM_MAX_MATCH", 64, min_match, 512);
    const int reject_cooldown = ds4_env_int_clamped("DS4_SPEC_NGRAM_REJECT_COOLDOWN", 128, 0, 4096);

    if (max_draft > max_tokens - n_accept) max_draft = max_tokens - n_accept;
    if (max_draft > accepted_cap - n_accept) max_draft = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (max_draft > room - 1) max_draft = room - 1;
    if (max_draft <= 0) return n_accept;

    int drafts[16];
    int match_len = 0;
    int source_pos = -1;
    int draft_n = ds4_ngram_find_draft(&s->checkpoint,
                                       min_match,
                                       max_match,
                                       max_draft,
                                       eos_token,
                                       drafts,
                                       &match_len,
                                       &source_pos);
    if (draft_n < min_draft) return n_accept;

    const bool log_ngram = getenv("DS4_SPEC_NGRAM_LOG") != NULL;
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (reject_cooldown > 0) s->ngram_spec_cooldown = (uint32_t)reject_cooldown;
        if (log_ngram) {
            fprintf(stderr,
                    "ds4: ngram spec reject-first match=%d source=%d drafted=%d draft0=%d target=%d cooldown=%d\n",
                    match_len,
                    source_pos,
                    draft_n,
                    drafts[0],
                    sample_argmax(s->logits, DS4_N_VOCAB),
                    reject_cooldown);
        }
        return n_accept;
    }

    ds4_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    const int start = s->checkpoint.len;
    bool have_frontier = spec_frontier_snapshot(&frontier, s);
    if (!have_frontier) {
        if (log_ngram) fprintf(stderr, "ds4: ngram spec snapshot failed\n");
        return n_accept;
    }

    int row_tops[16] = {0};
    for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
    bool ok = metal_graph_verify_suffix_tops(&s->graph,
                                             &e->model,
                                             &e->weights,
                                             &s->checkpoint,
                                             (uint32_t)start,
                                             (uint32_t)draft_n,
                                             false,
                                             row_tops,
                                             NULL);
    int commit_drafts = ok ? 1 : 0;
    if (ok) {
        for (int i = 1; i < draft_n; i++) {
            if (row_tops[i - 1] != drafts[i]) break;
            commit_drafts++;
        }
    }

    if (ok && commit_drafts == draft_n) {
        ok = metal_graph_read_spec_logits_row(&s->graph,
                                              (uint32_t)(draft_n - 1),
                                              s->logits);
        if (ok) {
            for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                accepted[n_accept++] = drafts[i];
                if (drafts[i] == eos_token) break;
            }
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            s->ngram_spec_cooldown = 0;
            if (log_ngram) {
                fprintf(stderr,
                        "ds4: ngram spec accept match=%d source=%d drafted=%d accepted=%d\n",
                        match_len,
                        source_pos,
                        draft_n,
                        draft_n);
            }
            spec_frontier_free(&frontier);
            return n_accept;
        }
    }

    s->checkpoint.len = start;
    (void)spec_frontier_restore(&frontier, s);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    if (reject_cooldown > 0) s->ngram_spec_cooldown = (uint32_t)reject_cooldown;
    if (log_ngram) {
        fprintf(stderr,
                "ds4: ngram spec reject match=%d source=%d drafted=%d verified_prefix=%d ok=%d cooldown=%d\n",
                match_len,
                source_pos,
                draft_n,
                commit_drafts,
                ok ? 1 : 0,
                reject_cooldown);
    }
    spec_frontier_free(&frontier);
    return n_accept;
#endif
}

