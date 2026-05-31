// Tiny ROCm GPU API smoke test for validation-sensitive helpers.
//
// Build after ds4_rocm.o exists:
//   cc -O2 -I. -c tools/rocm_api_smoke.c -o /tmp/rocm_api_smoke.o
//   /opt/rocm/bin/hipcc /tmp/rocm_api_smoke.o ds4_rocm.o \
//       -lm -pthread -L/opt/rocm/lib -lhipblas -lhipblaslt \
//       -o /tmp/rocm_api_smoke
// Run on an idle ROCm GPU:
//   /tmp/rocm_api_smoke

#include "../ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *msg) {
    fprintf(stderr, "rocm_api_smoke: %s\n", msg);
    ds4_gpu_cleanup();
    exit(1);
}

static void expect(int cond, const char *msg) {
    if (!cond) fail(msg);
}

static void smoke_argmax(void) {
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(8u * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(int32_t));
    expect(logits && out, "argmax alloc failed");

    const float h_logits[8] = {-2.0f, 1.0f, 5.0f, 7.0f, 3.0f, 7.0f, -1.0f, 0.0f};
    int32_t h_idx = -1;
    expect(ds4_gpu_tensor_write(logits, 0, h_logits, sizeof(h_logits)), "argmax logits write failed");
    expect(ds4_gpu_argmax_tensor(out, logits, 8), "argmax launch failed");
    expect(ds4_gpu_synchronize(), "argmax synchronize failed");
    expect(ds4_gpu_tensor_read(out, 0, &h_idx, sizeof(h_idx)), "argmax read failed");
    expect(h_idx == 3, "argmax tie-break changed");

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(logits);
}

static void smoke_directional_zero_scale(void) {
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(4u * sizeof(float));
    ds4_gpu_tensor *directions = ds4_gpu_tensor_alloc(4u * sizeof(float));
    expect(x && directions, "directional alloc failed");

    const float h_x[4] = {1.0f, -2.0f, 3.5f, 4.25f};
    const float h_dir[4] = {10.0f, 20.0f, -30.0f, 40.0f};
    float h_out[4] = {0};
    expect(ds4_gpu_tensor_write(x, 0, h_x, sizeof(h_x)), "directional x write failed");
    expect(ds4_gpu_tensor_write(directions, 0, h_dir, sizeof(h_dir)), "directional dir write failed");
    expect(ds4_gpu_directional_steering_project_tensor(x, directions, 0, 4, 1, 0.0f),
           "zero-scale directional steering should be a no-op success");
    expect(ds4_gpu_synchronize(), "directional synchronize failed");
    expect(ds4_gpu_tensor_read(x, 0, h_out, sizeof(h_out)), "directional read failed");
    expect(memcmp(h_x, h_out, sizeof(h_x)) == 0, "zero-scale directional steering changed x");

    ds4_gpu_tensor_free(directions);
    ds4_gpu_tensor_free(x);
}

static void smoke_ds4_shape_rejects(void) {
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(6u * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(6u * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(256u * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(256u * sizeof(float));
    expect(selected && weights && probs && logits, "router alloc failed");

    float h_logits[256];
    for (uint32_t i = 0; i < 256u; i++) h_logits[i] = (float)i;
    expect(ds4_gpu_tensor_write(logits, 0, h_logits, sizeof(h_logits)), "router logits write failed");

    const uint8_t fake_model[1024] = {0};
    int ok = ds4_gpu_router_select_tensor(selected, weights, probs,
                                          fake_model, sizeof(fake_model),
                                          0, 0, 0, 0,
                                          128, 6, 1.5f,
                                          1, 0, false, false, logits);
    expect(!ok, "router accepted non-DS4 expert count");

    ok = ds4_gpu_routed_moe_one_tensor(NULL, NULL, NULL, NULL, NULL,
                                       fake_model, sizeof(fake_model),
                                       0, 0, 0, 10, 10,
                                       0, 0, 0, 0,
                                       256, 256, 256,
                                       selected, weights,
                                       128, 6, 0.0f, logits);
    expect(!ok, "MoE accepted non-DS4 total expert count");

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
}

int main(void) {
    expect(ds4_gpu_init(), "ds4_gpu_init failed");
    smoke_argmax();
    smoke_directional_zero_scale();
    smoke_ds4_shape_rejects();
    ds4_gpu_cleanup();
    puts("rocm_api_smoke: ok");
    return 0;
}
