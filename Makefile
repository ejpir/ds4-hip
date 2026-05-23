CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
HIPCC ?= $(shell command -v hipcc 2>/dev/null)
HIPCXXFLAGS ?= -O3 --offload-arch=native -std=c++17 -Wno-unused-parameter -Wno-unused-function

LDLIBS ?= -lm -pthread
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)

ROCM_PATH ?= /opt/rocm
ROCM_ARCH ?= gfx1151
ROCM_HIPCC ?= $(if $(HIPCC),$(HIPCC),$(ROCM_PATH)/bin/hipcc)
ROCM_CFLAGS ?= -O3 -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -L$(ROCM_PATH)/lib -lhipblas -lhipblaslt

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
CPU_CORE_OBJS = ds4_cpu.o
LINK.ds4 = $(CC)
TEST_CORE_OBJS = $(CORE_OBJS)
TEST_CFLAGS = $(CFLAGS)
TEST_LINK = $(CC)
else
ifneq ($(HIPCC),)
CFLAGS += -D_GNU_SOURCE -DDS4_USE_HIP
CORE_OBJS = ds4.o ds4_hip.o
NATIVE_CORE_OBJS = ds4_native.o
CPU_CORE_OBJS = ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
HIP_LDLIBS := $(LDLIBS) -lhipblaslt
LINK.ds4 = $(HIPCC)
TEST_CORE_OBJS = $(NATIVE_CORE_OBJS)
TEST_CFLAGS = $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU
TEST_LINK = $(CC)
else
CFLAGS += -D_GNU_SOURCE -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
CPU_CORE_OBJS = ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
LINK.ds4 = $(CC)
TEST_CORE_OBJS = $(CORE_OBJS)
TEST_CFLAGS = $(CFLAGS)
TEST_LINK = $(CC)
endif
endif

.PHONY: all help clean test cpu rocm rocm-upstream cuda-regression

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make rocm         Build ROCm upstream-shaped binaries"
	@echo "  make rocm-upstream Build ROCm upstream-shaped binaries"
	@echo "                  (CLI, server, benchmark, eval, and agent)"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4-agent: ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))
endif

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "CUDA is not supported in this HIP/ROCm fork; use make rocm-upstream"

rocm rocm-upstream: ds4-rocm-upstream ds4-server-rocm-upstream ds4-bench-rocm-upstream ds4-eval-rocm-upstream ds4-agent-rocm-upstream
	@echo "ROCm upstream-shaped binaries built with ROCM_ARCH=$(ROCM_ARCH)"

ds4-mtp-oracle-bench-rocm-upstream: tools/mtp_oracle_microbench_gpuapi.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-rocm-upstream: ds4_cli_gpuapi.o linenoise.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-server-rocm-upstream: ds4_server_gpuapi.o ds4_kvstore.o rax.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-bench-rocm-upstream: ds4_bench_gpuapi.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-eval-rocm-upstream: ds4_eval_gpuapi.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-agent-rocm-upstream: ds4_agent_gpuapi.o ds4_kvstore.o linenoise.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4.o: ds4.c ds4.h ds4_metal.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_kvstore.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(TEST_CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h ds4_gpu.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cpu.o: ds4.c ds4.h ds4_metal.h ds4_gpu.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_kvstore.h linenoise.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_metal.h ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_hip.o: ds4_hip.cpp ds4_metal.h
	$(HIPCC) $(HIPCXXFLAGS) -c -o $@ ds4_hip.cpp

ds4_gpuapi.o: ds4.c ds4.h ds4_metal.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4.c

ds4_cli_gpuapi.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_cli.c

ds4_server_gpuapi.o: ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_server.c

ds4_bench_gpuapi.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_bench.c

ds4_eval_gpuapi.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_eval.c

ds4_agent_gpuapi.o: ds4_agent.c ds4.h ds4_kvstore.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_agent.c

tools/mtp_oracle_microbench_gpuapi.o: tools/mtp_oracle_microbench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -I. -c -o $@ tools/mtp_oracle_microbench.c

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc ds4_rocm.h rocm/ds4_rocm_runtime.cuh rocm/ds4_rocm_common.cuh rocm/ds4_rocm_q8.cuh rocm/ds4_rocm_norm_rope.cuh rocm/ds4_rocm_matmul.cuh rocm/ds4_rocm_fp8_kv.cuh rocm/ds4_rocm_attention.cuh rocm/ds4_rocm_attention_launch.cuh rocm/ds4_rocm_hc.cuh rocm/ds4_rocm_output.cuh rocm/ds4_rocm_indexer.cuh rocm/ds4_rocm_compressor.cuh rocm/ds4_rocm_shared_expert.cuh rocm/ds4_rocm_router.cuh rocm/ds4_rocm_moe.cuh rocm/ds4_rocm_hc_output_launch.cuh rocm/ds4_rocm_hipblaslt.cuh
	$(ROCM_HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

ifneq ($(HIPCC),)
hip-rocwmma-smoke: tools/hip_rocwmma_smoke.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<

hip-q2-moe-wmma-bench: tools/hip_q2_moe_wmma_microbench.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<

hip-q8-wmma-bench: tools/hip_q8_wmma_microbench.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<
else
hip-rocwmma-smoke hip-q2-moe-wmma-bench hip-q8-wmma-bench:
	@echo "hipcc not found; cannot build $@" >&2
	@exit 1
endif

ds4_test: ds4_test.o ds4_kvstore.o rax.o $(TEST_CORE_OBJS)
	$(TEST_LINK) -o $@ ds4_test.o ds4_kvstore.o rax.o $(TEST_CORE_OBJS) $(LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-rocm-upstream ds4-server-rocm-upstream ds4-bench-rocm-upstream ds4-eval-rocm-upstream ds4-agent-rocm-upstream ds4_cpu ds4_native ds4_server_test ds4_test hip-rocwmma-smoke hip-q2-moe-wmma-bench hip-q8-wmma-bench *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
