CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
HIPCC ?= $(shell command -v hipcc 2>/dev/null)
HIPCXXFLAGS ?= -O3 --offload-arch=native -std=c++17 -Wno-unused-parameter -Wno-unused-function

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)

ROCM_PATH ?= /opt/rocm
ROCM_ARCH ?= gfx1151
ROCM_HIPCC ?= $(if $(HIPCC),$(HIPCC),$(ROCM_PATH)/bin/hipcc)
ROCM_CFLAGS ?= -O3 -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -L$(ROCM_PATH)/lib -lhipblas

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
LINK.ds4 = $(CC)
TEST_CORE_OBJS = $(CORE_OBJS)
TEST_CFLAGS = $(CFLAGS)
TEST_LINK = $(CC)
else
ifneq ($(HIPCC),)
CFLAGS += -D_GNU_SOURCE -DDS4_USE_HIP
CORE_OBJS = ds4.o ds4_hip.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
HIP_LDLIBS := $(LDLIBS) -lhipblaslt
LINK.ds4 = $(HIPCC)
TEST_CORE_OBJS = $(NATIVE_CORE_OBJS)
TEST_CFLAGS = $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL
TEST_LINK = $(CC)
else
CFLAGS += -D_GNU_SOURCE -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
LINK.ds4 = $(CC)
TEST_CORE_OBJS = $(CORE_OBJS)
TEST_CFLAGS = $(CFLAGS)
TEST_LINK = $(CC)
endif
endif

.PHONY: all clean test rocm rocm-upstream

all: ds4 ds4-server

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(LINK.ds4) -o $@ $^ $(if $(HIPCC),$(HIP_LDLIBS),$(LDLIBS))

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)

rocm rocm-upstream: ds4-rocm-upstream ds4-server-rocm-upstream
	@echo "ROCm upstream-shaped binaries built with ROCM_ARCH=$(ROCM_ARCH)"

ds4-rocm-upstream: ds4_cli_gpuapi.o linenoise.o ds4_gpuapi.o ds4_cuda.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-server-rocm-upstream: ds4_server_gpuapi.o rax.o ds4_gpuapi.o ds4_cuda.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(TEST_CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -UDS4_USE_HIP -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_hip.o: ds4_hip.cpp ds4_metal.h
	$(HIPCC) $(HIPCXXFLAGS) -c -o $@ ds4_hip.cpp

ds4_gpuapi.o: ds4.c ds4.h ds4_metal.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4.c

ds4_cli_gpuapi.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_cli.c

ds4_server_gpuapi.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_server.c

ds4_bench_gpuapi.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_bench.c

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc ds4_rocm.h rocm/ds4_rocm_common.cuh rocm/ds4_rocm_q8.cuh rocm/ds4_rocm_fp8_kv.cuh rocm/ds4_rocm_attention.cuh rocm/ds4_rocm_hc.cuh rocm/ds4_rocm_output.cuh rocm/ds4_rocm_moe.cuh
	$(ROCM_HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_cuda.cu

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

ds4_test: ds4_test.o rax.o $(TEST_CORE_OBJS)
	$(TEST_LINK) -o $@ ds4_test.o rax.o $(TEST_CORE_OBJS) $(LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-rocm-upstream ds4-server-rocm-upstream ds4_native ds4_server_test ds4_test hip-rocwmma-smoke hip-q2-moe-wmma-bench hip-q8-wmma-bench *.o
