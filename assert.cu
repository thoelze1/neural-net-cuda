/*
 * Tanner Hoelzel
 */

#include <cstdio>

void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
