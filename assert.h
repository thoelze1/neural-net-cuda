/*
 * Tanner Hoelzel
 */

#ifndef ASSERT_H
#define ASSERT_H

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)

void gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true);

#endif
