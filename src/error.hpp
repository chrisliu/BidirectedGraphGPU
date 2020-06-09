/** error.hpp
 * Error handling code taken from the book CUDA by Example: An Introduction to
 * General-Purpose GPU Programmin
 */

#ifndef ERROR_HPP
#define ERROR_HPP

#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif /* ERROR_HPP */