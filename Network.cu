/*
 * Tanner Hoelzel
 */

#include "Network.h"

Network::~Network() {
    cudaFree(this->inputs);
    cudaFree(this->labels);
    cudaFree(this->weights1);
    cudaFree(this->weights2);
    cudaFree(this->scratch);
}
