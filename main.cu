/*
 * Tanner Hoelzel
 */

#include <cmath>
#include <iostream>

#include "io.h"
#include "Network.h"
#include "assert.h"

__global__ void
hidden_layer(float *input, float *weights, float *output) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    weights = weights + id*28*28;

    float dp = 0;
    for (unsigned int i = 0; i < 28*28; i++) {
        dp += weights[i]*input[i];
    }
    output[id] = dp > 0? dp : 0; //ReLU
}

/*
void
softmax_layer(float *input, float *output) {
    for(int class = 0; class < 10; class++) {
        float max = 0;
        for(int i = 0; i < 1024; i++) {
            if(input[i] > max) max = input[i];
        }
        float sum = 0;
        for(int i = 0; i < 1024; i++) {
            output[i] = std::exp(in[i] - C);
        }
    }
}

    T sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = std::exp(in[i] - C);
        sum += out[i];
    }
    std::transform(out.begin(), out.end(), out.begin(), [sum](auto e) { return e/sum; });

    // Verify that it is a probability: Sums to 1 and all >= 0.
    assert(approx_equal(std::accumulate(out.begin(), out.end(), T(0)), 1));
    #ifndef NDEBUG
    std::for_each(out.begin(), out.end(), [](auto e) { assert(e >= 0); });
    #endif

    return out;
}
*/


void
train(float *images, char *labels, float *weights) {

}

void
test(float *images, char *labels, float *weights) {

    float *hidden;
    cudaMalloc(&hidden, 1024*sizeof(float));

    hidden_layer<<<1, 1024>>>(images, weights, hidden);
    gpu_assert(cudaPeekAtLastError());
    gpu_assert(cudaDeviceSynchronize());

    cudaFree(hidden);
}

int main(int argc, char **argv) {

    float images[60000][28*28];
    unsigned char labels[60000];
    float test_images[10000][28*28];
    unsigned char test_labels[10000];

    read_mnist_images("mnist/train-images-idx3-ubyte", images);
    read_mnist_labels("mnist/train-labels-idx1-ubyte", labels);
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    Network *net = new Network(images, labels);

    /*    
    for(int e = 0; e < 1; e++) {
        train(d_images, d_labels, d_weights);
        test(d_test_images, d_test_labels, d_weights);
    }
    */

    return 0;
}
