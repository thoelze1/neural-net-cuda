/*
 * Tanner Hoelzel
 */

#include <random>

extern "C" float * read_images(char *, unsigned int);
extern "C" char * read_labels(char *, unsigned int);

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

float *
random_weights(unsigned int n_weights) {
    float *weights = (float *)malloc(n_weights*sizeof(float));
    std::default_random_engine eng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for(unsigned int i = 0; i < n_weights; i++) {
        weights[i] = dist(eng);
    }
    return weights;
}

__global__ void
hidden_layer(float *input, float *weights, float *output) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    weights = weights + id*28*28;

    float dp = 0;
    for (unsigned int i = 0; i < 28*28; i++) {
        dp += weights[i]*input[i];
    }
    output[id] = dp;
}

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

/*
 * Read inputs, labels, and weights into GPU
 */
int main(int argc, char **argv) {

    char *labels, *test_labels;
    float *images, *test_images, *weights;

    labels = read_labels("mnist/train-labels-idx1-ubyte", 60'000);
    test_labels = read_labels("mnist/t10k-labels-idx1-ubyte", 10'000);
    images = read_images("mnist/train-images-idx3-ubyte", 60'000);
    test_images = read_images("mnist/t10k-images-idx3-ubyte", 10'000);
    weights = random_weights(28*28*1024*sizeof(float));

    char *d_labels, *d_test_labels;
    float *d_images, *d_test_images, *d_weights;

    cudaMalloc(&d_labels, 60'000*sizeof(char));
    cudaMalloc(&d_test_labels, 10'000*sizeof(char));
    cudaMalloc(&d_images, 60'000*28*28*sizeof(float));
    cudaMalloc(&d_test_images, 10'000*28*28*sizeof(float));
    cudaMalloc(&d_weights, 28*28*1024*sizeof(float));

    cudaMemcpy(d_labels, labels, 60'000*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_labels, test_labels, 10'000*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_images, images, 60'000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images, test_images, 10'000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);

    for(int e = 0; e < 1; e++) {
        train(d_images, d_labels, d_weights);
        test(d_test_images, d_test_labels, d_weights);
    }

    cudaFree(d_labels);
    cudaFree(d_test_labels);
    cudaFree(d_images);
    cudaFree(d_test_images);
    cudaFree(d_weights);

    free(labels);
    free(test_labels);
    free(images);
    free(test_images);
    free(weights);
}
