/*
 * Tanner Hoelzel
 */

#include <random>

extern "C" float * read_images(char *, unsigned int);
extern "C" char * read_labels(char *, unsigned int);

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

/*
 * Read inputs, labels, and weights into GPU
 */
int main(int argc, char **argv) {

    float *images, *test_images, *weights;
    char *labels, *test_labels;

    labels = read_labels("mnist/train-labels-idx1-ubyte", 60'000);
    images = read_images("mnist/train-images-idx3-ubyte", 60'000);
    test_labels = read_labels("mnist/t10k-labels-idx1-ubyte", 10'000);
    test_images = read_images("mnist/t10k-images-idx3-ubyte", 10'000);
    weights = random_weights(28*28*1024*sizeof(float));

    float *d_images, *d_labels, *d_weights;
    cudaMalloc(&d_images, 60'000*28*28*sizeof(float));
    cudaMalloc(&d_labels, 60'000*sizeof(unsigned int));
    cudaMalloc(&d_weights, 28*28*1024*sizeof(unsigned int));

    cudaMemcpy(d_images, images, 60'000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, 60'000*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);

    for(unsigned int i_epoch = 0; i_epoch < 10; i_epoch++) {
    }

    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_weights);

    free(labels);
    free(images);
    free(test_labels);
    free(test_images);
    free(weights);
}
