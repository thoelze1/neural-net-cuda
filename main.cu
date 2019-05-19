#include <random>
#include <cstdio>
#include <cassert>

#define NUM_EPOCHS              1
#define NUM_BATCHS              1
#define NUM_INPUT_NODES     28*28
#define NUM_HIDDEN_NODES     1024

#define TRAIN_LABEL_FILE   "mnist/train-labels-idx1-ubyte"
#define TRAIN_IMAGE_FILE   "mnist/train-images-idx3-ubyte"
#define TEST_LABEL_FILE    "mnist/t10k-labels-idx1-ubyte"
#define TEST_IMAGE_FILE    "mnist/t10k-images-idx3-ubyte"

#define NUM_WEIGHTS        NUM_INPUT_NODES*NUM_HIDDEN_NODES

extern "C" char * read_file(char *, int *, unsigned int *);
extern "C" char * close_file(char *, int, unsigned int);

struct labels_header {
    int32_t magic;
    int32_t n_labels;
} __attribute__((packed));

struct images_header {
    int32_t magic;
    int32_t n_images;
    int32_t n_rows;
    int32_t n_cols;
} __attribute__((packed));

int
swap(int i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

float **
read_images(char *filename, unsigned int n_images) {
    int fd;
    unsigned int size;
    char *file = read_file(filename, &fd, &size);
    struct images_header *header = (struct images_header *)file;
    char *data = file + sizeof(struct images_header);
    assert(header->magic == swap(0x803));
    assert(header->n_images == swap(n_images));
    assert(header->n_rows == swap(28));
    assert(header->n_cols == swap(28));
    float **images = (float **)malloc(n_images*sizeof(float *));
    for(int i = 0; i < n_images; i++) {
        images[i] = (float *)malloc(28*28*sizeof(float));
        for (int j = 0; j < 28*28; j++) {
            images[i][j] = float(data[28*28*i+j])/127.5 - 1;
        }
    }
    assert(close_file((char *)file, fd, size) == 0);
    return images;
}

unsigned int *
read_labels(char *filename, int n_labels) {
    int fd;
    unsigned int size;
    char *file = read_file(filename, &fd, &size);
    struct labels_header *header = (struct labels_header *)file;
    char *data = file + sizeof(struct labels_header);
    assert(header->magic == swap(0x801));
    assert(header->n_labels == swap(n_labels));
    unsigned int *labels = (unsigned int *)malloc(n_labels*sizeof(unsigned int));
    for(int i = 0; i < n_labels; i++) {
        labels[i] = (unsigned int)data[i];
        assert(labels[i] >= 0 && labels[i] <= 9);
    }
    assert(close_file((char *)file, fd, size) == 0);
    return labels;
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

int main(int argc, char **argv) {
    float **images = read_images(TRAIN_IMAGE_FILE, 60'000);
    unsigned int *labels = read_labels(TRAIN_LABEL_FILE, 60'000);
    float **test_images = read_images(TEST_IMAGE_FILE, 10'000);
    unsigned int *test_labels = read_labels(TEST_LABEL_FILE, 10'000);
    float *weights = random_weights(NUM_WEIGHTS * sizeof(float));
    for(unsigned int i_epoch = 0; i_epoch < NUM_EPOCHS; i_epoch++) {
    }
}
