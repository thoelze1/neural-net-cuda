/*
 * Tanner Hoelzel
 */

#include "io.h"
#include "Network.h"

int main(int argc, char **argv) {

    float images[60000][28*28];
    unsigned char labels[60000];
    float test_images[10000][28*28];
    unsigned char test_labels[10000];

    read_mnist_images("mnist/train-images-idx3-ubyte", images);
    read_mnist_labels("mnist/train-labels-idx1-ubyte", labels);
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    Network net(images, labels);

    for(int e = 0; e < 1; e++) {
        net.train();
        net.test();
    }

    return 0;
}
