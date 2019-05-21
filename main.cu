/*
 * Tanner Hoelzel
 */

#include "io.h"
#include "Network.h"

#define N_EPOCHS    1

int main(int argc, char **argv) {

    float images[60000*28*28];
    unsigned char labels[60000];
    float test_images[10000*28*28];
    unsigned char test_labels[10000];

    read_mnist_images("mnist/train-images-idx3-ubyte", images, 60000);
    read_mnist_labels("mnist/train-labels-idx1-ubyte", labels, 60000);
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images, 10000);
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels, 10000);

    Network net(images, labels);

    std::cout << net.test(test_images, test_labels) << std::endl;
    for(unsigned int e = 0; e < N_EPOCHS; e++) {
        net.train();
        std::cout << net.test(test_images, test_labels) << std::endl;
    }

    return 0;
}
