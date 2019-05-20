/*
 * Tanner Hoelzel
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <random>
#include <thrust/device_vector.h>

#include "assert.h"

class Network {
public:
    Network(float *inputs, unsigned char *labels);
    ~Network();
    void train();
    float test(float *tests, unsigned char *labels);
private:
    /* fns */
    void run(unsigned int i);
    /* vars */
    std::default_random_engine *eng;
    char *labels;
    float *inputs;
    float *weights1;
    float *outputs;
    float *weights2;
    float *classes;
    float *softmax;
    thrust::device_vector<float> temp;
};

#endif
