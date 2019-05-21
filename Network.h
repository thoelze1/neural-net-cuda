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
    unsigned char *host_labels;
    float *input_l;
    float *input_w;
    float *hidden_l;
    float *hidden_w;
    float *output_l;
    float *softmax_l;
    float *softmax_ds;
    float *hidden_ds;
    float *input_w_grad;
    float *hidden_w_grad;
};

#endif
