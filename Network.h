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
    void train(unsigned int i);
    /* vars */
    std::default_random_engine *eng;
    unsigned char *host_labels;
    float *input_l;
    float *input_w;
    float *input_w_grad;
    float *input_bias;
    float *input_bias_grad;
    float *hidden_l;
    float *hidden_w;
    float *hidden_w_grad;
    float *hidden_bias;
    float *hidden_bias_grad;
    float *dropouts;
    float *output_l;
    float *softmax_l;
    float *softmax_ds;
    float *hidden_ds;
};

#endif
