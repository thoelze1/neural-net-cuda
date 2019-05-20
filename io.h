/*
 * Tanner Hoelzel
 */

#ifndef IO_H
#define IO_H

#include <string>

void read_mnist_images(const std::string &fn, float *images, unsigned int n);
void read_mnist_labels(const std::string &fn, unsigned char *labels, unsigned int n);

#endif
