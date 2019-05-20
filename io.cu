/*
 * Tanner Hoelzel
 */

#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>

void
swap(int &i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

void
read_mnist_images(const std::string &fn, float *images, unsigned int n) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == n);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    for (int i = 0; i < n; i++) {
        unsigned char tmp[28*28];
        rv = read(fd, tmp, 28*28); assert(rv == 28*28);
        for (int r = 0; r < 28*28; r++) {
            // Make go from -1 to 1.
            images[i*28*28+r] = float(tmp[r])/127.5 - 1;
        }
    }

    rv = close(fd); assert(rv == 0);
}

void
read_mnist_labels(const std::string &fn, unsigned char *labels, unsigned int n) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == n);

    rv = read(fd, labels, n); assert(rv == n);

    rv = close(fd); assert(rv == 0);
}
