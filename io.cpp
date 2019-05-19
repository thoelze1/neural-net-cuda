/*
 * Tanner Hoelzel
 */

#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>

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

float *
read_images(char *filename, unsigned int n_images) {

    int rv;

    int fd;
    fd = open(filename, O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n = read_int(fd);
    assert(n == n_images);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    unsigned char data[28*28*n_images];
    rv = read(fd, data, 28*28*n_images);
    assert(rv == 28*28*n_images);

    rv = close(fd);
    assert(rv == 0);

    float *images = new float[n_images*28*28*sizeof(float *)];
    for(int i = 0; i < n_images*28*28; i++) {
        images[i] = ((float)data[i])/127.5 - 1;
    }

    return images;
}

char *
read_labels(char *filename, int n_labels) {

    int rv;

    int fd;
    fd = open(filename, O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n = read_int(fd);
    assert(n == n_labels);

    char *labels = new char[n_labels*sizeof(char)];
    rv = read(fd, labels, n_labels);
    assert(rv == n_labels);

    for(int i = 0; i < n_labels; i++) {
        assert(labels[i] >= 0 && labels[i] <= 9);
    }

    rv = close(fd);
    assert(rv == 0);

    return labels;
}
