/*
 * Tanner Hoelzel
 */

#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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
    return
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
close_file(char *mmap, int fd, unsigned int size) {
    int rv;
    rv = munmap(mmap, size); assert(rv == 0);
    rv = close(fd); assert(rv == 0);
}

char *
read_file(char *filename, int *retfd, unsigned int *size) {

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        int en = errno;
        fprintf(stderr, "Couldn't open %s: %s\n", filename, strerror(en));
        exit(2);
    }

    struct stat sb;
    int rv = fstat(fd, &sb); assert(rv == 0);

    // Use some flags that will hopefully improve performance.
    void *vp = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
    }
    char *file_mem = (char *) vp;

    // Tell the kernel that it should evict the pages as soon as possible.
    rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);

    *retfd = fd;

    *size = sb.st_size;

    return file_mem;
}

float *
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
    float *images = (float *)malloc(n_images*28*28*sizeof(float *));
    for(int i = 0; i < n_images*28*28; i++) {
        images[i] = ((float)data[i])/127.5 - 1;
    }
    assert(close_file((char *)file, fd, size) == 0);
    return images;
}

char *
read_labels(char *filename, int n_labels) {
    int fd;
    unsigned int size;
    char *file = read_file(filename, &fd, &size);
    struct labels_header *header = (struct labels_header *)file;
    char *data = file + sizeof(struct labels_header);
    assert(header->magic == swap(0x801));
    assert(header->n_labels == swap(n_labels));
    char *labels = (char *)malloc(n_labels*sizeof(char));
    for(int i = 0; i < n_labels; i++) {
        labels[i] = data[i];
        assert(labels[i] >= 0 && labels[i] <= 9);
    }
    assert(close_file((char *)file, fd, size) == 0);
    return labels;
}
