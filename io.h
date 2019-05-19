/*
 * Tanner Hoelzel
 */

#ifndef _IO_H_
#define _IO_H_

struct labels_header {
    uint32_t magic;
    uint32_t n_labels;
};

struct images_header {
    uint32_t magic;
    uint32_t n_images;
    uint32_t n_rows;
    uint32_t n_cols;
};

char *
read_file(char *filename, int *retfd, unsigned int *size);

int
close_file(char *mmap, int fd, unsigned int size);

#endif
