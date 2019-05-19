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
