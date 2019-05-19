all: fcn

fcn: io.o main.cu
	nvcc main.cu io.o -o fcn

io.o: io.c
	gcc -c io.c

clean:
	rm -f fcn *.o
