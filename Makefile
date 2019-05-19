all: fcn

fcn: main.cu io.h io.o
	nvcc -std=c++11 main.cu io.o -o fcn

io.o: io.h io.cpp
	g++ io.cpp -c

clean:
	rm -f fcn *.o
