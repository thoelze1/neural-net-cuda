all: fcn

fcn: main.cu
	nvcc -std=c++11 main.cu -o fcn

clean:
	rm -f fcn
