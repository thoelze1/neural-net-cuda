all: fcn

fcn: main.cu io.h assert.h assert.cu Network.h Network.cu
	nvcc -std=c++11 main.cu Network.cu assert.cu -o fcn

clean:
	rm -f fcn
