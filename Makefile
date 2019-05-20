all: fcn

fcn: main.cu io.h io.cu assert.h assert.cu Network.h Network.cu
	nvcc -std=c++11 main.cu io.cu assert.cu Network.cu -o fcn

clean:
	rm -f fcn
