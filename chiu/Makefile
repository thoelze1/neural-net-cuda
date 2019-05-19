all: sequential vecadd dotprod

sequential:
	nvcc -std=c++11 sequential.cu -o sequential

vecadd:
	nvcc -std=c++11 vecadd.cu -o vecadd

dotprod:
	nvcc -std=c++11 dotprod.cu -o dotprod

clean:
	rm -f vecadd dotprod sequential
