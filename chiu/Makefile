all: sequential parallel vecadd dotprod

sequential:
	g++ sequential.cpp -o sequential

parallel:
	nvcc parallel.cu -o parallel

vecadd:
	nvcc vecadd.cu -o vecadd

dotprod:
	nvcc dotprod.cu -o dotprod

clean:
	rm -f sequential parallel vecadd dotprod
