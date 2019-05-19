all: sequential parallel

sequential:
	g++ sequential.cpp -o sequential

parallel:
	nvcc parallel.cu -o parallel

clean:
	rm -f sequential parallel
