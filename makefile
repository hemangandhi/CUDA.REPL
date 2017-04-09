all: ./test.cu
	nvcc ./test.cu -o test