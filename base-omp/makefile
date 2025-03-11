all: BUILD_OBJECT
	gcc 2nd_order_base_shll.o -fopenmp  -lm -o test.run -Wno-pedantic -O3
BUILD_OBJECT:
	gcc 2nd_order_base_shll.c -fopenmp -lm -c -Wpointer-arith -O3
