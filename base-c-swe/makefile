all: BUILD_OBJECT
	gcc base_shll_2d.o -static -march=raptorlake -lm -o test.run -Wno-pedantic -Wno-unused-result -O3
BUILD_OBJECT:
	gcc base_shll_2d.c -static -march=raptorlake -lm -c -Wpointer-arith -Wno-unused-result -O3
