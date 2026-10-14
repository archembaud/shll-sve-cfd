CC:=musl-gcc    # gcc or musl-gcc
all: BUILD_OBJECT
	${CC} base_shll_2d.o -static -march=raptorlake -lm -o test.run -Wno-pedantic -Wno-unused-result -O3
BUILD_OBJECT:
	${CC} base_shll_2d.c -static -march=raptorlake -lm -c -Wpointer-arith -Wno-unused-result -O3

