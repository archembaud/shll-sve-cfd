#COMPILER := aarch64-amazon-linux-gcc-11  or gcc  or  armclang
COMPILER := armclang
CPU := neoverse-v2
OPT_FLAGS := -O3 -fopenmp

all: BUILD_OBJECT
	${COMPILER} -mcpu=${CPU} arm_sve.o -o test.run -Wno-pedantic ${OPT_FLAGS} -msve-vector-bits=128
BUILD_OBJECT:
	${COMPILER} -mcpu=${CPU} arm_sve.c -c ${OPT_FLAGS} -msve-vector-bits=128
