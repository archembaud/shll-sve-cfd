FROM alpine:latest

# Install GCC and essential build dependencies
# This will simply add the default gcc
RUN apk add --no-cache build-base

WORKDIR /workspace

COPY base_shll_2d.c makefile ./

# Override the default musl-gcc, alpine gcc using musl by default
RUN make CC=gcc

CMD ["./test.run"]
