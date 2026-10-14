FROM alpine:latest

# Install GCC and essential build dependencies
# This will simply add the default gcc
RUN apk add --no-cache build-base

WORKDIR /workspace

COPY base_shll_2d.c makefile ./

RUN make

CMD ["./test.run"]
