FROM gcc:14

# GNU time (the time package) reports wall-clock, user, and system time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY base_shll_2d.c makefile ./

RUN make

CMD ["/usr/bin/time", "-f", "real %e sec | user %U sec | sys %S sec", "./test.run"]
