# Timing Details for Docker vs Native execution

Low Resolution (512x512 grid, 1475 time steps, 1st Order)
2D Shallow Water Equations using the SHLL solver

## Native (kinda), Running WSL2

Details: Intel(R) Core(TM) i7-14700HX, 32 GB RAM, gcc version 13.3.0

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 34.107 | 35.822 | 34.547 | 33.270 |  32.838  | 33.667  |  33.503  | 33.474 |  33.9035  | 0.8677 |

## Docker, Running in WSL2

Details: Intel(R) Core(TM) i7-14700HX, 32 GB RAM, using gcc:14 image (1.52G)

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 34.40 | 34.65 | 32.60 |33.690 | 32.321|34.288 | 34.329 |35.384 |  33.9578  | 1.0786 |

## Docker, Running in WSL2

Details: Intel(R) Core(TM) i7-14700HX, 32 GB RAM, using gcc:13 image (1.4GB)

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
|33.868 | 34.646| 33.818|34.622 | 34.768|33.747 |33.500  |33.269 | 34.0297 | 0.3266 | 

## Native (WSL2) using MUSL

Details:  Intel(R) Core(TM) i7-14700HX, 32 GB RAM, musl-gcc version 13.3.0 (based on gnu-gcc using musl backend)

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 10.732 | 11.522 | 10.893 | 11.032 | 10.849|  10.759 | 11.587 | 11.337 |  11.0889   |    0.1189   | 


## Docker (in WSL2) using Alpine

Details:  Intel(R) Core(TM) i7-14700HX, 32 GB RAM, gcc version 13.3.0 (based on musl backend in Alpine) using alpine image (289MB)

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 10.708 | 10.745 |  10.779 | 11.757 | 11.065 |  11.080 | 11.036 | 12.027 |  11.1496 | 0.2371 |

## Native (Ubuntu) using GCC on Intel Ultra 9

Details: Intel(R) Core(TM) Ultra 9 285K, 128GB RAM, using gcc 13.3.0

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 21.711 | 21.743 | 23.705 | 23.458 | 23.111 | 23.130 | 23.501  | 23.025 |  22.9230 | 0.5974 |

## Docker (Ubuntu) using GCC image (gcc:13) on Intel Ultra 9

Details: Intel(R) Core(TM) Ultra 9 285K, 128GB RAM, using gcc 13.3.0 on gcc:13 image

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 21.009 | 21.740 | 21.674 | 21.542 | 22.966 | 22.505 |  22.190 | 22.344 | 21.9962 |  0.3869 |


## Native (Ubuntu) using GCC-MUSL on Intel Ultra 9

Details: Intel(R) Core(TM) Ultra 9 285K, 128GB RAM, using gcc 13.3.0 with MUSL backend

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 9.728 | 9.726 | 9.636 | 9.805 | 9.704 | 9.309 | 9.338 | 9.834 | 9.6350 | 0.0407 |


## Docker (Ubuntu) using Alpine image on Intel Ultra 9

Details: Intel(R) Core(TM) Ultra 9 285K, 128GB RAM, using alpine:latest@sha256:294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6 

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |  Run 7 | Run 8 | AVG | Var |
|-------|-------|-------|-------|-------|-------|--------|-------|-----|-----|
| 9.904 |  9.559 | 9.890 | 9.738 | 9.734 | 9.884 | 9.860 | 9.782 | 9.7939 | 0.0137 |