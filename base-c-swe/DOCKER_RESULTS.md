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