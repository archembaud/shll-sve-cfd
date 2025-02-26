# Finite Volume Method (FVM) CDK using SHLL and SVE

These cores are ssed as part of research into Scalable Vector Extension (SVE) intrinsic functions executed on AWS Graviton4 processors.

## Target Systems

These codes are designed for execution on AWS instances powered using Graviton4 cores; these codes will not compile and run anywhere else, with the exception of some Nvidia development boards and the Apple M4 processor.

In order to run these, you'll need to make sure that:

* NPM and Nodejs are installed,
* You have configured your python venv correctly, activated it, and have aws-cdk.core and aws-cdk-lib installed,
* You have installed the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
* You have your AWS credentials properly set up. Test this using:

```bash
aws sts get-caller-identity
```

## Deployment using CDK

To deploy the AWS resources required to run these codes:

* Navigate to the CDK directory;
* Make sure you've installed your node modules:

```bash
npm install
```

* Synthesize your cloud formation template - this is a test to make sure you have properly configured your AWS as well as have the requirements installed:

```bash
npm run cdk synth
```
* Finally, to deploy:

```bash
npm run cdk deploy
```

## Additional notes

To create a new tmux session:

```bash
tmux new -s mysession
```

To attach to an existing session:

```bash
tmux attach -t mysession
```

To select the ARM compiler:

```bash
module avail
module load acfl/24.10.1
module load gnu/14.2.0
```

## Timing Results

Shown below are timing results when using the medium deployment as outlined in the CDK scripts.

### 1D

Timings shown are for a one dimensional shock tube problem (Sod's shock tube) using a density ratio of 10 and temperature ratio of 1.
Simulations are run until a dimensionless time of 0.2. The kinetic CFL number is fixed at 0.25 for stability.

#### Using standard C (sequential) code

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256           | 410           | 0.006             | 0.006         |
| 512           | 820           | 0.021             | 0.021         |
| 1024          | 1639          | 0.082             | 0.082          |
| 2048          | 3277          | 0.332             | 0.330         |
| 4096          | 6554          | 1.349             | 1.349         |
| 8192          | 13108         | 5.404             | 5.286         |
| 16384         | 26215         | 21.584            | 21.669        |
| 32768         | 52429         | 92.176            | 91.594        |
| 65536         | 104858        | 367.05            | 367.117       |

**Table 1**: Timings for no optimization (-O0) using GCC v14.1 when using the base C code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256           | 410           | 0.002             | 0.002         |
| 512           | 820           | 0.007             | 0.007         |
| 1024          | 1639          | 0.025             | 0.025         |
| 2048          | 3277          | 0.099             | 0.099         |
| 4096          | 6554          | 0.404             | 0.403         |
| 8192          | 13108         | 1.597             | 1.595         |
| 16384         | 26215         | 6.320             | 6.312         |
| 32768         | 52429         | 29.466            | 29.495        |
| 65536         | 104858        | 121.902           | 121.748       |

**Table 2**: Timings for maximum optimization (-O3) using GCC v14.1 when using the base C code.


#### Using SVE vector intrinsic functions

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256           | 410           | 0.004             | 0.004         |
| 512           | 820           | 0.016             | 0.016         |
| 1024          | 1639          | 0.06              | 0.06          |
| 2048          | 3277          | 0.235             | 0.234         |
| 4096          | 6554          | 0.941             | 0.943         |
| 8192          | 13108         | 3.756             | 3.752         |
| 16384         | 26215         | 15.018            | 15.003        |
| 32768         | 52429         | 63.542            | 63.792        |
| 65536         | 104858        | 257.973           | 257.974       |

**Table 3**: Timings for no optimization (-O0) using GCC v14.1 when using SVE intrinsics

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256           | 410           | 0.001             | 0.001         |
| 512           | 820           | 0.004             | 0.004         |
| 1024          | 1639          | 0.014             | 0.014         |
| 2048          | 3277          | 0.057             | 0.057         |
| 4096          | 6554          | 0.222             | 0.231         |
| 8192          | 13108         | 0.892             | 0.892         |
| 16384         | 26215         | 3.488             | 3.478         |
| 32768         | 52429         | 17.834            | 17.818        |
| 65536         | 104858        | 71.269            | 71.268       |

**Table 4**: Timings for maximum optimization (-O3) using GCC v14.1 when using SVE intrinsics

#### Using standard C (sequential) code with the Arm C compiler

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256           | 410           | 0.002             | 0.002         |
| 512           | 820           | 0.005             | 0.005         |
| 1024          | 1639          | 0.015             | 0.015         |
| 2048          | 3277          | 0.059             | 0.059         |
| 4096          | 6554          | 0.240             | 0.239         |
| 8192          | 13108         | 0.945             | 0.945         |
| 16384         | 26215         | 3.505             | 3.505         |
| 32768         | 52429         | 15.926            | 15.933        |
| 65536         | 104858        | 68.001            | 67.858        |

**Table 5**: Timings for maximum optimization (-O3) using the ARM C compiler and the base C code.

## Timing Results

Shown below are timing results when using the medium deployment as outlined in the CDK scripts.

### 2D

Timings shown are for a twi dimensional implosion problem using a density ratio of 10 and temperature ratio of 1.
Simulations are run until a dimensionless time of 0.2. The kinetic CFL number is fixed at 0.25 for stability.

#### Using SVE vector instrinsic code with various compilers

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           | 1.236        |  1.234         |
| 512 x 512     | 410           | 10.583       |  10.385      |
| 1024 x 1024   | 820           | 91.237       |  91.935      |
| 2048 x 2048   | 1639          | 719.181      | 720.263      |

**Table 6**: Timings for maximum optimization (-O0) using the GCC compiler and the vector instrinsic code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           | 0.547             | 0.545        |
| 512 x 512     | 410           | 5.945             | 6.007        |
| 1024 x 1024   | 820           | 40.095            | 40.021       |
| 2048 x 2048   | 1639          | 316.08            | 315.525      |

**Table 7**: Timings for maximum optimization (-O3) using the GCC compiler and the vector instrinsic code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           | 0.488             |   0.486      |
| 512 x 512     | 410           | 5.545             |   5.862      |
| 1024 x 1024   | 820           | 39.375            |   39.403     |
| 2048 x 2048   | 1639          | 314.208           |  314.288     |

**Table 8**: Timings for maximum optimization (-O3) using the ARM compiler and the vector instrinsic code.

#### Using standard C code (sequential) with the various compilers

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           |  2.203     |   2.209       |
| 512 x 512     | 410           |  18.863    |  19.885       |
| 1024 x 1024   | 820           |  154.002   |  154.035     |
| 2048 x 2048   | 1639          |  1219.093   | 1212.102    |

**Table 9**: Timings for maximum optimization (-O0) using the GCC compiler and the base C code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           | 1.193      | 1.197        |
| 512 x 512     | 410           | 15.120     | 14.743       |
| 1024 x 1024   | 820           | 79.508     | 79.248       |
| 2048 x 2048   | 1639          | 646.733    | 644.250      |

**Table 10**: Timings for maximum optimization (-O3) using the GCC compiler and the base C code.

#### Using standard C code (sequential) with ARM compiler

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     | 205           | 1.511    |  1.512      |
| 512 x 512     | 410           | 14.182   |  13.739     |
| 1024 x 1024   | 820           | 80.789   |  80.469     |
| 2048 x 2048   | 1639          | 655.031    | 655.697   |

**Table 11**: Timings for maximum optimization (-O3) using the ARM compiler and the base C code.

### 2D (2nd order)

Timings shown are for a two dimensional Euler four shock problem.
Simulations are run until a dimensionless time of 0.8. The kinetic CFL number is fixed at 0.25 for stability.

#### Using SVE vector instrinsic (2nd order) code with various compilers

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     |   1649     |  15.891       |  16.148       |
| 512 x 512     |   3277     |  191.984      |  199.657      |
| 1024 x 1024   |   6554     |  1323.063     |  1320.593     |

**Table 12** Timings for maximum optimisation (-O3) using GCC compiler with intrinsics code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     |   1649     |  15.039       |  14.321         |
| 512 x 512     |   3277     |  175.403      |  190.902      |
| 1024 x 1024   |   6554     |  1237.76      |  1237.057     |

**Table 13** Timings for maximum optimisation (-O3) using ARM compiler with intrinsics code.

#### Using base c (2nd order) code with various compilers

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------|
| 256 x 256     |   1649     | 26.465      |  26.270      |
| 512 x 512     |   3277     | 321.852     |  308.381     |
| 1024 x 1024   |   6554     | 1939.418    |  1944.395    |

**Table 14** Timings for maximum optimisation (-O3) using GCC compiler with the base C code.

| Number of Cells | Number of Time Steps | Timing (Run A), s | Timing (Run B), s |
|----------------| ---------------| ----------------| ---------------| 
| 256 x 256     |   1649     | 25.458     |  25.943      |
| 512 x 512     |   3277     | 268.653    |  270.34      |
| 1024 x 1024   |   6554     | 1908.971   |  1892.251    |

**Table 15** Timings for maximum optimisation (-O3) using ARM compiler with the base C code.