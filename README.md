# BatchHessianArtifact

This is the artifact repository for the article "Generated Reduced Space Hessian Accumulation for SIMD Architectures".
It includes all the scripts necessary to reproduce the results found in the article.

## Quickstart

### Settings
To reproduce the results, you need:

- Linux
- Julia 1.6 (the binary `julia` should be specified in `PATH`)
- and a GPU (otherwise, the results will be computed only on the CPU)

The required packages, with their versions, are all specified in `Manifest.toml`.


### Reproduce results
Results can be reproduced locally with:
```shell
make

```
The command instantiates all the packages locally, and
loads the script `sc2021.jl` to generate
all results directly in the directory `results/`.

To launch the script on CUDA GPU, `CUDA.jl` must detect a
GPU (`CUDA.has_cuda_gpu() = true`).

### Generate plots

If a Python environment is available with `numpy` and `matplotlib` installed,
the plots can be generated with
```shell
make plot

```

## Structure of the repo

The batch reduced Hessian algorithm is implemented in this repo, in `src/BatchHessian.jl`.
The script implementing all the benchmarks is `scripts/run_benchmarks.jl`.
The main launcher is the script `sc2021.jl`.

### Dependencies
This repo depends on the following package:

- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/): the Julia wrapper for CUDA
- [KernelAbstraction.jl](https://github.com/JuliaGPU/KernelAbstractions.jl): an abstraction layer to write GPU kernels in Julia
- [ExaPF.jl](https://github.com/exanauts/ExaPF.jl/): a power system package implemented in Julia
- [BlockPowerFlow.jl](https://github.com/exanauts/BlockPowerFlow.jl/): this package was developed for the SC21 paper. It implements a Julia wrapper for `cusolverRF`. It implements also a block BICGSTAB algorithm.


