# CUDA Numba Example

This repository contains a simple example of using Numba with CUDA for GPU acceleration.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate numba_cuda
```

## Running the Example

Run the Python script:

```bash
python example.py
```

If CUDA is available, it will perform vector addition on the GPU and verify the result.

## Requirements

- CUDA-compatible GPU
- CUDA toolkit installed (via conda in the environment)
- Numba
