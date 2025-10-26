# CUDA Numba Example

This repository contains a simple example of using Numba with CUDA for GPU acceleration.

## Setup

Create the conda environment (if conda is available):

```bash
conda env create -f environment.yml
conda activate numba_cuda
```

Alternatively, install dependencies with pip:

```bash
pip install numba numba-cuda[cu13]  # Adjust cu13 to your CUDA version, e.g., cu12
```

## Running the Example

Run the Python script:

```bash
python example.py
```

If CUDA is available, it will perform vector addition on the GPU and verify the result.

## Requirements

- CUDA-compatible GPU
- CUDA toolkit and drivers installed
- Numba and numba-cuda
