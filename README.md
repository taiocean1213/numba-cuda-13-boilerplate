# CUDA Numba Example

This boilerplate epository contains a simple example of using Numba with CUDA for GPU acceleration, supports cuda 13.

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

## Dependencies

This repository depends on:

- Python 3.9 or higher
- Numba (for JIT compilation)
- numba-cuda (for CUDA support, with version matching your CUDA toolkit, e.g., [cu13] for CUDA 13)
- CUDA toolkit and drivers (matching the numba-cuda version)
- A CUDA-compatible NVIDIA GPU

## Virtual Environment

It is recommended to use a virtual environment for isolation:

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install numba numba-cuda[cu13]
```

Then run the example as usual.

## Requirements

- CUDA-compatible GPU
- CUDA toolkit and drivers installed
- Numba and numba-cuda
