# CUDA Tensor

CUDA-Tensor is currently focused on building tensor primitives and tensor operations from scratch with CUDA/C++.

The longer-term direction is still PINNs (Physics-Informed Neural Networks), but the active implementation right now is the tensor core needed for that future work.

## Current Scope

- CUDA-backed tensor type in `include/tensor.cuh`
- Elementwise arithmetic (`add`, `sub`, `mul`, `div`) with broadcasting support
- Matrix/tensor operations such as `matmul`, `dot`, and `transpose`
- Runtime/device support under `include/device/`
- A small executable example in `src/main.cu`

## Project Structure

```text
CUDA-Tensor
├── include
│   ├── tensor.cuh              # Core tensor implementation
│   ├── network.cuh             # Reserved for future PINN/network work
│   ├── host/
│   │   └── init_tensor.h       # Tensor initialization/base host utilities
│   └── device/                 # CUDA runtime, interface, constants, kernels, variables
├── src
│   ├── main.cu                 # Example entry point using tensor operations
│   ├── network.cu              # Reserved for future PINN/network work
│   └── utils.cu                # Reserved for future utilities
├── other
│   └── legacy_tensor.cuh       # Legacy tensor implementation
├── CMakeLists.txt
└── README.md
```

## Build

Requirements:
- CMake (3.18+)
- CUDA Toolkit (with `nvcc`)
- A C++20-capable compiler

Build commands:

```bash
cmake -S . -B build
cmake --build build
```

This produces the `PINN` executable target (name kept for now).

## Run

```bash
./build/PINN
```

The current `main.cu` demonstrates a simple tensor division example.

## Roadmap

- Expand and harden tensor functionality/performance
- Add testing and benchmarking
- Reintroduce network/PINN layers on top of the tensor core
