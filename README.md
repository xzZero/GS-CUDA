# GS-CUDA

This is the parallel implementation of [Generalized Series (GS) model](https://pubmed.ncbi.nlm.nih.gov/23744655/) for functional MRI reconstruction with sparse spiral sampling.

## Installation:
- Clone the project 
``
git clone https://github.com/xzZero/GS-CUDA.git
``
- To run directly from release: 
  - Open x64/Release/Config.txt
  - Config as follows:
    - Input_dir: Input image data from sparse sampling in txt
    - Output_dir: Directory of output reconstructed images
    - Nslice: number of slices
    - My: number of pixels in y axis (i.e 64, 128, 192)
    - Mx: number of pixels in x axis (i.e 64, 128, 192)
    - Nframe: time frames
    - nnz: number of non-null sampling indexes
    - p_width: number of slices in each frame in horizontal
    - r: horizontal sampling rate
- To research the code, please refer to **kernel.cu**

## Advanced CUDA methods
- Grid-stride
- Unrolling loops
- cuBLAS implementation (not good as own built kernel)
- Read-only cache
- Single vs. Double Precision

## Result:
- Speed-up factor: 40x
- Error: 3.6%
![alt text](https://github.com/xzZero/GS-CUDA/blob/main/results.png)
