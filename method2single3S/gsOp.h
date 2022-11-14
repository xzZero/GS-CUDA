#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fft.h"
#include <stdlib.h>
#include <iostream>

void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz);
void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void Ahop_(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz);
void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);