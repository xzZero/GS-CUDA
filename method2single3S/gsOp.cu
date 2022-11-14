#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fft.h"
#include <stdlib.h>
#include <iostream>



__global__ void subsref(cufftDoubleComplex* indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nnz)
		outdata[idx] = indata[(long)S[idx]];
}
__global__ void subsasgn(cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	//*data must be memset
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nnz)
		outdata[(long)S[idx]] = indata[idx];

}
__global__ void piecewiseMatMul(cufftDoubleReal *ref, cufftDoubleComplex *mat) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	mat[idx].x = ref[idx] * mat[idx].x;
	mat[idx].y = ref[idx] * mat[idx].y;
}

__global__ void piecewiseMatMul_(const double* __restrict__ ref, double2* __restrict__ mat, int NN) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//double tmpx, tmpy;
#pragma unroll
	for (; idx < NN; idx += blockDim.x*gridDim.x) {
		mat[idx].x = ref[idx] * mat[idx].x;
		mat[idx].y = ref[idx] * mat[idx].y;
		/*double tmpx = mat[idx].x;
		double tmpy = mat[idx].y;

		tmpx *= ref[idx];
		tmpy *= ref[idx];

		mat[idx].x = tmpx;
		mat[idx].y = tmpy;*/
	}


}

__global__ void subsref_(const cufftDoubleComplex* __restrict__ indata, cufftDoubleComplex* __restrict__ outdata, const double* __restrict__ S, long nnz) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

#pragma unroll
	for (; idx < nnz; idx += blockDim.x*gridDim.x) {
		outdata[idx] = indata[(long)S[idx]];
	}

}
__global__ void subsasgn_(const cufftDoubleComplex* __restrict__ indata, cufftDoubleComplex* __restrict__ outdata, const double* __restrict__ S, long nnz) {
	//*data must be memset
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
#pragma unroll
	for (; idx < nnz; idx += blockDim.x*gridDim.x) {
		outdata[(long)S[idx]] = indata[idx];
	}


}

void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz) {



	cuFFTR2C_(stream, plan, indata, outdata_, Nslice, My, Mx);


	subsref_ << <((nnz + 1024 - 1) / 1024), 128, 0, stream >> > (outdata_, outdata, S, nnz);
	//cudaDeviceSynchronize(); 
	//cudaFree(outdata_); 


}

void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	//cufftComplex *outdata_; 
	//cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftComplex));
	//AWARE:  INPUT OF AOPCOMPL IS CHANGED, SO DON'T USE INPUT AGAIN!!!! IF YOU WANT TO USE, COPY IT TO ANOTHER VARIABLE. 
	//cufftComplex *outdata_; 
	//cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftComplex));

	cuFFTC2C(stream, plan, indata, Nslice, My, Mx);

	subsref_ << <((nnz + 1024 - 1) / 1024), 128, 0, stream >> > (indata, outdata, S, nnz);
	//cudaDeviceSynchronize();


	//cudaFree(outdata_); 

}

void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	cudaMemsetAsync(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex), stream);
	subsasgn_ << <((nnz + 1024 - 1) / 1024), 128, 0, stream >> > (indata, outdata, S, nnz);


	//cudaDeviceSynchronize();

	cuIFFTC2C(stream, plan, handle, outdata, Nslice, My, Mx);



}

void Ahop_(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	cudaMemsetAsync(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex), stream);
	subsasgn_ << <((nnz + 1024 - 1) / 1024), 128, 0, stream >> > (indata, outdata, S, nnz);


	//cudaDeviceSynchronize();

	cuIFFTC2C(stream, plan, handle, outdata, Nslice, My, Mx);



}

void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz) {



	//cudaMemsetAsync(outdata_, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex), stream);

	//cudaMemset(outdata_, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	Ahop(Nslice, My, Mx, plan, handle, stream, indata, outdata_, S, nnz);

	piecewiseMatMul_ << <(Nslice*My*Mx / 1024), 128, 0, stream >> >(ref, outdata_, Nslice*My*Mx);
	//cudaDeviceSynchronize();
	AopCompl(Nslice, My, Mx, plan, handle, stream, outdata_, outdata, S, nnz);
	//cudaFree(outdata_);

}
void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	//cudaMemset(outdata, 0, Nslice*My*Mx * sizeof(cufftComplex));
	//cudaMemsetAsync(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex), stream);

	Ahop(Nslice, My, Mx, plan, handle, stream, indata, outdata, S, nnz);
	piecewiseMatMul_ << <(Nslice*My*Mx / 1024), 128, 0, stream >> >(ref, outdata, Nslice*My*Mx);

}