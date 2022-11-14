#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>
#include<cublas_v2.h>
#include<iostream>

#define BATCH 1
__global__ void real2complex(cufftDoubleReal *in, cufftDoubleComplex *out) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = make_cuDoubleComplex(in[i], 0);
}

__global__ void real2complex_(const cufftDoubleReal* __restrict__ in, cufftDoubleComplex* __restrict__ out, int NN) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;

#pragma unroll
	for (; i < NN; i += blockDim.x*gridDim.x) {
		out[i] = make_cuDoubleComplex(in[i], 0);
	}

}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

void cuFFTR2C(cudaStream_t stream, cufftHandle plan, cufftDoubleReal *indata, cuDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice * My*Mx;
	//dim3 block(64);
	real2complex << < (nElem / 128), 128, 0, stream >> > (indata, outdata);



	//cufftResult status;
	//status = cufftExecZ2Z(plan, (cufftDoubleComplex*)outdata, (cufftDoubleComplex*)outdata, CUFFT_FORWARD);
	//std::cout << status << std::endl;

	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



}

void cuFFTR2C_1(cudaStream_t stream, cufftHandle plan, cufftDoubleReal *indata, cuDoubleComplex *outdata_, cuDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice * My*Mx;
	//dim3 block(64);
	real2complex << < (nElem / 128), 128, 0, stream >> > (indata, outdata_);



	//cufftResult status;
	//status = cufftExecZ2Z(plan, (cufftDoubleComplex*)outdata, (cufftDoubleComplex*)outdata, CUFFT_FORWARD);
	//std::cout << status << std::endl;

	if (cufftExecZ2Z(plan, outdata_, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



}


void cuFFTR2C_(cudaStream_t stream, cufftHandle plan, cufftDoubleReal *indata, cuDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice * My*Mx;
	//dim3 block(64);
	real2complex_ << < (nElem / 1024), 128, 0, stream >> > (indata, outdata, nElem);



	//cufftResult status;
	//status = cufftExecZ2Z(plan, (cufftDoubleComplex*)outdata, (cufftDoubleComplex*)outdata, CUFFT_FORWARD);
	//std::cout << status << std::endl;

	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



}

void cuFFTC2C(cudaStream_t stream, cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {


	//cufftResult status;
	//status = cufftExecZ2Z(plan, (cufftDoubleComplex*)outdata, (cufftDoubleComplex*)outdata, CUFFT_FORWARD);
	//std::cout << status << std::endl;

	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}

}

void cuIFFTC2C(cudaStream_t stream, cufftHandle plan, cublasHandle_t handle, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {   // because C-> C we only need 1 variable: outdata. 


																																		  //cublasHandle_t handle;
																																		  //cublasStatus_t status;
	double alpha = double(1) / (Nslice*My*Mx);



	//status = cublasCreate(&handle);
	//cufftResult status;
	//status = cufftExecZ2Z(plan, (cufftDoubleComplex*)outdata, (cufftDoubleComplex*)outdata, CUFFT_INVERSE);
	//std::cout << status << std::endl;


	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z inverse failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



	//std::cout <<"FFT Error" <<_cudaGetErrorEnum(cublasZdscal(handle, Nslice*My*Mx, &alpha, outdata, 1)) << std::endl;
	cublasZdscal(handle, Nslice*My*Mx, &alpha, outdata, 1);
}