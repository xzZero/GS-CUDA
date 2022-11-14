#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FileOpen.h"
#include "fft.h"
#include "gsOp.h"
#include <iostream>
#include "TimerCuda.h"
#include <time.h>
#include <cublas_v2.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <Windows.h>
#include <fstream>
#include <time.h>
#include <sys/timeb.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <direct.h>
#define GetCurrentDir _getcwd

//Func prototype
void saveBitmap(const char *p_output, int p_width, int p_height, unsigned char *Image);
void saveBitmap(const char *p_output, int p_width, int p_height, int Mx, int My, int x_cor, int y_cor, unsigned char *Image);
double nrmse(double *a, double *b, int p_width, int p_height, int Mx, int My, int x_cor, int y_cor);
int replacestr(char *line, const char *search, const char *replace);
void checkDeviceProperty(cudaDeviceProp deviceProp);
void DisplayHeader();
void printInfo();
void info();
void checkInput(char *iff, char *in);

char name[200], under[200], method[200], imgres[200], outputname[200], opt;
char inURL[FILENAME_MAX], outURL[FILENAME_MAX];
int p_width, p_height, x_cor, y_cor;
/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		cudaDeviceReset();
		if (abort) exit(code);
	}

}
//CHECK CUBLAS ERROR
#define cublasER(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cublasStatus_t code, char *file, int line, bool abort = true)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cublasER: %s %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}
//CHECK CUFFT ERROR
#define cufftER(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult code, char *file, int line, bool abort = true)
{
	if (code != CUFFT_SUCCESS)
	{
		fprintf(stderr, "cufftER: %s %s %d\n", code, file, line);
		if (abort) exit(code);
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

__global__ void doubleMatrix(double *x, double *out, int Nslice, int My, int Mx, int residual) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 2);
	if (idx <= Nslice * My*Mx - residual - 1) {
		out[idx] = x[idx] * x[idx];
	}
}

__global__ void Sminus1(double *S) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	S[idx] = S[idx] - 1;
}

__global__ void Sminus1_(double *S, int NN) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

#pragma unroll
	for (; idx < NN; idx += blockDim.x*gridDim.x) {
		S[idx] = S[idx] - 1;
	}

}

__global__ void minusAbs(double *a, double *b, int NN) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

#pragma unroll
	for (; idx < NN; idx += blockDim.x*gridDim.x) {
		b[idx] = abs(a[idx] - b[idx]);
		//b[idx] = a[idx] - b[idx];
		//if (b[idx] < 0) {
		//	b[idx] = 0;
		//}
	}
}

__global__ void abs_matrix(cufftDoubleComplex *x, double *y, int Nslice, int My, int Mx, int residual) {//check----------------------------

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 2);
	if (idx <= Nslice * My*Mx - residual - 1) {
		y[idx] = sqrt(x[idx].x*x[idx].x + x[idx].y*x[idx].y);
	}
}

__global__ void abs_matrix_(const cufftDoubleComplex* __restrict__ x, double* __restrict__ y, int Nslice, int My, int Mx, int residual) {//check----------------------------

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 2);
	//if (idx <= Nslice * My*Mx - residual - 1) {
	//	y[idx] = sqrt(x[idx].x*x[idx].x + x[idx].y*x[idx].y);
	//}

#pragma unroll
	for (; idx <= Nslice * My*Mx - residual - 1; idx += blockDim.x*gridDim.x) {
		y[idx] = sqrt(x[idx].x*x[idx].x + x[idx].y*x[idx].y);
	}
}


__global__ void Dmultiply(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {   //check------------------
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 3);
	if (idx <= Nslice * My*Mx - 2) {

		output[idx].x = x[idx].x - x[idx + 1].x;
		output[idx].y = x[idx].y - x[idx + 1].y;
	}
	// copy final var in x to output since D = 1
}
__global__ void DmultiplyAbs(cufftDoubleComplex *x, double *output, int Nslice, int My, int Mx) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	idx = max(idx, (long)0);
	idx = min(idx, (long)(Nslice*My*Mx - 2));

	output[idx] = (double)sqrt((x[idx].x - x[idx + 1].x)*(x[idx].x - x[idx + 1].x) + (x[idx].y - x[idx + 1].y)*(x[idx].y - x[idx + 1].y));

	// copy final var in x to output since D = 1
}
__global__ void DtransMultiply(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

	idx = max(idx, (long)1);
	idx = min(idx, (long)(Nslice*My*Mx - 2));

	output[idx].x = -x[idx - 1].x + x[idx].x;
	output[idx].y = -x[idx - 1].y + x[idx].y;
	// copy final var in x to output since D = 1
}
__global__ void muDivide(cufftDoubleComplex *x, cufftDoubleComplex *y, double mu) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	y[idx].x = x[idx].x / mu;
	y[idx].y = x[idx].y / mu;
}

__global__ void Checknorm(double *norm_u, double *out, double mu) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

	out[idx] = ((norm_u[idx] * norm_u[idx]) / (2 * mu)) * (norm_u[idx] < mu);
	if (out[idx] == 0) {
		out[idx] = (norm_u[idx] - mu / 2);
	}

}

__global__ void Checknorm_(const double* __restrict__ norm_u, double* __restrict__ out, double mu, int NN) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
#pragma unroll
	for (; idx < NN; idx += blockDim.x*gridDim.x) {
		out[idx] = ((norm_u[idx] * norm_u[idx]) / (2 * mu)) * (norm_u[idx] < mu);
		if (out[idx] == 0) {
			out[idx] = (norm_u[idx] - mu / 2);
		}
	}



}

__global__ void checkMu(double *norm_u, double mu, cufftDoubleComplex *u_mu, int Nslice, int My, int Mx) { //u_mu is Dx now, check-------------------

	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	idx = max(idx, (long)0);
	idx = min(idx, (long)(Nslice*My*Mx - 2));
	if (norm_u[idx] >= mu) {
		u_mu[idx].x = u_mu[idx].x * mu / norm_u[idx];
		u_mu[idx].y = u_mu[idx].y * mu / norm_u[idx];
	}

}

__global__ void checkMu_(const double* __restrict__ norm_u, double mu, cufftDoubleComplex* __restrict__ u_mu, int Nslice, int My, int Mx) { //u_mu is Dx now, check-------------------

	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, (long)0);
	//idx = min(idx, (long)(Nslice*My*Mx - 2));

#pragma unroll
	for (; idx <= Nslice * My*Mx - 2; idx += blockDim.x*gridDim.x) {
		if (norm_u[idx] >= mu) {
			u_mu[idx].x = u_mu[idx].x * mu / norm_u[idx];
			u_mu[idx].y = u_mu[idx].y * mu / norm_u[idx];
		}
	}


}

__global__ void mapping2GLubyte_range_val(double *in, int index_max, int index_min, double *out, double val) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	out[idx] = (255 * val / ((in[index_max - 1] - in[index_min - 1])))*(in[idx] - in[index_min - 1]);
}

__global__ void mapping2GLubyte_range(double *in, int index_max, int index_min, double *out) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	out[idx] = (255 / ((in[index_max - 1] - in[index_min - 1])))*(in[idx] - in[index_min - 1]);
}

__global__ void permute(double *in, double *out, int Nslice, int My, int Mx) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_in = (idx % Mx) * Nslice * My + ((idx / Mx) % My) * Nslice + idx / (Mx * My);
	out[idx] = in[idx_in];
}
__global__ void cast(double *in, unsigned char *out) {
	int idx_out = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_in = blockIdx.x*blockDim.x + threadIdx.x + 64 * 64 * 10 + 1;
	out[idx_out] = static_cast<unsigned char>(in[idx_in]);
}
__global__ void expand(double *in, unsigned char *out, int My, int Mx, int p_width) {
	int idx_in = blockIdx.x*blockDim.x + threadIdx.x;
	int Num = My * Mx * p_width;
	int idx_out = idx_in % Mx + ((idx_in / My) % Mx) * (Mx * p_width) + ((idx_in % Num) / (Mx * My)) * My + (idx_in / Num) * Num;
	out[idx_out] = static_cast<unsigned char>(in[idx_in]);
}

__global__ void expand2(double *in, unsigned char *out, double *out2, int My, int Mx, int p_width) {
	int idx_in = blockIdx.x*blockDim.x + threadIdx.x;
	int Num = My * Mx * p_width;
	int idx_out = idx_in % Mx + ((idx_in / My) % Mx) * (Mx * p_width) + ((idx_in % Num) / (Mx * My)) * My + (idx_in / Num) * Num;
	out[idx_out] = static_cast<unsigned char>(in[idx_in]);
	out2[idx_out] = in[idx_in];
}

__global__ void DtransMul(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {

}

__global__ void permuteSlice(cufftDoubleReal *x, cufftDoubleReal *output, int Nslice, int My, int Mx, int Nframe, int frame) {
	//x: 3d form (My, Mx, Nslice)
	//output: 4d form (My, Mx, Nframe, Nslice)
	//blockdim: 64x64 (My x Mx)
	output[(frame - 1)*blockDim.x + Nframe * Mx*My*blockIdx.x + threadIdx.x] = x[My*Mx*blockIdx.x + threadIdx.x];
}

//---------------------------------TEST NEW KERNELS-------------------------------------------------------
__global__ void step1a(const cufftDoubleComplex  *__restrict__ x, cufftDoubleComplex * __restrict__ Dx, double mu, int Nslice, int My, int Mx) {
	cufftDoubleComplex tmp_u;
	double tmp_norm;
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx > Nslice * My*Mx - 2) {
		return;
	}


	tmp_u.x = x[idx].x - x[idx + 1].x;
	tmp_u.y = x[idx].y - x[idx + 1].y;

	tmp_norm = sqrt(tmp_u.x*tmp_u.x + tmp_u.y*tmp_u.y);

	//u_mu
	Dx[idx].x = tmp_u.x / mu;
	Dx[idx].y = tmp_u.y / mu;

	if (tmp_norm >= mu) {
		Dx[idx].x = Dx[idx].x * mu / tmp_norm;
		Dx[idx].y = Dx[idx].y * mu / tmp_norm;
	}
}

//________________________________________________________________________________________________________

void calDy_(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx, cudaStream_t stream) {  //another better solution ????
	cudaMemcpyAsync(&output[0], &x[0], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, stream);
	cufftDoubleComplex *t;
	//gpuErrchk(cudaHostAlloc((cufftDoubleComplex**)&t, sizeof(cufftDoubleComplex), cudaHostAllocPortable));
	cudaMallocHost((cufftDoubleComplex**)&t, sizeof(cufftDoubleComplex));
	t[0].x = 0.0f;
	t[0].y = 0.0f;
	cudaMemcpyAsync(&t[0], &x[Nslice*My*Mx - 2], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	t[0].x = t[0].x * (-1);
	t[0].y = t[0].y * (-1);

	cudaMemcpyAsync(&output[Nslice*My*Mx - 1], &t[0], sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice, stream);
	cudaStreamSynchronize(stream);

	//delete[] & t;
	//cannot use delete with static variable. Also, it is unecessary since t deleted auto when goes out of scope. 





	DtransMultiply << < Nslice*My*Mx / 128, 128, 0, stream >> > (x, output, Nslice, My, Mx);
	//cudaDeviceSynchronize();


}

void calDy(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {  //another better solution ????
	cudaMemcpy(&output[0], &x[0], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cufftDoubleComplex t;
	t.x = 0.0f;
	t.y = 0.0f;
	cudaMemcpy(&t, &x[Nslice*My*Mx - 2], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	t.x = t.x * (-1);
	t.y = t.y * (-1);

	cudaMemcpy(&output[Nslice*My*Mx - 1], &t, sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);


	//delete[] & t;
	//cannot use delete with static variable. Also, it is unecessary since t deleted auto when goes out of scope. 





	DtransMultiply << < Nslice*My*Mx / 128, 128 >> > (x, output, Nslice, My, Mx);
	//cudaDeviceSynchronize();

}


#define NUM_STREAMS 3
#define maxit 2
long N;
//long nnz[4] = { 220800, 142592, 142592, 220992 }; // 64x64x112x31
//long nnz[4] = { 833024, 622976, 622336, 832256 }; // 128x128x112x31
//long nnz[4] = { 1273120, 999840, 1000480, 1272320 }; //160x160x112x31

//long nnz[4] = { 1790400, 1484736, 1483200, 1790208 }; //192x192x112x31

//long nnz[4] = { 2347072, 2107840, 2114560, 2349088 }; //224x224x112x32
//long nnz[4] = { 3000832, 2827264, 2817024, 2996480 }; // 256x256x112x32
//long nnz[4] = { 3722688, 3656448, 3654144, 3719520 }; // 288x288x112x32

//long nnz1[4] = { 1440800, 1440800, 1440800, 1440800 }; //160 - 0.5
//long nnz[4] = { 124928, 124928, 124928, 124928 };// 64 - 0.3
//long nnz[4] = { 187200, 187200, 187200, 187200 };// 64 - 0.5
//long nnz[4] = { 260608, 260608, 260608, 260608 };// 64 - 0.7
//long nnz[4] = { 614016, 614016, 614016, 614016 };// 128 - 0.3
//long nnz[4] = { 886528, 886528, 886528, 886528 };// 128 - 0.5
//long nnz[4] = { 1170688, 1170688, 1170688, 1170688 };// 128 - 0.7
//long nnz[4] = { 1023200, 1023200, 1023200, 1023200 }; //160 - 0.3
//long nnz[4] = { 1238880, 1238880, 1238880, 1238880 }; //160 - 0.4
//long nnz[4] = { 1440800, 1440800, 1440800, 1440800 }; //160 - 0.5
//long nnz[4] = { 1657280, 1657280, 1657280, 1657280 }; //160 - 0.6
//long nnz[4] = { 1901440, 1901440, 1901440, 1901440 }; //160 - 0.7
//long nnz[4] = { 2161344, 2161344, 2161344, 2161344 }; //192 - 0.5
long nnz;

int Nslice;
int My;
int Mx;
int Nframe;
float iternumCG = 250;
double tol = 1e-20;
char link1[200], link2[300];
float sumTime = 0;
int main() {
	DisplayHeader();
	StartCounter();
	//printInfo();
	info();

	// --- Creates CUDA streams
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

	// --- Creates cuFFT plans and sets them in streams
	cufftHandle* plans = (cufftHandle*)malloc(sizeof(cufftHandle)*NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++) {
		if (cufftPlan3d(&plans[i], Mx, My, Nslice, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT eror: PLan creation failed");
			return;
		}
		cufftSetStream(plans[i], streams[i]);
	}
	//--- Creates cuBlasHandle and sets streams
	cublasHandle_t *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t)*NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++) {
		if (cublasCreate(&handle[i]) != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLAS CREATE error");
			return;
		}

		cublasSetStream(handle[i], streams[i]);
	}

	//______________________________________________________________________________________________________________________________________________________________

	//init data
	cufftDoubleReal *ref_h;
	double **rho_h = new double *[NUM_STREAMS]; //concurrent 2 frames
	double *S_h; //Load 4 S_h first





	gpuErrchk(cudaHostAlloc((double**)&S_h, nnz * sizeof(double), cudaHostAllocPortable));

	for (int ii = 0; ii < NUM_STREAMS; ii++) {
		//gpuErrchk(cudaHostAlloc((double2**)&a_gs_h[ii], N * sizeof(double2), cudaHostAllocPortable));
		//gpuErrchk(cudaHostAlloc((double**)&rho_h[ii], N * sizeof(double), cudaHostAllocPortable));
		gpuErrchk(cudaMallocHost((double**)&rho_h[ii], N * sizeof(double)));
	}
	//gpuErrchk(cudaHostAlloc((double**)&ref_h, N * sizeof(double), cudaHostAllocPortable));
	gpuErrchk(cudaMallocHost((cufftDoubleReal**)&ref_h, N * sizeof(cufftDoubleReal)));

	//--------------------------------------------------------Open S_h and Ref first
	//sprintf(link1, "D:/cuda demo/Frame_data_test/192x192x112x32/S.txt");
	//FileOpen(link1, S_h);

	//FileOpen("D:/cuda demo/Frame_data_test/192x192x112x32/Frame_1.txt", ref_h);

	//FileOpen("D:/newS.txt", S_h);
	//FileOpen("D:/workspace/workspace/Frame_1.txt", ref_h);
	//sprintf(link1, "D:/Frames/%s/%s/S.txt", name, under);
	sprintf(link1, "%s/%s/S.txt", inURL, under);
	FileOpen(link1, S_h);

	//sprintf(link1, "D:/Frames/%s/Frame_1.txt", name);
	sprintf(link1, "%s/Frame_1.txt", inURL);
	FileOpen(link1, ref_h);

	//for (int kk = 0; kk < 100; kk++) {
	//	std::cout << "0               " << kk << "     " << S_h[0][kk] << std::endl;
	//}
	//for (int kk = 0; kk < 100; kk++) {
	//	std::cout << "1               " << kk << "     " << S_h[1][kk] << std::endl;
	//}



	//--------------------------Allocate CUDA MEM
	cufftDoubleReal *ref_d1, *ref_d2, *ref_d3;
	cufftDoubleComplex *data_d1;
	cufftDoubleComplex *data_d2;
	cufftDoubleComplex *data_d3;
	cufftDoubleComplex *y_d1;
	cufftDoubleComplex *y_d2;
	cufftDoubleComplex *y_d3;
	cufftDoubleReal **rho_d = new cufftDoubleReal *[NUM_STREAMS];
	double *S_d1;
	double *S_d2, *S_d3;

	gpuErrchk(cudaMalloc(&ref_d1, N * sizeof(cufftDoubleReal)));
	gpuErrchk(cudaMalloc(&ref_d2, N * sizeof(cufftDoubleReal)));
	gpuErrchk(cudaMalloc(&ref_d3, N * sizeof(cufftDoubleReal)));

	gpuErrchk(cudaMalloc(&S_d1, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc(&S_d2, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc(&S_d3, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc(&data_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&data_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&data_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&y_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&y_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&y_d3, nnz * sizeof(cufftDoubleComplex)));

	for (int ii = 0; ii < NUM_STREAMS; ii++) {
		gpuErrchk(cudaMalloc(&rho_d[ii], N * sizeof(cufftDoubleReal)));
	}

	cudaMemcpy(S_d1, S_h, nnz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(S_d2, S_d1, nnz * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(S_d3, S_d1, nnz * sizeof(double), cudaMemcpyDeviceToDevice);

	gpuErrchk(cudaFreeHost(S_h));

	std::cout << "______________________" << std::endl;
	//cudaEvent_t startEvent, stopEvent;
	//cudaEventCreate(&startEvent);
	//cudaEventCreate(&stopEvent);
	//cudaEventRecord(startEvent, 0);

	/*Sminus1 << <(nnz[0] / 64), 64, 0, streams[0] >> > (S_d1[0]);
	Sminus1 << <(nnz[1] / 64), 64, 0, streams[1] >> > (S_d1[1]);
	Sminus1 << <(nnz[2] / 64), 64, 0, streams[0] >> > (S_d1[2]);
	Sminus1 << <(nnz[3] / 64), 64, 0, streams[1] >> > (S_d1[3]);

	Sminus1 << <(nnz[0] / 64), 64, 0, streams[0] >> > (S_d2[0]);
	Sminus1 << <(nnz[1] / 64), 64, 0, streams[1] >> > (S_d2[1]);
	Sminus1 << <(nnz[2] / 64), 64, 0, streams[0] >> > (S_d2[2]);
	Sminus1 << <(nnz[3] / 64), 64, 0, streams[1] >> > (S_d2[3]);*/


	Sminus1_ << <((nnz + 1024 - 1) / 1024), 128, 0, streams[0] >> > (S_d1, nnz);

	Sminus1_ << <((nnz + 1024 - 1) / 1024), 128, 0, streams[1] >> > (S_d2, nnz);
	Sminus1_ << <((nnz + 1024 - 1) / 1024), 128, 0, streams[2] >> > (S_d3, nnz);
	//Sminus1_ << <(nnz[0] / 1024), 128, 0, streams[0] >> > (S_d2[0], nnz[0]);
	//Sminus1_ << <(nnz[1] / 1024), 128, 0, streams[1] >> > (S_d2[1], nnz[1]);
	//Sminus1_ << <(nnz[2] / 1024), 128, 0, streams[0] >> > (S_d2[2], nnz[2]);
	//Sminus1_ << <(nnz[3] / 1024), 128, 0, streams[1] >> > (S_d2[3], nnz[3]);

	//cudaEventRecord(stopEvent, 0);
	//cudaEventSynchronize(stopEvent);
	//float time;
	//cudaEventElapsedTime(&time, startEvent, stopEvent);
	//std::cout << "----------               " << time << std::endl;

	//double result;
	//cublasDasum(handle[0], nnz[0], S_d2[2], 1, &result);
	//cudaDeviceSynchronize();
	//std::cout << "----------   RS            " << result << std::endl;

	cudaMemcpy(ref_d1, ref_h, N * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
	cudaMemcpy(ref_d2, ref_d1, N * sizeof(cufftDoubleReal), cudaMemcpyDeviceToDevice);
	cudaMemcpy(ref_d3, ref_d1, N * sizeof(cufftDoubleReal), cudaMemcpyDeviceToDevice);
	gpuErrchk(cudaFreeHost(ref_h));


	//----------------Test for Unified memory -> almost equal to device memory----------------------
	//double *ref_um;
	//double2 *outtest_um_, *outtest_um;
	//cudaMallocManaged(&ref_um, N * sizeof(double));
	//cudaMallocManaged(&outtest_um, N * sizeof(double2));
	//cudaMallocManaged(&outtest_um_, N * sizeof(double2));
	//int device = -1;
	//cudaGetDevice(&device);
	//cudaMemPrefetchAsync(outtest_um_, N * sizeof(double2), device, streams[1]);
	//cudaMemPrefetchAsync(outtest_um, N * sizeof(double2), device, streams[1]);
	//cudaMemPrefetchAsync(ref_um, N * sizeof(double), device, streams[1]);

	//cudaMemcpyAsync(ref_um, ref_d1, N * sizeof(double), cudaMemcpyDeviceToDevice, streams[1]);

	//double2 *outtest_, *outtest;
	//cudaMalloc((void**)&outtest, N * sizeof(double2));
	//cudaMalloc((void**)&outtest_, N * sizeof(double2));
	//cudaEvent_t startEvent, stopEvent;
	//cudaEventCreate(&startEvent);
	//cudaEventCreate(&stopEvent);
	//cudaEventRecord(startEvent, 0);

	//cuFFTR2C_1(streams[1], plans[1], ref_d1, outtest_, outtest, Nslice, My, Mx);
	//cuFFTR2C_(streams[1], plans[1], ref_d1, outtest, Nslice, My, Mx);
	//cudaDeviceSynchronize();
	//cudaEventRecord(stopEvent, 0);
	//cudaEventSynchronize(stopEvent);
	//float time;
	//cudaEventElapsedTime(&time, startEvent, stopEvent);
	//std::cout << "----------               " << time << std::endl;

	//-----------------------------------------------------------------------------
	int nframe_ = 0;
	int n = 0;
	int iter = 0;

	cufftDoubleComplex *x_d1;
	cufftDoubleComplex *x_d2;
	cufftDoubleComplex *x_d3;
	cufftDoubleComplex *r_d1;
	cufftDoubleComplex *r_d2;
	cufftDoubleComplex *r_d3;
	cufftDoubleComplex *d_d1;
	cufftDoubleComplex *d_d2;
	cufftDoubleComplex *d_d3;
	cufftDoubleComplex *u_d1;
	cufftDoubleComplex *u_d2;
	cufftDoubleComplex *u_d3;
	cufftDoubleComplex *bestx_d1;
	cufftDoubleComplex *bestx_d2;
	cufftDoubleComplex *bestx_d3;
	cufftDoubleComplex *u1_d1;
	cufftDoubleComplex *u1_d2;
	cufftDoubleComplex *u1_d3;
	cufftDoubleComplex **outdata_ = new cufftDoubleComplex *[NUM_STREAMS];

	//cufftDoubleComplex *xtest1, *xtest2;
	//cudaMalloc(&xtest1, nnz[2] * sizeof(cufftDoubleComplex));
	//cudaMalloc(&xtest2, nnz[2] * sizeof(cufftDoubleComplex));

	gpuErrchk(cudaMalloc(&x_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&x_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&x_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&r_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&r_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&r_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&d_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&d_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&d_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&bestx_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&bestx_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&bestx_d3, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u1_d1, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u1_d2, nnz * sizeof(cufftDoubleComplex)));
	gpuErrchk(cudaMalloc(&u1_d3, nnz * sizeof(cufftDoubleComplex)));

	for (int ii = 0; ii < NUM_STREAMS; ii++) {
		gpuErrchk(cudaMalloc(&outdata_[ii], N * sizeof(cufftDoubleComplex)));
	}


	////---------------------------ALlocate abs value for rho---------------------------------------------------------
	cufftDoubleComplex **rho_gs_d = new cufftDoubleComplex *[NUM_STREAMS];
	cufftDoubleReal **rho_abs = new cufftDoubleReal*[NUM_STREAMS];
	cufftDoubleReal **rho_mapping = new cufftDoubleReal*[NUM_STREAMS];
	cufftDoubleReal **rho_permute = new cufftDoubleReal*[NUM_STREAMS];
	cufftDoubleReal **rho_mapping_ = new cufftDoubleReal*[NUM_STREAMS];
	cufftDoubleReal **rho_permute_ = new cufftDoubleReal*[NUM_STREAMS];
	cufftDoubleReal **rho_mapping_error = new cufftDoubleReal*[NUM_STREAMS];
	//cufftDoubleReal **rho_permute_error = new cufftDoubleReal*[NUM_STREAMS];
	//unsigned char **checkImage_h = new unsigned char *[NUM_STREAMS];
	unsigned char **checkImage_d = new unsigned char *[NUM_STREAMS];
	unsigned char *checkImage_h1, *checkImage_h2, *checkImage_h3;
	for (int ii = 0; ii < NUM_STREAMS; ii++) {
		gpuErrchk(cudaMalloc(&rho_gs_d[ii], N * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&rho_abs[ii], N * sizeof(cufftDoubleReal)));
		gpuErrchk(cudaMalloc(&rho_mapping[ii], N * sizeof(cufftDoubleReal)));
		gpuErrchk(cudaMalloc(&rho_permute[ii], N * sizeof(cufftDoubleReal)));
		gpuErrchk(cudaMalloc(&rho_mapping_[ii], N * sizeof(cufftDoubleReal)));
		gpuErrchk(cudaMalloc(&rho_permute_[ii], N * sizeof(cufftDoubleReal)));
		gpuErrchk(cudaMalloc(&rho_mapping_error[ii], N * sizeof(cufftDoubleReal)));
		//	gpuErrchk(cudaMalloc(&rho_permute_error[ii], N * sizeof(cufftDoubleReal)));
		//gpuErrchk(cudaHostAlloc(&checkImage_h, N * sizeof(unsigned char), cudaHostAllocWriteCombined));
		
		gpuErrchk(cudaMalloc(&checkImage_d[ii], N * sizeof(unsigned char)));
	}

	gpuErrchk(cudaHostAlloc(&checkImage_h1, N * sizeof(unsigned char), cudaHostAllocWriteCombined));
	gpuErrchk(cudaHostAlloc(&checkImage_h2, N * sizeof(unsigned char), cudaHostAllocWriteCombined));
	gpuErrchk(cudaHostAlloc(&checkImage_h3, N * sizeof(unsigned char), cudaHostAllocWriteCombined));
	


	int index_max[NUM_STREAMS], index_min[NUM_STREAMS];
	size_t size_Image = Nslice * My*Mx * sizeof(unsigned char);

	int k;
	//cufftDoubleComplex **b = new cufftDoubleComplex*[NUM_STREAMS];
	//for (int iii = 0; iii < NUM_STREAMS; iii++) {
	//	cudaMalloc(&b[iii], nnz[3] * sizeof(cufftDoubleComplex));
	//}


	char linka1[300], linka2[300];
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	int loop = 0;
	double sumNom = { 0.0 };
	double sumDen = { 0.0 };
	for (int nframe = 2; nframe < (Nframe - 2) - (Nframe - 2) % 3; nframe = nframe + 3) {

		//-----Assume that we have rho_h data here
		double delta0[NUM_STREAMS], delta[NUM_STREAMS];
		double bestres[NUM_STREAMS];
		double gamma[NUM_STREAMS], res[NUM_STREAMS];
		double alpha[NUM_STREAMS];
		double delta_old[NUM_STREAMS];
		cufftDoubleComplex numerator[NUM_STREAMS], denom[NUM_STREAMS], delta_[NUM_STREAMS];

		for (int i = 0; i < NUM_STREAMS; i++) {
			numerator[i] = make_cuDoubleComplex(0, 0);
			denom[i] = make_cuDoubleComplex(0, 0);
			delta_[i] = make_cuDoubleComplex(0, 0);
			bestres[i] = 1;
			alpha[i] = 0;
		}
		//std::cout << "-----------------" << std::endl;

		//k = 0;

		//k = 3;

		/*size_t free_byte;

		size_t total_byte;

		cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

		if (cudaSuccess != cuda_status) {

			printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));

			exit(1);

		}



		double free_db = (double)free_byte;

		double total_db = (double)total_byte;

		double used_db = total_db - free_db;

		printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

			used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);*/

		//int k = 2;


		//StartCounter();
		//sprintf(link1, "D:/cuda demo/Frame_data_test/192x192x112x32/Frame_%d.txt", nframe);
		//sprintf(link1, "D:/workspace/workspace/Frame_%d.txt", nframe);
		//FileOpen(link1, rho_h[0]);
		//sprintf(linka1, "D:/cuda demo/Frame_data_test/192x192x112x32/Frame_%d.txt", nframe + 1);
		//sprintf(linka1, "D:/workspace/workspace/Frame_%d.txt", (nframe + 1));
		//FileOpen(linka1, rho_h[1]);
		//sprintf(linka1, "D:/cuda demo/Frame_data_test/192x192x112x32/Frame_%d.txt", nframe + 2);
		//sprintf(linka1, "D:/workspace/workspace/Frame_%d.txt", (nframe + 2));
		//FileOpen(linka1, rho_h[2]);

		//sprintf(link1, "D:/Frames/%s/Frame_%d.txt", name, nframe);
		sprintf(link1, "%s/Frame_%d.txt", inURL, nframe);
		FileOpen(link1, rho_h[0]);
		if ((nframe + 1) < Nframe) {
			//n = sprintf(link1, "D:/workspace/workspace/Frame_%d.txt", (nframe + 1));
			//sprintf(link1, "D:/Frames/%s/Frame_%d.txt", name, nframe + 1);
			sprintf(link1, "%s/Frame_%d.txt", inURL, nframe + 1);
			FileOpen(link1, rho_h[1]);
		}
		if ((nframe + 2) <= Nframe) {
			//n = sprintf(link1, "D:/workspace/workspace/Frame_%d.txt", (nframe + 2));
			//sprintf(link1, "D:/Frames/%s/Frame_%d.txt", name, nframe + 2);
			sprintf(link1, "%s/Frame_%d.txt", inURL, nframe + 2);
			FileOpen(link1, rho_h[2]);
		}

		//std::cout << "-----------------" << GetCounter() << std::endl;
		//for (int kk = 0; kk < 100; kk++) {
		//	std::cout <<"0               " <<kk << "     "<< rho_h[0][kk] << std::endl;
		//}
		//for (int kk = 0; kk < 100; kk++) {
		//	std::cout <<"1               "<< kk << "     " << rho_h[1][kk] << std::endl;
		//}
		

		//Copy rho to device pointer
		gpuErrchk(cudaMemcpyAsync(rho_d[0], rho_h[0], N * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(rho_d[1], rho_h[1], N * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice, streams[1]));
		gpuErrchk(cudaMemcpyAsync(rho_d[2], rho_h[2], N * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice, streams[2]));




		

		AopReal(Nslice, My, Mx, plans[0], handle[0], streams[0], rho_d[0], outdata_[0], data_d1, S_d1, nnz);
		AopReal(Nslice, My, Mx, plans[1], handle[1], streams[1], rho_d[1], outdata_[1], data_d2, S_d2, nnz);
		AopReal(Nslice, My, Mx, plans[2], handle[2], streams[2], rho_d[2], outdata_[2], data_d3, S_d3, nnz);

		
		/*if (k == 3) {
		gpuErrchk(cudaMemcpyAsync(b[0], data_d1[k], nnz[3] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(b[1], data_d2[k], nnz[3] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[1]));
		}*/

		//cudaDeviceSynchronize();
		//cudaStreamSynchronize();

		Bhop_gs(Nslice, My, Mx, plans[0], handle[0], streams[0], ref_d1, data_d1, outdata_[0], y_d1, S_d1, nnz);
		Bhop_gs(Nslice, My, Mx, plans[1], handle[1], streams[1], ref_d2, data_d2, outdata_[1], y_d2, S_d2, nnz);
		Bhop_gs(Nslice, My, Mx, plans[2], handle[2], streams[2], ref_d3, data_d3, outdata_[2], y_d3, S_d3, nnz);

		//-------------------------------------CG---------------------------------------------------------
		cudaEventRecord(startEvent, 0);

		//Memset x_d[NUM_STREAMS]
		//cudaDeviceSynchronize();

		//cudaDeviceSynchronize();

		gpuErrchk(cudaMemsetAsync(x_d1, 0, nnz * sizeof(cufftDoubleComplex), streams[0]));
		gpuErrchk(cudaMemsetAsync(x_d2, 0, nnz * sizeof(cufftDoubleComplex), streams[1]));
		gpuErrchk(cudaMemsetAsync(x_d3, 0, nnz * sizeof(cufftDoubleComplex), streams[2]));


		//gpuErrchk(cudaMemsetAsync(xtest1, 0, nnz[k] * sizeof(cufftDoubleComplex), streams[0]));
		//gpuErrchk(cudaMemsetAsync(xtest2, 0, nnz1[k] * sizeof(cufftDoubleComplex), streams[1]));

		gpuErrchk(cudaMemcpyAsync(d_d1, y_d1, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(d_d2, y_d2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[1]));
		gpuErrchk(cudaMemcpyAsync(d_d3, y_d3, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[2]));

		gpuErrchk(cudaMemcpyAsync(r_d1, y_d1, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(r_d2, y_d2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[1]));
		gpuErrchk(cudaMemcpyAsync(r_d3, y_d3, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[2]));

		gpuErrchk(cudaMemcpyAsync(bestx_d1, x_d1, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(bestx_d2, x_d2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[1]));
		gpuErrchk(cudaMemcpyAsync(bestx_d3, x_d3, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[2]));

		//host initialization 

		//std::cout << _cudaGetErrorEnum(cublasZdotc(handle[0], nnz[k], y_d1[k], 1, y_d1[k], 1, &delta_[0])) << std::endl;
		//std::cout << _cudaGetErrorEnum(cublasZdotc(handle[1], nnz1[k], y_d2[k], 1, y_d2[k], 1, &delta_[1])) << std::endl;
		cublasZdotc(handle[0], nnz, r_d1, 1, r_d1, 1, &delta_[0]);
		cublasZdotc(handle[1], nnz, r_d2, 1, r_d2, 1, &delta_[1]);
		cublasZdotc(handle[2], nnz, r_d3, 1, r_d3, 1, &delta_[2]);

		
		for (int i = 0; i < NUM_STREAMS; i++) {
			delta0[i] = delta_[i].x;
			delta[i] = delta0[i];
		}

		for (int iter_cg = 2; iter_cg < iternumCG; iter_cg++) {
			Bhop_gs(Nslice, My, Mx, plans[0], handle[0], streams[0], ref_d1, d_d1, outdata_[0], u1_d1, S_d1, nnz);
			Bhop_gs(Nslice, My, Mx, plans[1], handle[1], streams[1], ref_d2, d_d2, outdata_[1], u1_d2, S_d2, nnz);
			Bhop_gs(Nslice, My, Mx, plans[2], handle[2], streams[2], ref_d3, d_d3, outdata_[2], u1_d3, S_d3, nnz);

			Bhop_gs(Nslice, My, Mx, plans[0], handle[0], streams[0], ref_d1, u1_d1, outdata_[0], u_d1, S_d1, nnz);
			Bhop_gs(Nslice, My, Mx, plans[1], handle[1], streams[1], ref_d2, u1_d2, outdata_[1], u_d2, S_d2, nnz);
			Bhop_gs(Nslice, My, Mx, plans[2], handle[2], streams[2], ref_d3, u1_d3, outdata_[2], u_d3, S_d3, nnz);

			cublasER(cublasZdotc(handle[0], nnz, d_d1, 1, r_d1, 1, &numerator[0]));
			cublasER(cublasZdotc(handle[1], nnz, d_d2, 1, r_d2, 1, &numerator[1]));
			cublasER(cublasZdotc(handle[2], nnz, d_d3, 1, r_d3, 1, &numerator[2]));

			cublasER(cublasZdotc(handle[0], nnz, d_d1, 1, u_d1, 1, &denom[0]));
			cublasER(cublasZdotc(handle[1], nnz, d_d2, 1, u_d2, 1, &denom[1]));
			cublasER(cublasZdotc(handle[2], nnz, d_d3, 1, u_d3, 1, &denom[2]));

			for (int i = 0; i < NUM_STREAMS; i++) {
				alpha[i] = (double)(numerator[i].x / denom[i].x);
				if (isnan(alpha[i])) {
					alpha[i] = 0;
				}
			}

			cublasER(cublasZaxpy(handle[0], nnz, &make_cuDoubleComplex(alpha[0], 0), d_d1, 1, x_d1, 1));
			cublasER(cublasZaxpy(handle[1], nnz, &make_cuDoubleComplex(alpha[1], 0), d_d2, 1, x_d2, 1));
			cublasER(cublasZaxpy(handle[2], nnz, &make_cuDoubleComplex(alpha[2], 0), d_d3, 1, x_d3, 1));

			cublasER(cublasZaxpy(handle[0], nnz, &make_cuDoubleComplex(-alpha[0], 0), u_d1, 1, r_d1, 1));
			cublasER(cublasZaxpy(handle[1], nnz, &make_cuDoubleComplex(-alpha[1], 0), u_d2, 1, r_d2, 1));
			cublasER(cublasZaxpy(handle[2], nnz, &make_cuDoubleComplex(-alpha[2], 0), u_d3, 1, r_d3, 1));

			for (int i = 0; i < NUM_STREAMS; i++) {
				delta_old[i] = delta[i];
			}

			cublasER(cublasZdotc(handle[0], nnz, r_d1, 1, r_d1, 1, &delta_[0]));
			cublasER(cublasZdotc(handle[1], nnz, r_d2, 1, r_d2, 1, &delta_[1]));
			cublasER(cublasZdotc(handle[2], nnz, r_d3, 1, r_d3, 1, &delta_[2]));

			for (int i = 0; i < NUM_STREAMS; i++) {
				delta[i] = delta_[i].x;
				gamma[i] = (double)delta[i] / delta_old[i];
			}

			cublasER(cublasZaxpy(handle[0], nnz, &make_cuDoubleComplex(1 / gamma[0], 0), r_d1, 1, d_d1, 1));
			cublasER(cublasZaxpy(handle[1], nnz, &make_cuDoubleComplex(1 / gamma[1], 0), r_d2, 1, d_d2, 1));
			cublasER(cublasZaxpy(handle[2], nnz, &make_cuDoubleComplex(1 / gamma[2], 0), r_d3, 1, d_d3, 1));

			cublasER(cublasZdscal(handle[0], nnz, &gamma[0], d_d1, 1));
			cublasER(cublasZdscal(handle[1], nnz, &gamma[1], d_d2, 1));
			cublasER(cublasZdscal(handle[2], nnz, &gamma[2], d_d3, 1));

			for (int i = 0; i < NUM_STREAMS; i++) {
				res[i] = (double)sqrt(delta[i] / delta0[i]);
				//std::cout << "STREAMS " << i << " " << res[i] << std::endl;
			}

			// check for bestres -- up to now cannot find a better way 
			//Should we use cudaMemcpyAsync
			if (res[0] < bestres[0]) {
				cudaMemcpyAsync(bestx_d1, x_d1, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]);
				bestres[0] = res[0];
			}
			if (res[1] < bestres[1]) {
				cudaMemcpyAsync(bestx_d2, x_d2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[1]);
				bestres[1] = res[1];
			}
			if (res[2] < bestres[2]) {
				cudaMemcpyAsync(bestx_d3, x_d3, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, streams[2]);
				bestres[2] = res[2];
			}

			// check for tolerance
			for (int i = 0; i < NUM_STREAMS; i++) {
				if (res[i] < tol) {

					break;
				}

			}



		}
		//_________________________WE can check bestx_d here______________________________________
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		float time;
		cudaEventElapsedTime(&time, startEvent, stopEvent);
		sumTime += time;
		std::cout << "----------               " << time << std::endl;
		//std::cout << "-----------------------------" << std::endl;
		Iop_gs(Nslice, My, Mx, plans[0], handle[0], streams[0], ref_d1, bestx_d1, rho_gs_d[0], S_d1, nnz);
		Iop_gs(Nslice, My, Mx, plans[1], handle[1], streams[1], ref_d2, bestx_d2, rho_gs_d[1], S_d2, nnz);
		Iop_gs(Nslice, My, Mx, plans[2], handle[2], streams[2], ref_d3, bestx_d3, rho_gs_d[2], S_d3, nnz);
		//Right results for CG--------------------------------------





		int index_min_[NUM_STREAMS], index_max_[NUM_STREAMS];

		abs_matrix_ << <N / 1024, 128, 0, streams[0] >> > (rho_gs_d[0], rho_abs[0], Nslice, My, Mx, 0);
		abs_matrix_ << <N / 1024, 128, 0, streams[1] >> > (rho_gs_d[1], rho_abs[1], Nslice, My, Mx, 0);
		abs_matrix_ << <N / 1024, 128, 0, streams[2] >> > (rho_gs_d[2], rho_abs[2], Nslice, My, Mx, 0);

		cublasER(cublasIdamax(handle[0], Nslice*My*Mx, rho_abs[0], 1, &index_max[0]));
		cublasER(cublasIdamax(handle[1], Nslice*My*Mx, rho_abs[1], 1, &index_max[1]));
		cublasER(cublasIdamax(handle[2], Nslice*My*Mx, rho_abs[2], 1, &index_max[2]));

		cublasER(cublasIdamin(handle[0], Nslice*My*Mx, rho_abs[0], 1, &index_min[0]));
		cublasER(cublasIdamin(handle[1], Nslice*My*Mx, rho_abs[1], 1, &index_min[1]));
		cublasER(cublasIdamin(handle[2], Nslice*My*Mx, rho_abs[2], 1, &index_min[2]));

		mapping2GLubyte_range << <N / 128, 128, 0, streams[0] >> > (rho_abs[0], index_max[0], index_min[0], rho_mapping[0]);
		mapping2GLubyte_range << <N / 128, 128, 0, streams[1] >> > (rho_abs[1], index_max[1], index_min[1], rho_mapping[1]);
		mapping2GLubyte_range << <N / 128, 128, 0, streams[2] >> > (rho_abs[2], index_max[2], index_min[2], rho_mapping[2]);

		permute << <N / 128, 128, 0, streams[0] >> > (rho_mapping[0], rho_permute[0], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[1] >> > (rho_mapping[1], rho_permute[1], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[2] >> > (rho_mapping[2], rho_permute[2], Nslice, My, Mx);

		expand2 << <N / 128, 128, 0, streams[0] >> > (rho_permute[0], checkImage_d[0], rho_mapping_error[0], My, Mx, p_width);
		expand2 << <N / 128, 128, 0, streams[1] >> > (rho_permute[1], checkImage_d[1], rho_mapping_error[1], My, Mx, p_width);
		expand2 << <N / 128, 128, 0, streams[2] >> > (rho_permute[2], checkImage_d[2], rho_mapping_error[2], My, Mx, p_width);

		gpuErrchk(cudaMemcpyAsync(checkImage_h1, checkImage_d[0], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h2, checkImage_d[1], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[1]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h3, checkImage_d[2], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[2]));

		for (int i = 0; i < NUM_STREAMS; i++)
			gpuErrchk(cudaStreamSynchronize(streams[i]));

	//	sprintf(link2, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/GS/Frame_(%d, %d, %d).bmp", 2, 4, nframe);
	//	saveBitmap(link2, 6, 5, My, My, 4, 2, checkImage_h1);

	//	sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/GS/Frame_(%d, %d, %d).bmp", 2, 4, nframe + 1);
	//	saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h2);

	//	sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/GS/Frame_(%d, %d, %d).bmp", 2, 4, nframe + 2);
	//	saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h3);

		//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe);
		sprintf(link2, "%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe);
		saveBitmap(link2, p_width, p_height, My, Mx, x_cor, y_cor, checkImage_h1);

		//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-full/Frame_%d.bmp", outputname, method, imgres, nframe);
		sprintf(link2, "%s/Rcst-full/Frame_%d.bmp", outURL, nframe);
		saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h1);
		if ((nframe + 1) < Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 1);
			sprintf(link2, "%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 1);
			saveBitmap(link2, p_width, p_height, My, Mx, x_cor, y_cor, checkImage_h2);

			//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-full/Frame_%d.bmp", outputname, method, imgres, nframe + 1);
			sprintf(link2, "%s/Rcst-full/Frame_%d.bmp", outURL, nframe + 1);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h2);
		}
		if ((nframe + 2) <= Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 2);
			sprintf(link2, "%s/Rcst-spc/Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 2);
			saveBitmap(link2, p_width, p_height, My, Mx, x_cor, y_cor, checkImage_h3);

			//sprintf(link2, "D:/Results/%s/%s/%s/Rcst-full/Frame_%d.bmp", outputname, method, imgres, nframe + 2);
			sprintf(link2, "%s/Rcst-full/Frame_%d.bmp", outURL, nframe + 2);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h3);
		}

		//sprintf(link2, "D:/workspace/newtest/64x64x112x32/test/Frame_%d.bmp", nframe + j);
		//saveBitmap(link2, 16 * My, 7 * My, checkImage_h1);
		/*if ((nframe + j + 4) <= 32) {
		n = sprintf(link2, "D:/workspace/test/Frame_%d.bmp", (nframe + j + 4));
		saveBitmap(link2, 16 * 64, 7 * 64, checkImage_h2);
		}*/

		double *test1, *test2, *test3, *test4, *test5, *test6;
		cudaMallocHost(&test1, N * sizeof(double));
		cudaMallocHost(&test2, N * sizeof(double));
		cudaMallocHost(&test3, N * sizeof(double));
		cudaMallocHost(&test4, N * sizeof(double));
		cudaMallocHost(&test5, N * sizeof(double));
		cudaMallocHost(&test6, N * sizeof(double));

		gpuErrchk(cudaMemcpyAsync(test1, rho_mapping_error[0], N * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(test3, rho_mapping_error[1], N * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
		gpuErrchk(cudaMemcpyAsync(test5, rho_mapping_error[2], N * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));

		cublasER(cublasIdamax(handle[0], Nslice*My*Mx, rho_d[0], 1, &index_max_[0]));
		cublasER(cublasIdamax(handle[1], Nslice*My*Mx, rho_d[1], 1, &index_max_[1]));
		cublasER(cublasIdamax(handle[1], Nslice*My*Mx, rho_d[2], 1, &index_max_[2]));

		cublasER(cublasIdamin(handle[0], Nslice*My*Mx, rho_d[0], 1, &index_min_[0]));
		cublasER(cublasIdamin(handle[1], Nslice*My*Mx, rho_d[1], 1, &index_min_[1]));
		cublasER(cublasIdamin(handle[1], Nslice*My*Mx, rho_d[2], 1, &index_min_[2]));

		mapping2GLubyte_range << <N / 128, 128, 0, streams[0] >> > (rho_d[0], index_max_[0], index_min_[0], rho_mapping_[0]);
		mapping2GLubyte_range << <N / 128, 128, 0, streams[1] >> > (rho_d[1], index_max_[1], index_min_[1], rho_mapping_[1]);
		mapping2GLubyte_range << <N / 128, 128, 0, streams[2] >> > (rho_d[2], index_max_[2], index_min_[2], rho_mapping_[2]);

		permute << <N / 128, 128, 0, streams[0] >> > (rho_mapping_[0], rho_permute[0], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[1] >> > (rho_mapping_[1], rho_permute[1], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[2] >> > (rho_mapping_[2], rho_permute[2], Nslice, My, Mx);

		expand2 << <N / 128, 128, 0, streams[0] >> > (rho_permute[0], checkImage_d[0], rho_mapping_error[0], My, Mx, p_width);
		expand2 << <N / 128, 128, 0, streams[1] >> > (rho_permute[1], checkImage_d[1], rho_mapping_error[1], My, Mx, p_width);
		expand2 << <N / 128, 128, 0, streams[2] >> > (rho_permute[2], checkImage_d[2], rho_mapping_error[2], My, Mx, p_width);

		gpuErrchk(cudaMemcpyAsync(checkImage_h1, checkImage_d[0], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h2, checkImage_d[1], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[1]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h3, checkImage_d[2], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[2]));

		gpuErrchk(cudaMemcpyAsync(test2, rho_mapping_error[0], N * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(test4, rho_mapping_error[1], N * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
		gpuErrchk(cudaMemcpyAsync(test6, rho_mapping_error[2], N * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));


		for (int i = 0; i < NUM_STREAMS; i++)
			gpuErrchk(cudaStreamSynchronize(streams[i]));

		//sprintf(link2, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Ref/RefFrame_(%d, %d, %d).bmp", 2, 4, nframe);
		//saveBitmap(link2, 6, 5, My, My, 4, 2, checkImage_h1);

		//sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Ref/RefFrame_(%d, %d, %d).bmp", 2, 4, nframe + 1);
		//saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h2);

		//sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Ref/RefFrame_(%d, %d, %d).bmp", 2, 4, nframe + 2);
		//saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h3);

		//sprintf(link2, "D:/Results/%s/%s/%s/Ref-spc/RefFrame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe);
		sprintf(link2, "%s/Ref-spc/RefFrame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe);
		saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h1);

		//sprintf(link2, "D:/Results/%s/%s/%s/Ref-full/RefFrame_%d.bmp", outputname, method, imgres, nframe);
		sprintf(link2, "%s/Ref-full/RefFrame_%d.bmp", outURL, nframe);
		saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h1);

		if ((nframe + 1) < Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Ref-spc/RefFrame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 1);
			sprintf(link2, "%s/Ref-spc/RefFrame_(%d, %d, %d).bmp.bmp", outURL, y_cor, x_cor, nframe + 1);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h2);

			//sprintf(link2, "D:/Results/%s/%s/%s/Ref-full/RefFrame_%d.bmp", outputname, method, imgres, nframe + 1);
			sprintf(link2, "%s/Ref-full/RefFrame_%d.bmp", outURL, nframe + 1);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h2);
		}
		if ((nframe + 2) <= Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Ref-spc/RefFrame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 2);
			sprintf(link2, "%s/Ref-spc/RefFrame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 2);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h3);

			//sprintf(link2, "D:/Results/%s/%s/%s/Ref-full/RefFrame_%d.bmp", outputname, method, imgres, nframe + 2);
			sprintf(link2, "%s/Ref-full/RefFrame_%d.bmp", outURL, nframe + 2);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h3);
		}

		//saveBitmap(link2, 16 * My, 7 * My, checkImage_h1);
		/*if ((nframe + j + 4) <= 32) {
		n = sprintf(link2, "D:/workspace/ref/RefFrame_%d.bmp", (nframe + j + 4));
		saveBitmap(link2, 16 * 64, 7 * 64, checkImage_h2);
		}*/

		//============================================individual Error========================================================================
		std::cout << "individual error Frame " << nframe << "   " << nrmse(test2, test1, p_width, p_height, Mx, My, x_cor, y_cor) << std::endl;
		std::cout << "individual error Frame " << nframe + 1 << "   " << nrmse(test4, test3, p_width, p_height, Mx, My, x_cor, y_cor) << std::endl;
		std::cout << "individual error Frame " << nframe + 2 << "   " << nrmse(test6, test5, p_width, p_height, Mx, My, x_cor, y_cor) << std::endl;
		//====================================================================================================================================
		minusAbs << <N / 1024, 128, 0, streams[0] >> > (rho_d[0], rho_abs[0], N);
		minusAbs << <N / 1024, 128, 0, streams[1] >> > (rho_d[1], rho_abs[1], N);
		minusAbs << <N / 1024, 128, 0, streams[2] >> > (rho_d[2], rho_abs[2], N);

		minusAbs << <N / 1024, 128, 0, streams[0] >> > (rho_mapping_[0], rho_mapping[0], N);
		minusAbs << <N / 1024, 128, 0, streams[1] >> > (rho_mapping_[1], rho_mapping[1], N);
		minusAbs << <N / 1024, 128, 0, streams[2] >> > (rho_mapping_[2], rho_mapping[2], N);

		double results[NUM_STREAMS], results_[NUM_STREAMS];
		cublasER(cublasDnrm2(handle[0], N, rho_abs[0], 1, &results_[0]));
		cublasER(cublasDnrm2(handle[0], N, rho_d[0], 1, &results[0]));
		cublasER(cublasDnrm2(handle[1], N, rho_abs[1], 1, &results_[1]));
		cublasER(cublasDnrm2(handle[1], N, rho_d[1], 1, &results[1]));
		cublasER(cublasDnrm2(handle[2], N, rho_abs[2], 1, &results_[2]));
		cublasER(cublasDnrm2(handle[2], N, rho_d[2], 1, &results[2]));

		for (int i = 0; i < NUM_STREAMS; i++) {
			sumNom += results_[i] * results_[i];
			sumDen += results[i] * results[i];
		}


		std::cout << "l2 norm error 1  Frame " << nframe << "   " << results_[0] / results[0] << std::endl;
		std::cout << "l2 norm error 1  Frame " << nframe + 1 << "   " << results_[1] / results[1] << std::endl;
		std::cout << "l2 norm error 1  Frame " << nframe + 2 << "   " << results_[2] / results[2] << std::endl;

		cublasER(cublasDnrm2(handle[0], N, rho_mapping[0], 1, &results_[0]));
		cublasER(cublasDnrm2(handle[0], N, rho_mapping_[0], 1, &results[0]));
		cublasER(cublasDnrm2(handle[1], N, rho_mapping[1], 1, &results_[1]));
		cublasER(cublasDnrm2(handle[1], N, rho_mapping_[1], 1, &results[1]));
		cublasER(cublasDnrm2(handle[2], N, rho_mapping[2], 1, &results_[2]));
		cublasER(cublasDnrm2(handle[2], N, rho_mapping_[2], 1, &results[2]));

		std::cout << "l2 norm error 2 Frame   " << nframe << "   " << results_[0] / results[0] << std::endl;
		std::cout << "l2 norm error 2 Frame   " << nframe + 1 << "   " << results_[1] / results[1] << std::endl;
		std::cout << "l2 norm error 2 Frame   " << nframe + 2 << "   " << results_[2] / results[2] << std::endl;

		//std::cout << "-----------------" << GetCounter() << std::endl;
		//------------------------------------------Error 1 -------------------------------------------------------
		permute << <N / 128, 128, 0, streams[0] >> > (rho_mapping[0], rho_permute[0], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[1] >> > (rho_mapping[1], rho_permute[1], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[2] >> > (rho_mapping[2], rho_permute[2], Nslice, My, Mx);

		expand << <N / 128, 128, 0, streams[0] >> > (rho_permute[0], checkImage_d[0], My, Mx, p_width);
		expand << <N / 128, 128, 0, streams[1] >> > (rho_permute[1], checkImage_d[1], My, Mx, p_width);
		expand << <N / 128, 128, 0, streams[2] >> > (rho_permute[2], checkImage_d[2], My, Mx, p_width);

		gpuErrchk(cudaMemcpyAsync(checkImage_h1, checkImage_d[0], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h2, checkImage_d[1], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[1]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h3, checkImage_d[2], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[2]));

		for (int i = 0; i < NUM_STREAMS; i++)
			gpuErrchk(cudaStreamSynchronize(streams[i]));

		//sprintf(link2, "D:/workspace/newtest/128x128x112x32/error 1/ErrorFrame_%d.bmp", nframe + j);
		//saveBitmap(link2, 16 * My, 7 * My, checkImage_h1);
		/*if ((nframe + j + 4) <= 32) {
		n = sprintf(link2, "D:/workspace/error 2/ErrorFrame_%d.bmp", (nframe + j + 4));
		saveBitmap(link2, 16 * 64, 7 * 64, checkImage_h2);
		}*/

		//sprintf(link2, "D:/Results/%s/%s/%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe);
		sprintf(link2, "%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe);
		saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h1);

		//sprintf(link2, "D:/Results/%s/%s/%s/Error1-full/Error1_Frame_%d.bmp", outputname, method, imgres, nframe);
		sprintf(link2, "%s/Error1-full/Error1_Frame_%d.bmp", outURL, nframe);
		saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h1);

		if ((nframe + 1) < Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 1);
			sprintf(link2, "%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 1);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h2);

			//sprintf(link2, "D:/Results/%s/%s/%s/Error1-full/Error1_Frame_%d.bmp", outputname, method, imgres, nframe + 1);
			sprintf(link2, "%s/Error1-full/Error1_Frame_%d.bmp", outURL, nframe + 1);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h2);
		}
		if ((nframe + 2) <= Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 2);
			sprintf(link2, "%s/Error1-spc/Error1_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 2);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h3);

			//sprintf(link2, "D:/Results/%s/%s/%s/Error1-full/Error1_Frame_%d.bmp", outputname, method, imgres, nframe + 2);
			sprintf(link2, "%s/Error1-full/Error1_Frame_%d.bmp", outURL, nframe + 2);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h3);
		}

		//------------------------------Error 2------------------------------------

		cublasER(cublasIdamax(handle[0], Nslice*My*Mx, rho_abs[0], 1, &index_max[0]));
		cublasER(cublasIdamax(handle[1], Nslice*My*Mx, rho_abs[1], 1, &index_max[1]));
		cublasER(cublasIdamax(handle[2], Nslice*My*Mx, rho_abs[2], 1, &index_max[2]));

		cublasER(cublasIdamin(handle[0], Nslice*My*Mx, rho_abs[0], 1, &index_min[0]));
		cublasER(cublasIdamin(handle[1], Nslice*My*Mx, rho_abs[1], 1, &index_min[1]));
		cublasER(cublasIdamin(handle[2], Nslice*My*Mx, rho_abs[2], 1, &index_min[2]));
		//index_max[0] = index_max[0] / 2;
		//index_min[0] = 0;

		mapping2GLubyte_range_val << <N / 128, 128, 0, streams[0] >> > (rho_abs[0], index_max[0], index_min[0], rho_mapping[0], 2);
		mapping2GLubyte_range_val << <N / 128, 128, 0, streams[1] >> > (rho_abs[1], index_max[1], index_min[1], rho_mapping[1], 2);
		mapping2GLubyte_range_val << <N / 128, 128, 0, streams[2] >> > (rho_abs[2], index_max[2], index_min[2], rho_mapping[2], 2);


		permute << <N / 128, 128, 0, streams[0] >> > (rho_mapping[0], rho_permute[0], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[1] >> > (rho_mapping[1], rho_permute[1], Nslice, My, Mx);
		permute << <N / 128, 128, 0, streams[2] >> > (rho_mapping[2], rho_permute[2], Nslice, My, Mx);

		expand << <N / 128, 128, 0, streams[0] >> > (rho_permute[0], checkImage_d[0], My, Mx, p_width);
		expand << <N / 128, 128, 0, streams[1] >> > (rho_permute[1], checkImage_d[1], My, Mx, p_width);
		expand << <N / 128, 128, 0, streams[2] >> > (rho_permute[2], checkImage_d[2], My, Mx, p_width);

		gpuErrchk(cudaMemcpyAsync(checkImage_h1, checkImage_d[0], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[0]));
		gpuErrchk(cudaMemcpyAsync(checkImage_h2, checkImage_d[1], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[1]))
		gpuErrchk(cudaMemcpyAsync(checkImage_h3, checkImage_d[2], N * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[2]));

		for (int i = 0; i < NUM_STREAMS; i++)
			gpuErrchk(cudaStreamSynchronize(streams[i]));

		//sprintf(link2, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Error/ErrorFrame_(%d, %d, %d).bmp", 2, 4, nframe);
		//saveBitmap(link2, 6, 5, My, My, 4, 2, checkImage_h1);

		//sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Error/ErrorFrame_(%d, %d, %d).bmp", 2, 4, nframe + 1);
		//saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h2);

		//sprintf(linka1, "D:/Thesis/Thesis results/GPU/M3/64 - fMRI - 3S/Error/ErrorFrame_(%d, %d, %d).bmp", 2, 4, nframe + 2);
		//saveBitmap(linka1, 6, 5, My, My, 4, 2, checkImage_h3);

		//sprintf(link2, "D:/Results/%s/%s/%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe);
		sprintf(link2, "%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe);
		saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h1);

		//sprintf(link2, "D:/Results/%s/%s/%s/Error2-full/Error2_Frame_%d.bmp", outputname, method, imgres, nframe);
		sprintf(link2, "%s/Error2-full/Error2_Frame_%d.bmp", outURL, nframe);
		saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h1);

		if ((nframe + 1) < Nframe) {

			//sprintf(link2, "D:/Results/%s/%s/%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 1);
			sprintf(link2, "%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 1);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h2);

			//sprintf(link2, "D:/Results/%s/%s/%s/Error2-full/Error2_Frame_%d.bmp", outputname, method, imgres, nframe + 1);
			sprintf(link2, "%s/Error2-full/Error2_Frame_%d.bmp", outURL, nframe + 1);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h2);
		}

		if ((nframe + 2) <= Nframe) {
			//sprintf(link2, "D:/Results/%s/%s/%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outputname, method, imgres, y_cor, x_cor, nframe + 2);
			sprintf(link2, "%s/Error2-spc/Error2_Frame_(%d, %d, %d).bmp", outURL, y_cor, x_cor, nframe + 2);
			saveBitmap(link2, p_width, p_height, My, My, x_cor, y_cor, checkImage_h3);

			//sprintf(link2, "D:/Results/%s/%s/%s/Error2-full/Error2_Frame_%d.bmp", outputname, method, imgres, nframe + 2);
			sprintf(link2, "%s/Error2-full/Error2_Frame_%d.bmp", outURL, nframe + 2);
			saveBitmap(link2, p_width * My, p_height * Mx, checkImage_h3);
		}

		/*if ((nframe + j + 4) <= 32) {
		n = sprintf(link2, "D:/workspace/error 2/ErrorFrame_%d.bmp", (nframe + j + 4));
		saveBitmap(link2, 16 * 64, 7 * 64, checkImage_h2);
		}*/

	}
	std::cout << "Sum Time " << sumTime << std::endl;
	std::cout << "-----------------" << GetCounter() << std::endl;
	std::cout << "Error full " << (double)sqrt(sumNom / sumDen) << std::endl;

}

double nrmse(double *a, double *b, int p_width, int p_height, int Mx, int My, int x_cor, int y_cor) {
	double sumMinus = 0.0;
	double sumDenom = 0.0;
	for (int y(0); y < My; ++y) {
		for (int x(0); x < Mx; ++x) {
			int idx = y * p_width*Mx + y_cor * p_width*Mx*My + x_cor * Mx + x;
			sumMinus += (a[idx] - b[idx])*(a[idx] - b[idx]);
			sumDenom += a[idx] * a[idx];
		}
	}

	return (double)sqrt(sumMinus / sumDenom);
}

void saveBitmap(const char *p_output, int p_width, int p_height, unsigned char *Image)
{
	BITMAPFILEHEADER bitmapFileHeader;
	memset(&bitmapFileHeader, 0xff, sizeof(BITMAPFILEHEADER));
	bitmapFileHeader.bfType = ('B' | 'M' << 8);
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	bitmapFileHeader.bfSize = bitmapFileHeader.bfOffBits + p_width * p_height; // multiply by 3 if you wanna switch to RGB

	BITMAPINFOHEADER bitmapInfoHeader;
	memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = p_width;
	bitmapInfoHeader.biHeight = -p_height;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 8; // this means grayscale, change to 24 if you wanna switch to RGB

	ofstream file(p_output, ios::binary);


	file.write(reinterpret_cast< char * >(&bitmapFileHeader), sizeof(bitmapFileHeader));
	file.write(reinterpret_cast< char * >(&bitmapInfoHeader), sizeof(bitmapInfoHeader));

	// bitmaps grayscale must have a table of colors... don't write this table if you want RGB
	unsigned char grayscale[4];

	for (int i(0); i < 256; ++i)
	{
		memset(grayscale, i, sizeof(grayscale));
		file.write(reinterpret_cast< char * >(grayscale), sizeof(grayscale));
	}

	// here we record the pixels... I created a gradient...
	// remember that the pixels order is from left to right, "bottom to top"...
	unsigned char pixel[1];
	for (int y(0); y < p_height; ++y)
	{
		for (int x(0); x < p_width; ++x) // + ( p_width % 4 ? ( 4 - p_width % 4 ) : 0 ) because BMP has a padding of 4 bytes per line
		{
			pixel[0] = Image[y*p_width + x];
			file.write(reinterpret_cast< char * >(pixel), sizeof(pixel));
		}
	}

	file.close();
}


void saveBitmap(const char *p_output, int p_width, int p_height, int Mx, int My, int x_cor, int y_cor, unsigned char *Image)
{
	BITMAPFILEHEADER bitmapFileHeader;
	memset(&bitmapFileHeader, 0xff, sizeof(BITMAPFILEHEADER));
	bitmapFileHeader.bfType = ('B' | 'M' << 8);
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	bitmapFileHeader.bfSize = bitmapFileHeader.bfOffBits + Mx * My; // multiply by 3 if you wanna switch to RGB

	BITMAPINFOHEADER bitmapInfoHeader;
	memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = Mx;
	bitmapInfoHeader.biHeight = -My;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 8; // this means grayscale, change to 24 if you wanna switch to RGB

	ofstream file(p_output, ios::binary);


	file.write(reinterpret_cast< char * >(&bitmapFileHeader), sizeof(bitmapFileHeader));
	file.write(reinterpret_cast< char * >(&bitmapInfoHeader), sizeof(bitmapInfoHeader));

	// bitmaps grayscale must have a table of colors... don't write this table if you want RGB
	unsigned char grayscale[4];

	for (int i(0); i < 256; ++i)
	{
		memset(grayscale, i, sizeof(grayscale));
		file.write(reinterpret_cast< char * >(grayscale), sizeof(grayscale));
	}

	// here we record the pixels... I created a gradient...
	// remember that the pixels order is from left to right, "bottom to top"...
	unsigned char pixel[1];
	for (int y(0); y < My; ++y)
	{
		for (int x(0); x < Mx; ++x) // + ( p_width % 4 ? ( 4 - p_width % 4 ) : 0 ) because BMP has a padding of 4 bytes per line
		{
			pixel[0] = Image[y*p_width*Mx + y_cor * p_width*Mx*My + x_cor * Mx + x];
			file.write(reinterpret_cast< char * >(pixel), sizeof(pixel));
		}
	}



	file.close();
}


void DisplayHeader()
{
	const int kb = 1024;
	const int mb = kb * kb;
	wcout << "NBody.GPU" << endl << "=========" << endl << endl;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;
	//wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		checkDeviceProperty(props);
		wcout << endl;
	}
}


void checkDeviceProperty(cudaDeviceProp deviceProp)
{
	if ((deviceProp.concurrentKernels == 0)) //check concurrent kernel support
	{
		printf("> GPU does not support concurrent kernel execution\n");
		printf("  CUDA kernel runs will be serialized\n");
	}
	if (deviceProp.asyncEngineCount == 0) //check concurrent data transfer support
	{
		printf("GPU does not support concurrent Data transer and overlaping of kernel execution & data transfer\n");
		printf("Mem copy call will be blocking calls\n");
	}
}

int replacestr(char *line, const char *search, const char *replace)
{
	int count;
	char *sp; // start of pattern

			  //printf("replacestr(%s, %s, %s)\n", line, search, replace);
	if ((sp = strstr(line, search)) == NULL) {
		return(0);
	}
	count = 1;
	int sLen = strlen(search);
	int rLen = strlen(replace);
	if (sLen > rLen) {
		// move from right to left
		char *src = sp + sLen;
		char *dst = sp + rLen;
		while ((*dst = *src) != '\0') { dst++; src++; }
	}
	else if (sLen < rLen) {
		// move from left to right
		int tLen = strlen(sp) - sLen;
		char *stop = sp + rLen;
		char *src = sp + sLen + tLen;
		char *dst = sp + rLen + tLen;
		while (dst >= stop) { *dst = *src; dst--; src--; }
	}
	memcpy(sp, replace, rLen);

	count += replacestr(sp + rLen, search, replace);

	return(count);
}


void printInfo() {
	printf("//////////////////////////////////////////////////////////////////////////////////////////\n");
	printf("///////////////////////////////ENTER AN OPTION TO RUN/////////////////////////////////////\n");
	printf("/////                     1 - fMRI data - 64 x 64 x 30 x 131                        //////\n");
	printf("/////                     2 - DCE - MRI data - 64 x 64 x 112 x 32                   //////\n");
	printf("/////                     3 - DCE - MRI data - 128 x 128 x 112 x 32                 //////\n");
	printf("/////                     4 - DCE - MRI data - 160 x 160 x 112 x 32                 //////\n");
	printf("/////                     5 - DCE - MRI data - 192 x 192 x 112 x 32                 //////\n");
	printf("/////                                                                               //////\n");
	printf("/////                                                                               //////\n");
	printf("/////                                                                               //////\n");
	printf("/////   *NOTE: Sampling trajectory variables: r = 0.5, alpha = 9                    //////\n");
	printf("//////////////////////////////////////////////////////////////////////////////////////////\n");
	printf("//////////////////////////////////////////////////////////////////////////////////////////\n");

	scanf("%d", &opt);
	switch (opt)
	{
	case 1:
		sprintf(name, "fMRI");
		sprintf(outputname, "fMRI");
		sprintf(under, "0.5");
		sprintf(imgres, "64x64");
		sprintf(method, "M3");
		Mx = 64;
		My = 64;
		Nslice = 30;
		Nframe = 131;
		p_width = 6;
		p_height = Nslice / p_width;
		nnz = 56000;

		break;
	case 2:
		sprintf(name, "DCE-MRI/64x64");
		sprintf(outputname, "DCE-MRI");
		sprintf(under, "0.5");
		sprintf(imgres, "64x64");
		sprintf(method, "M3");
		Mx = 64;
		My = 64;
		Nslice = 112;
		Nframe = 32;
		p_width = 16;
		p_height = Nslice / p_width;
		nnz = 187200;

		break;
	case 3:
		sprintf(name, "DCE-MRI/128x128");
		sprintf(outputname, "DCE-MRI");
		sprintf(under, "0.5");
		sprintf(imgres, "128x128");
		sprintf(method, "M3");
		Mx = 128;
		My = 128;
		Nslice = 112;
		Nframe = 32;
		p_width = 16;
		p_height = Nslice / p_width;
		nnz = 886528;

		break;
	case 4:
		sprintf(name, "DCE-MRI/160x160");
		sprintf(outputname, "DCE-MRI");
		sprintf(under, "0.5");
		sprintf(imgres, "160x160");
		sprintf(method, "M3");
		Mx = 160;
		My = 160;
		Nslice = 112;
		Nframe = 32;
		p_width = 16;
		p_height = Nslice / p_width;
		nnz = 1440800;

		break;
	case 5:
		sprintf(name, "DCE-MRI/192x192");
		sprintf(outputname, "DCE-MRI");
		sprintf(under, "0.5");
		sprintf(imgres, "192x192");
		sprintf(method, "M3");
		Mx = 192;
		My = 192;
		Nslice = 112;
		Nframe = 32;
		p_width = 16;
		p_height = Nslice / p_width;
		nnz = 2161344;

		break;

	default:
		sprintf(name, "fMRI");
		sprintf(outputname, "fMRI");
		sprintf(under, "0.5");
		sprintf(imgres, "64x64");
		sprintf(method, "M3");
		Mx = 64;
		My = 64;
		Nslice = 30;
		Nframe = 131;
		p_width = 6;
		p_height = Nslice / p_width;
		nnz = 56000;

		break;
	}
	N = Nslice * My*Mx;
label:
	printf("Input the coordinates (x,y) of the specific slice\n");
	printf("Must start from (0, 0) -> (%d, %d) \n", p_height - 1, p_width - 1);
	scanf("%d %d", &y_cor, &x_cor);

	if ((y_cor > p_height - 1) || (x_cor > p_width - 1)) {
		goto label;
	}

}

void info() {
	char buff[FILENAME_MAX], line[256];
	FILE *pFile;
	GetCurrentDir(buff, FILENAME_MAX);
	printf("Current working dir: %s\n", buff);
	replacestr(strcat(buff, "/Config.txt"), "\\", "/");
	pFile = fopen(buff, "r");
	if (pFile == NULL) perror("Error opening file");
	else {
		while (fgets(line, 256, pFile) != NULL) {
			char a[256], b[256];
			//puts(line);
			if (sscanf(line, "%s %s", a, b) == 2)
			{
				puts(line);
				replacestr(b, "\\", "/");
				checkInput(a, b);
			}
		}
		fclose(pFile);

	label:
		printf("Input the coordinates (x,y) of the specific slice\n");
		printf("Must start from (0, 0) -> (%d, %d) \n", p_height - 1, p_width - 1);
		scanf("%d %d", &y_cor, &x_cor);

		if ((y_cor > p_height - 1) || (x_cor > p_width - 1)) {
			goto label;
		}

	}
}

void checkInput(char *iff, char *in) {
	//char buff[FILENAME_MAX];
	if (strcmp(iff, "Input_dir:") == 0)
		strcpy(inURL, in);
	else if (strcmp(iff, "Output_dir:") == 0) {
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Error2-spc"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Error2-full"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Error1-spc"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Error1-full"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Rcst-spc"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Rcst-full"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Ref-full"));
		strcpy(outURL, in);
		mkdir(strcat(outURL, "/Ref-spc"));
		strcpy(outURL, in);
	}

	else if (strcmp(iff, "Nslice:") == 0)
		Nslice = atoi(in);
	else if (strcmp(iff, "My:") == 0)
		My = atoi(in);
	else if (strcmp(iff, "Mx:") == 0)
		Mx = atoi(in);
	else if (strcmp(iff, "Nframe:") == 0)
		Nframe = atoi(in);
	else if (strcmp(iff, "p_width:") == 0) {
		p_width = atoi(in);
		p_height = Nslice / p_width;
	}
	else if (strcmp(iff, "nnz:") == 0) {
		nnz = atoi(in);
	}
	else if (strcmp(iff, "r:") == 0) {
		strcpy(under, in);
	}

	N = Nslice * My*Mx;
}
