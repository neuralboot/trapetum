// Fused additive-VQ decode GEMV, v3: vectorized uint32 code reads.
// v2 (sync reduction) did not help: the kernel is memory-access-efficiency bound, not
// sync bound (it sat at ~22% of the A40 roofline reading 1-byte codes). Here each
// thread owns 4 consecutive outputs and reads their 4 codes as ONE uint32 (128 B per
// warp transaction, full sector use) with 4 accumulators (ILP to hide load latency).
//
// build:  nvcc -O3 -arch=sm_86 -DGT=4 avq_gemv3.cu -lcublas -o avq3 && ./avq3
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef M
#define M 2          // additive codebooks (M=2 -> 2-bit, M=4 -> 4-bit at K=256, D=8)
#endif
#define K 256
#define D 8
#define CPB 256
#ifndef GT
#define GT 4
#endif

__global__ void avq_gemv3(const __half* __restrict__ X, const unsigned char* __restrict__ codes,
                          const __half* __restrict__ CB, float* __restrict__ Y, int IC, int OC) {
    int ng = IC / D;
    int o = (blockIdx.x * CPB + threadIdx.x) * 4;     // 4 outputs per thread
    int g0 = blockIdx.y * GT;
    __shared__ __half s_CB[M*K*D];
    __shared__ float s_LUT[M*GT*K];
    __shared__ __half s_x[GT*D];
    for (int t = threadIdx.x; t < M*K*D; t += CPB) s_CB[t] = CB[t];
    for (int t = threadIdx.x; t < GT*D; t += CPB) { int gg = g0 + t/D; s_x[t] = (gg<ng) ? X[gg*D + t%D] : __float2half(0.f); }
    __syncthreads();
    for (int t = threadIdx.x; t < M*GT*K; t += CPB) {
        int m = t/(GT*K), r = t%(GT*K), gt = r/K, k = r%K; float dd = 0;
        #pragma unroll
        for (int e = 0; e < D; e++) dd += __half2float(s_x[gt*D+e]) * __half2float(s_CB[(m*K+k)*D+e]);
        s_LUT[t] = dd;
    }
    __syncthreads();
    if (o < OC) {
        float a0=0,a1=0,a2=0,a3=0;
        #pragma unroll
        for (int gt = 0; gt < GT; gt++) {
            int g = g0 + gt; if (g >= ng) break;
            #pragma unroll
            for (int m = 0; m < M; m++) {
                unsigned cc = *reinterpret_cast<const unsigned*>(&codes[((size_t)m*ng + g)*OC + o]); // 4 codes
                const float* L = &s_LUT[(m*GT + gt)*K];
                a0 += L[cc & 0xFF]; a1 += L[(cc>>8) & 0xFF]; a2 += L[(cc>>16) & 0xFF]; a3 += L[(cc>>24) & 0xFF];
            }
        }
        atomicAdd(&Y[o], a0); atomicAdd(&Y[o+1], a1); atomicAdd(&Y[o+2], a2); atomicAdd(&Y[o+3], a3);
    }
}

__global__ void reconstruct(const unsigned char* codes, const __half* CB, __half* W, int IC, int OC) {
    int ng = IC / D, o = blockIdx.x*blockDim.x + threadIdx.x;
    if (o >= OC) return;
    for (int g = 0; g < ng; g++) for (int e = 0; e < D; e++) {
        float v = 0; for (int m = 0; m < M; m++) v += __half2float(CB[(m*K + codes[((size_t)m*ng+g)*OC+o])*D + e]);
        W[(size_t)o*IC + g*D + e] = __float2half(v);
    }
}

int main() {
    int IC = 4096, OC = 4096, ng = IC/D;
    std::mt19937 rng(0); std::normal_distribution<float> nd(0,0.05f); std::uniform_int_distribution<int> ud(0,K-1);
    std::vector<__half> hX(IC), hCB(M*K*D); std::vector<unsigned char> hcodes((size_t)M*ng*OC);
    for (auto& v: hX) v=__float2half(nd(rng));
    for (auto& v: hCB) v=__float2half(nd(rng));
    for (auto& v: hcodes) v=(unsigned char)ud(rng);
    __half *X,*CB,*W; unsigned char* codes; float *Y;
    cudaMalloc(&X,IC*2); cudaMalloc(&CB,M*K*D*2); cudaMalloc(&codes,(size_t)M*ng*OC);
    cudaMalloc(&W,(size_t)OC*IC*2); cudaMalloc(&Y,OC*4);
    cudaMemcpy(X,hX.data(),IC*2,cudaMemcpyHostToDevice); cudaMemcpy(CB,hCB.data(),M*K*D*2,cudaMemcpyHostToDevice);
    cudaMemcpy(codes,hcodes.data(),(size_t)M*ng*OC,cudaMemcpyHostToDevice);
    reconstruct<<<(OC+127)/128,128>>>(codes,CB,W,IC,OC); cudaDeviceSynchronize();

    dim3 grid(OC/(CPB*4), (ng+GT-1)/GT), block(CPB);
    cudaMemset(Y,0,OC*4); avq_gemv3<<<grid,block>>>(X,codes,CB,Y,IC,OC); cudaDeviceSynchronize();
    std::vector<float> hY(OC); cudaMemcpy(hY.data(),Y,OC*4,cudaMemcpyDeviceToHost);

    cublasHandle_t h; cublasCreate(&h); float *Yref; cudaMalloc(&Yref,OC*4); float al=1,be=0;
    cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,OC,1,IC,&al,W,CUDA_R_16F,IC,X,CUDA_R_16F,IC,&be,Yref,CUDA_R_32F,OC,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize(); std::vector<float> hYref(OC); cudaMemcpy(hYref.data(),Yref,OC*4,cudaMemcpyDeviceToHost);
    double num=0,den=0; for (int i=0;i<OC;i++){double d=hY[i]-hYref[i];num+=d*d;den+=(double)hYref[i]*hYref[i];}
    printf("rel err = %.2e\n", sqrt(num/den));

    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int it=300;
    for (int i=0;i<30;i++){cudaMemset(Y,0,OC*4); avq_gemv3<<<grid,block>>>(X,codes,CB,Y,IC,OC);}
    cudaDeviceSynchronize(); cudaEventRecord(a);
    for (int i=0;i<it;i++){cudaMemset(Y,0,OC*4); avq_gemv3<<<grid,block>>>(X,codes,CB,Y,IC,OC);}
    cudaEventRecord(b); cudaEventSynchronize(b); float t_avq; cudaEventElapsedTime(&t_avq,a,b); t_avq/=it;
    for (int i=0;i<30;i++) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,OC,1,IC,&al,W,CUDA_R_16F,IC,X,CUDA_R_16F,IC,&be,Yref,CUDA_R_32F,OC,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize(); cudaEventRecord(a);
    for (int i=0;i<it;i++) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,OC,1,IC,&al,W,CUDA_R_16F,IC,X,CUDA_R_16F,IC,&be,Yref,CUDA_R_32F,OC,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
    cudaEventRecord(b); cudaEventSynchronize(b); float t_cub; cudaEventElapsedTime(&t_cub,a,b); t_cub/=it;
    double codesMB=(double)M*ng*OC/1e6; cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("%s GT=%d : avq3 %.4f ms | cuBLAS %.4f ms | x%.2f | eff %.0f GB/s\n", pr.name, GT, t_avq, t_cub, t_cub/t_avq, codesMB/t_avq);
    return 0;
}
