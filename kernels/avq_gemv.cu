// Fused additive vector-quantization decode GEMV (AQLM-style).
// AVQ = Additive Vector Quantization (the compression scheme; nothing else).
//
// A group of D consecutive input weights for output o is reconstructed as a SUM of
// M codebook vectors:  w_group = sum_m C_m[ code_m[o,g] ]  (C_m is [K, D]).
// For y[o] = sum_g <x_g, w_group>, the dot <x_g, C_m[k]> is independent of the
// output, so we precompute LUT[m][k] = <x_g, C_m[k]> once per group (in shared),
// then y[o] = sum_g sum_m LUT[m][ code_m[o,g] ]. The kernel reads the CODES, never
// the dense weights: at 2 bits (M=2, K=256, D=8) that is 4 MB of codes vs 32 MB of
// fp16 weight, the whole point. Split-K over groups (grid.y) for occupancy.
//
// build:  nvcc -O3 -arch=sm_89 avq_gemv.cu -lcublas -o avq && ./avq
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define M 2          // number of additive codebooks
#define K 256        // entries per codebook (8-bit codes)
#define D 8          // group dimension
#define CPB 256      // output channels per block
#ifndef GS
#define GS 20        // split-K groups
#endif

__global__ void avq_gemv(const __half* __restrict__ X, const unsigned char* __restrict__ codes,
                         const __half* __restrict__ CB, float* __restrict__ Y, int IC, int OC) {
    int ng = IC / D;
    int o = blockIdx.x * CPB + threadIdx.x;
    int per = (ng + gridDim.y - 1) / gridDim.y;
    int g0 = blockIdx.y * per, g1 = min(ng, g0 + per);
    __shared__ __half s_CB[M*K*D];
    for (int t = threadIdx.x; t < M*K*D; t += CPB) s_CB[t] = CB[t];
    __shared__ float s_LUT[M*K];
    __shared__ __half s_x[D];
    float acc = 0;
    __syncthreads();
    for (int g = g0; g < g1; g++) {
        if (threadIdx.x < D) s_x[threadIdx.x] = X[g*D + threadIdx.x];
        __syncthreads();
        for (int t = threadIdx.x; t < M*K; t += CPB) {       // LUT[m][k] = <x_g, C_m[k]>
            int m = t / K, k = t % K; float dd = 0;
            #pragma unroll
            for (int e = 0; e < D; e++) dd += __half2float(s_x[e]) * __half2float(s_CB[(m*K+k)*D + e]);
            s_LUT[t] = dd;
        }
        __syncthreads();
        if (o < OC) {
            #pragma unroll
            for (int m = 0; m < M; m++)
                acc += s_LUT[m*K + codes[((size_t)m*ng + g)*OC + o]];
        }
        __syncthreads();
    }
    if (o < OC) atomicAdd(&Y[o], acc);
}

// reference: reconstruct dense W[o, g*D+e] = sum_m C_m[code_m[o,g]][e]
__global__ void reconstruct(const unsigned char* codes, const __half* CB, __half* W, int IC, int OC) {
    int ng = IC / D;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OC) return;
    for (int g = 0; g < ng; g++)
        for (int e = 0; e < D; e++) {
            float v = 0;
            for (int m = 0; m < M; m++) v += __half2float(CB[(m*K + codes[((size_t)m*ng+g)*OC+o])*D + e]);
            W[(size_t)o*IC + g*D + e] = __float2half(v);
        }
}

int main() {
    int IC = 4096, OC = 4096, ng = IC / D;
    std::mt19937 rng(0); std::normal_distribution<float> nd(0, 0.05f);
    std::uniform_int_distribution<int> ud(0, K-1);

    std::vector<__half> hX(IC), hCB(M*K*D);
    std::vector<unsigned char> hcodes((size_t)M*ng*OC);
    for (auto& v : hX) v = __float2half(nd(rng));
    for (auto& v : hCB) v = __float2half(nd(rng));
    for (auto& v : hcodes) v = (unsigned char)ud(rng);

    __half *X, *CB, *W; unsigned char* codes; float *Y;
    cudaMalloc(&X, IC*2); cudaMalloc(&CB, M*K*D*2); cudaMalloc(&codes, (size_t)M*ng*OC);
    cudaMalloc(&W, (size_t)OC*IC*2); cudaMalloc(&Y, OC*4);
    cudaMemcpy(X, hX.data(), IC*2, cudaMemcpyHostToDevice);
    cudaMemcpy(CB, hCB.data(), M*K*D*2, cudaMemcpyHostToDevice);
    cudaMemcpy(codes, hcodes.data(), (size_t)M*ng*OC, cudaMemcpyHostToDevice);
    reconstruct<<<(OC+127)/128, 128>>>(codes, CB, W, IC, OC); cudaDeviceSynchronize();

    // our kernel
    dim3 grid(OC/CPB, GS), block(CPB);
    cudaMemset(Y, 0, OC*4);
    avq_gemv<<<grid, block>>>(X, codes, CB, Y, IC, OC);
    cudaDeviceSynchronize();
    std::vector<float> hY(OC); cudaMemcpy(hY.data(), Y, OC*4, cudaMemcpyDeviceToHost);

    // cuBLAS fp16 reference y = W @ x (dense)
    cublasHandle_t h; cublasCreate(&h);
    float *Yref; cudaMalloc(&Yref, OC*4);
    float al = 1, be = 0;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, OC, 1, IC, &al, W, CUDA_R_16F, IC,
                 X, CUDA_R_16F, IC, &be, Yref, CUDA_R_32F, OC, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    std::vector<float> hYref(OC); cudaMemcpy(hYref.data(), Yref, OC*4, cudaMemcpyDeviceToHost);

    double num = 0, den = 0;
    for (int i = 0; i < OC; i++) { double d = hY[i]-hYref[i]; num += d*d; den += (double)hYref[i]*hYref[i]; }
    printf("rel err vs dense cuBLAS = %.2e\n", sqrt(num/den));

    // timing
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); int it = 300;
    for (int i = 0; i < 30; i++) { cudaMemset(Y,0,OC*4); avq_gemv<<<grid,block>>>(X,codes,CB,Y,IC,OC); }
    cudaDeviceSynchronize(); cudaEventRecord(a);
    for (int i = 0; i < it; i++) { cudaMemset(Y,0,OC*4); avq_gemv<<<grid,block>>>(X,codes,CB,Y,IC,OC); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float t_avq; cudaEventElapsedTime(&t_avq, a, b); t_avq /= it;

    for (int i = 0; i < 30; i++) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,OC,1,IC,&al,W,CUDA_R_16F,IC,X,CUDA_R_16F,IC,&be,Yref,CUDA_R_32F,OC,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize(); cudaEventRecord(a);
    for (int i = 0; i < it; i++) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,OC,1,IC,&al,W,CUDA_R_16F,IC,X,CUDA_R_16F,IC,&be,Yref,CUDA_R_32F,OC,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float t_cub; cudaEventElapsedTime(&t_cub, a, b); t_cub /= it;

    cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0);
    printf("%s | M=%d K=%d D=%d GS=%d : %.2f bits/weight\n", pr.name, M, K, D, GS, (float)M*8/D);
    printf("avq_gemv  %.4f ms  |  cuBLAS fp16  %.4f ms  |  speedup x%.2f\n", t_avq, t_cub, t_cub/t_avq);
    printf("codes %.1f MB vs fp16 weight %.1f MB\n", (double)M*ng*OC/1e6, (double)OC*IC*2/1e6);
    return 0;
}
