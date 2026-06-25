// prof11.cu -- raw mma.sync (no wmma API) + register-pipelined dequant.
// Fragments are built by hand at the exact m16n8k16 register layout (PTX ISA),
// so correctness is guaranteed without guessing ldmatrix swizzles. Removes the
// wmma load_matrix_sync overhead. nvcc -O3 -arch=sm_86 prof11.cu -lcublas -o prof11
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define BM 128
#define BN 128
#define BK 16
#define NWARP 8
#define NEL (BK*BN/(NWARP*32))   // 8
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ void cpasync16(void* s,const void* g){
    unsigned a=__cvta_generic_to_shared(s); asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(g));
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void wait0(){ asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void mma16816(float* d,const unsigned* a,const unsigned* b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__global__ void dequant(const uint8_t* idx,const __half* cb,__half* W,int IC,int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y;
    if(j<OC) W[(size_t)i*OC+j]=cb[(size_t)idx[(size_t)i*OC+j]*OC+j];
}

__global__ void mma_kernel(const __half* __restrict__ X,const uint8_t* __restrict__ idx,
                           const __half* __restrict__ cb,float* __restrict__ Yf,int M,int IC,int OC){
    __shared__ __half  sX[2][BM*BK];
    __shared__ uint8_t sIdx[2][BK*BN];
    __shared__ __half  sW[2][BK*BN];
    const int tid=threadIdx.y*32+threadIdx.x, warp=threadIdx.y, lane=threadIdx.x;
    const int warpM=warp/2, warpN=warp%2, gid=lane>>2, tig=lane&3;
    const int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN, nb=IC/BK;
    float acc[2][8][4];
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<8;j++)for(int k=0;k<4;k++) acc[i][j][k]=0.f;

    auto load=[&](int kt,int buf){ int k0=kt*BK;
        #pragma unroll
        for(int c=tid;c<BM*BK/8;c+=NWARP*32){ int e=c*8,mm=e/BK,kk=e%BK; cpasync16(&sX[buf][e],&X[(size_t)(bm0+mm)*IC+k0+kk]); }
        for(int c=tid;c<BK*BN/16;c+=NWARP*32){ int e=c*16,kk=e/BN,nn=e%BN; cpasync16(&sIdx[buf][e],&idx[(size_t)(k0+kk)*OC+bn0+nn]); }
    };
    auto compute=[&](int buf){
        unsigned a[2][4], b[8][2];
        #pragma unroll
        for(int fm=0;fm<2;fm++){ int mr=warpM*32+fm*16;
            a[fm][0]=*(unsigned*)&sX[buf][(mr+gid)*BK   + tig*2];
            a[fm][1]=*(unsigned*)&sX[buf][(mr+gid+8)*BK + tig*2];
            a[fm][2]=*(unsigned*)&sX[buf][(mr+gid)*BK   + tig*2+8];
            a[fm][3]=*(unsigned*)&sX[buf][(mr+gid+8)*BK + tig*2+8];
        }
        #pragma unroll
        for(int fn=0;fn<8;fn++){ int nc=warpN*64+fn*8+gid;
            __half2 h0=__halves2half2(sW[buf][(tig*2+0)*BN+nc], sW[buf][(tig*2+1)*BN+nc]);
            __half2 h1=__halves2half2(sW[buf][(tig*2+8)*BN+nc], sW[buf][(tig*2+9)*BN+nc]);
            b[fn][0]=*(unsigned*)&h0; b[fn][1]=*(unsigned*)&h1;
        }
        #pragma unroll
        for(int fm=0;fm<2;fm++)for(int fn=0;fn<8;fn++) mma16816(acc[fm][fn],a[fm],b[fn]);
    };

    load(0,0); commit(); wait0(); __syncthreads();
    #pragma unroll
    for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32,nn=t%BN; sW[0][t]=__ldg(&cb[(size_t)sIdx[0][t]*OC+bn0+nn]); }
    __syncthreads();
    if(nb>1){ load(1,1); commit(); }
    for(int kt=0;kt<nb;kt++){
        int cur=kt&1,nxt=(kt+1)&1; __half wreg[NEL];
        if(kt+1<nb){ wait0(); __syncthreads();
            #pragma unroll
            for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32,nn=t%BN; wreg[q]=__ldg(&cb[(size_t)sIdx[nxt][t]*OC+bn0+nn]); } }
        compute(cur);
        __syncthreads();
        if(kt+1<nb){
            #pragma unroll
            for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32; sW[nxt][t]=wreg[q]; }
            if(kt+2<nb){ load(kt+2,cur); commit(); } }
        __syncthreads();
    }
    #pragma unroll
    for(int fm=0;fm<2;fm++)for(int fn=0;fn<8;fn++){
        int rb=bm0+warpM*32+fm*16, cb2=bn0+warpN*64+fn*8;
        Yf[(size_t)(rb+gid)*OC   + cb2+tig*2+0]=acc[fm][fn][0];
        Yf[(size_t)(rb+gid)*OC   + cb2+tig*2+1]=acc[fm][fn][1];
        Yf[(size_t)(rb+gid+8)*OC + cb2+tig*2+0]=acc[fm][fn][2];
        Yf[(size_t)(rb+gid+8)*OC + cb2+tig*2+1]=acc[fm][fn][3];
    }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dIdx; static float* dYf;
static int M=2048,IC=4096,OC=4096; static cublasHandle_t H;
void rk(){ mma_kernel<<<dim3(OC/BN,M/BM),dim3(32,NWARP)>>>(dX,dIdx,dCb,dYf,M,IC,OC); }
void rcb(){ const float al=1,be=0; cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,OC,M,IC,&al,dW,CUDA_R_16F,OC,dX,CUDA_R_16F,IC,&be,dYc,CUDA_R_16F,OC,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
float tm(int n,void(*f)()){ cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());
    cudaEventRecord(a);for(int i=0;i<n;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/n;}
int main(){
    cublasCreate(&H); cublasSetMathMode(H,CUBLAS_TENSOR_OP_MATH);
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)256*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for(auto&v:hx)v=__float2half(nf(r)*0.1f);
    CK(cudaMalloc(&dIdx,hi.size()));CK(cudaMalloc(&dCb,hc.size()*2));CK(cudaMalloc(&dX,hx.size()*2));
    CK(cudaMalloc(&dW,(size_t)IC*OC*2));CK(cudaMalloc(&dYc,(size_t)M*OC*2));CK(cudaMalloc(&dYf,(size_t)M*OC*4));
    CK(cudaMemcpy(dIdx,hi.data(),hi.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCb,hc.data(),hc.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    dequant<<<dim3(OC/256,IC),256>>>(dIdx,dCb,dW,IC,OC); CK(cudaDeviceSynchronize());
    rk(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); rcb(); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)M*OC); std::vector<__half> yc((size_t)M*OC);
    CK(cudaMemcpy(yf.data(),dYf,yf.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yc.data(),dYc,yc.size()*2,cudaMemcpyDeviceToHost));
    double me=0,den=0; for(size_t i=0;i<yf.size();i++){double a=yf[i],b=__half2float(yc[i]);me=fmax(me,fabs(a-b));den=fmax(den,fabs(b));}
    double flop=2.0*M*IC*OC; float tt=tm(50,rk), tc=tm(50,rcb);
    printf("prefill M=%d  mma.sync_raw %.3f ms (%.1f TFLOP/s)  cuBLAS %.3f ms (%.1f TFLOP/s)  ratio=%.2f  relerr=%.3g\n",
           M,tt,flop/(tt*1e-3)/1e12,tc,flop/(tc*1e-3)/1e12,tc/tt,me/den);
    return 0;
}
