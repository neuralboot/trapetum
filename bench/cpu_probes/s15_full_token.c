// S15: ONE COMPLETE 671B DECODE TOKEN on CPU, every component measured.
// Answers Grok v6's estimate table (220-310ms "optimiste") with measurements.
// Components per token, 4-bit packed unless noted:
//   attention MLA weights  61 x 93.6 MB   (q_a,q_b,kv_a,kv_b,o_proj real dims)
//   KV cache reads (4K ctx) 61 x 2.36 MB  int8 compressed MLA KV
//   router                  58 x 0.9 MB
//   dense FFN               3  x 198 MB   (inter 18432)
//   shared expert           58 x 22 MB    (full gate/up/SiLU/requant/down)
//   routed experts          58 x 8 x 22MB (full structure, random picks)
//   lm_head                 1  x 462 MB
// Total ~18.4 GB streamed. Work-stealing chunks, spin barriers, all charged.
// Build: clang -O3 -march=armv8.4-a+dotprod -o s15 s15_full_token.c -lpthread -lm
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define HID   7168
#define INTER 2048
#define NLAYERS 61
#define DENSE_L 3
#define MOE_L  58
#define TOPK   8          // routed; +1 shared appended in phase lists
#define NEXP   9
#define POOL   24
#define ATT_SLICES 8
#define SHARED_SLICES 8
#define ROWB   3584       // packed bytes per 7168-weight row
#define ATT_ROWS 26112    // ~93.6 MB per layer
#define DENSE_ROWS 55296  // 3*18432 rows per dense layer (198 MB)
#define LM_ROWS 129024    // ~462 MB
#define RT_ROWS 256
#define CTX 4096
#define KVD 576
#define CHUNKA 128
#define CHUNKC 256
#define CH_ATT 256
#define CH_DENSE 512
#define CH_LM 1024
#define CH_KV 512

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

typedef struct { uint8_t *w_gate,*w_up,*w_down; float *s_gate,*s_up,*s_down; } expert_t;
static expert_t pool[POOL + SHARED_SLICES]; // routed samples + shared ring
static uint8_t *att_buf, *dense_buf, *lm_buf, *rt_buf;
static int8_t  *kv_buf;
static int NTHREADS = 8, TOKENS = 10;
static int picks[MOE_L][NEXP];
static int8_t x_act[HID]; static int8_t x_kv[KVD];
static float g_buf[NEXP][INTER], u_buf[NEXP][INTER];
static int8_t y_buf[NEXP][INTER]; static float yscale[NEXP];
static float out_buf[NEXP][HID];
static volatile float sinkf; static atomic_llong sinki;

static inline int32_t dot_row(const uint8_t *row, const int8_t *x, size_t packed_len){
    static const int8_t cbvals[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
    int8x16_t cb = vld1q_s8(cbvals);
    uint8x16_t masklo = vdupq_n_u8(0x0F);
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t b = 0; b + 16 <= packed_len; b += 16){
        uint8x16_t v  = vld1q_u8(row + b);
        int8x16_t wlo = vqtbl1q_s8(cb, vandq_u8(v, masklo));
        int8x16_t whi = vqtbl1q_s8(cb, vshrq_n_u8(v, 4));
        acc = vdotq_s32(acc, wlo, vld1q_s8(x));
        acc = vdotq_s32(acc, whi, vld1q_s8(x + 16));
        x += 32;
    }
    return vaddvq_s32(acc);
}
static inline int32_t dot_i8(const int8_t *row, const int8_t *x, size_t len){
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t b = 0; b + 16 <= len; b += 16)
        acc = vdotq_s32(acc, vld1q_s8(row+b), vld1q_s8(x+b));
    return vaddvq_s32(acc);
}

// parallel incompressible fill: uniform memset pages get eaten by the macOS
// memory compressor under pressure and every read pays decompression
typedef struct { uint8_t *p; size_t n; uint64_t seed; } fill_t;
static void *fill_worker(void *a){
    fill_t *f = (fill_t*)a; uint64_t r = f->seed;
    uint64_t *q = (uint64_t*)f->p;
    for (size_t i = 0; i < f->n/8; i++){ r ^= r<<13; r ^= r>>7; r ^= r<<17; q[i] = r; }
    return NULL;
}
static void fill_random(uint8_t *p, size_t n){
    int NF = 8; pthread_t th[8]; fill_t jobs[8];
    size_t chunk = (n/8/NF)*8;
    for (int t = 0; t < NF; t++){
        jobs[t] = (fill_t){ p + t*chunk, t==NF-1 ? n - t*chunk : chunk, 0x9E3779B97F4A7C15ULL*(t+1) };
        pthread_create(&th[t], NULL, fill_worker, &jobs[t]);
    }
    for (int t = 0; t < NF; t++) pthread_join(th[t], NULL);
}

static atomic_int bar_count = 0, bar_gen = 0;
static void barrier_wait(void){
    int gen = atomic_load(&bar_gen);
    if (atomic_fetch_add(&bar_count, 1) == NTHREADS - 1){
        atomic_store(&bar_count, 0); atomic_fetch_add(&bar_gen, 1);
    } else while (atomic_load_explicit(&bar_gen, memory_order_acquire) == gen)
        __asm__ volatile("yield");
}

// dynamic counters: [token][layer][phase 0..5] + lm per token
#define NPH 6
static atomic_int *ctrs; static atomic_int *lm_ctr;
#define CTR(tok,l,ph) ctrs[((tok)*NLAYERS + (l))*NPH + (ph)]

// component ms accumulators, measured by thread 0 between barriers
static double ms_att, ms_kv, ms_dense, ms_moe, ms_lm;

static void stream_rows(uint8_t *base, int nrows, int chunk, atomic_int *ctr){
    int t; long long s = 0;
    while ((t = atomic_fetch_add(ctr, 1)) * chunk < nrows){
        int r0 = t*chunk, r1 = r0+chunk > nrows ? nrows : r0+chunk;
        for (int r = r0; r < r1; r++)
            s += dot_row(base + (size_t)r*ROWB, x_act, ROWB);
    }
    atomic_fetch_add(&sinki, s);
}

static void *worker(void *arg){
    long tid = (long)arg;
    for (int tok = 0; tok < TOKENS; tok++){
        for (int l = 0; l < NLAYERS; l++){
            double t0 = tid==0 ? now_s() : 0;
            // phase 0: attention MLA weight GEMVs
            stream_rows(att_buf + (size_t)(l%ATT_SLICES)*ATT_ROWS*ROWB, ATT_ROWS, CH_ATT, &CTR(tok,l,0));
            barrier_wait();
            double t1 = tid==0 ? now_s() : 0;
            // phase 1: KV cache pass (int8) + router if MoE
            { int t; long long s = 0;
              while ((t = atomic_fetch_add(&CTR(tok,l,1), 1)) * CH_KV < CTX){
                  int r0=t*CH_KV, r1=r0+CH_KV>CTX?CTX:r0+CH_KV;
                  for (int r=r0;r<r1;r++)
                      s += dot_i8(kv_buf + ((size_t)l*CTX + r)*KVD, x_kv, KVD);
              }
              atomic_fetch_add(&sinki, s); }
            if (l >= DENSE_L)
                stream_rows(rt_buf + (size_t)(l-DENSE_L)*RT_ROWS*ROWB, RT_ROWS, 64, &CTR(tok,l,2));
            barrier_wait();
            double t2 = tid==0 ? now_s() : 0;
            if (l < DENSE_L){
                // dense FFN as row stream (SiLU cost proven negligible in S13-M)
                stream_rows(dense_buf, DENSE_ROWS, CH_DENSE, &CTR(tok,l,3));
                barrier_wait();
                if (tid==0){ ms_att+=(t1-t0)*1e3; ms_kv+=(t2-t1)*1e3; ms_dense+=(now_s()-t2)*1e3; }
            } else {
                int ml = l - DENSE_L;
                // phase A: gate+up for 9 experts (8 routed + shared), full structure
                int t;
                while ((t = atomic_fetch_add(&CTR(tok,l,3), 1)) < NEXP*(INTER/CHUNKA)){
                    int e = t/(INTER/CHUNKA), c = t%(INTER/CHUNKA);
                    const expert_t *ex = &pool[picks[ml][e]];
                    for (int r = c*CHUNKA; r < (c+1)*CHUNKA; r++){
                        g_buf[e][r] = (float)dot_row(ex->w_gate+(size_t)r*(HID/2), x_act, HID/2)*ex->s_gate[r]*0.017f;
                        u_buf[e][r] = (float)dot_row(ex->w_up  +(size_t)r*(HID/2), x_act, HID/2)*ex->s_up[r]  *0.017f;
                    }
                }
                barrier_wait();
                // phase B: SiLU + requant
                while ((t = atomic_fetch_add(&CTR(tok,l,4), 1)) < NEXP){
                    float amax=1e-8f, tv[INTER];
                    for (int r=0;r<INTER;r++){
                        float s = g_buf[t][r]/(1.0f+expf(-g_buf[t][r]));
                        tv[r]=s*u_buf[t][r]; float a=fabsf(tv[r]); if(a>amax)amax=a;
                    }
                    float q=127.0f/amax;
                    for (int r=0;r<INTER;r++) y_buf[t][r]=(int8_t)lrintf(tv[r]*q);
                    yscale[t]=amax/127.0f;
                }
                barrier_wait();
                // phase C: down
                while ((t = atomic_fetch_add(&CTR(tok,l,5), 1)) < NEXP*(HID/CHUNKC)){
                    int e=t/(HID/CHUNKC), c=t%(HID/CHUNKC);
                    const expert_t *ex = &pool[picks[ml][e]];
                    for (int r=c*CHUNKC;r<(c+1)*CHUNKC;r++)
                        out_buf[e][r]=(float)dot_row(ex->w_down+(size_t)r*(INTER/2), y_buf[e], INTER/2)*ex->s_down[r]*yscale[e];
                }
                barrier_wait();
                if (tid==0){ ms_att+=(t1-t0)*1e3; ms_kv+=(t2-t1)*1e3; ms_moe+=(now_s()-t2)*1e3; }
            }
        }
        // lm_head
        double t3 = tid==0 ? now_s() : 0;
        stream_rows(lm_buf, LM_ROWS, CH_LM, &lm_ctr[tok]);
        barrier_wait();
        if (tid==0) ms_lm += (now_s()-t3)*1e3;
    }
    return NULL;
}

int main(int argc, char **argv){
    if (argc > 1) TOKENS = atoi(argv[1]);
    if (argc > 2) NTHREADS = atoi(argv[2]);
    size_t att_sz = (size_t)ATT_SLICES*ATT_ROWS*ROWB, dense_sz=(size_t)DENSE_ROWS*ROWB;
    size_t lm_sz = (size_t)LM_ROWS*ROWB, rt_sz=(size_t)MOE_L*RT_ROWS*ROWB;
    size_t kv_sz = (size_t)NLAYERS*CTX*KVD;
    size_t exp_sz = (POOL+MOE_L)*(3UL*INTER*HID/2);
    printf("allocating: att %.2f GB, experts %.2f GB, dense %.2f GB, lm %.2f GB, kv %.2f GB, router %.2f GB\n",
        att_sz/1e9, exp_sz/1e9, dense_sz/1e9, lm_sz/1e9, kv_sz/1e9, rt_sz/1e9);
    att_buf=malloc(att_sz); dense_buf=malloc(dense_sz); lm_buf=malloc(lm_sz); rt_buf=malloc(rt_sz);
    kv_buf=malloc(kv_sz);
    if(!att_buf||!dense_buf||!lm_buf||!rt_buf||!kv_buf){ printf("alloc fail\n"); return 1; }
    fill_random(att_buf,att_sz); fill_random(dense_buf,dense_sz);
    fill_random(lm_buf,lm_sz); fill_random(rt_buf,rt_sz); fill_random((uint8_t*)kv_buf,kv_sz);
    for (int e = 0; e < POOL+SHARED_SLICES; e++){
        pool[e].w_gate=malloc((size_t)INTER*(HID/2)); pool[e].w_up=malloc((size_t)INTER*(HID/2));
        pool[e].w_down=malloc((size_t)HID*(INTER/2));
        pool[e].s_gate=malloc(INTER*4); pool[e].s_up=malloc(INTER*4); pool[e].s_down=malloc(HID*4);
        fill_random(pool[e].w_gate,(size_t)INTER*(HID/2));
        fill_random(pool[e].w_up,  (size_t)INTER*(HID/2));
        fill_random(pool[e].w_down,(size_t)HID*(INTER/2));
        for(int r=0;r<INTER;r++){ pool[e].s_gate[r]=0.008f; pool[e].s_up[r]=0.008f; }
        for(int r=0;r<HID;r++) pool[e].s_down[r]=0.008f;
    }
    srand(7);
    for (int i=0;i<HID;i++) x_act[i]=(int8_t)(rand()%15)-7;
    for (int i=0;i<KVD;i++) x_kv[i]=(int8_t)(rand()%15)-7;
    for (int l=0;l<MOE_L;l++){
        for (int k=0;k<TOPK;k++) picks[l][k]=rand()%POOL;
        picks[l][TOPK]=POOL+(l%SHARED_SLICES); // shared expert ring
    }
    ctrs = calloc((size_t)TOKENS*NLAYERS*NPH, sizeof(atomic_int));
    lm_ctr = calloc(TOKENS, sizeof(atomic_int));

    pthread_t th[16];
    int saved=TOKENS; TOKENS=1;   // warmup
    for (long t=0;t<NTHREADS;t++) pthread_create(&th[t],NULL,worker,(void*)t);
    for (int t=0;t<NTHREADS;t++) pthread_join(th[t],NULL);
    TOKENS=saved;
    memset(ctrs,0,(size_t)TOKENS*NLAYERS*NPH*sizeof(atomic_int));
    memset(lm_ctr,0,TOKENS*sizeof(atomic_int));
    ms_att=ms_kv=ms_dense=ms_moe=ms_lm=0;

    double t0=now_s();
    for (long t=0;t<NTHREADS;t++) pthread_create(&th[t],NULL,worker,(void*)t);
    for (int t=0;t<NTHREADS;t++) pthread_join(th[t],NULL);
    double dt=now_s()-t0;

    double total_gb = ((double)NLAYERS*ATT_ROWS*ROWB + (double)DENSE_L*DENSE_ROWS*ROWB + lm_sz + rt_sz + kv_sz
                     + (double)MOE_L*NEXP*(3UL*INTER*HID/2)) / 1e9;
    double ms_tok = dt*1000.0/TOKENS;
    printf("\nS15 FULL 671B DECODE TOKEN, pure CPU, %d threads, ctx %d\n", NTHREADS, CTX);
    printf("streamed per token: %.2f GB  (routed %.2f + shared %.2f + attention %.2f + dense %.2f + lm %.2f + kv %.2f + router %.2f)\n",
        total_gb, (double)MOE_L*TOPK*(3UL*INTER*HID/2)/1e9, (double)MOE_L*(3UL*INTER*HID/2)/1e9,
        (double)NLAYERS*ATT_ROWS*ROWB/1e9, (double)DENSE_L*DENSE_ROWS*ROWB/1e9, lm_sz/1e9, kv_sz/1e9, rt_sz/1e9);
    printf("tokens=%d  ->  %.1f ms/token  ->  %.2f tok/s FULL MODEL pure CPU on this M4\n",
        TOKENS, ms_tok, 1000.0/ms_tok);
    printf("component ms/token: attention %.1f | kv %.1f | dense %.1f | moe(routed+shared) %.1f | lm_head %.1f\n",
        ms_att/TOKENS, ms_kv/TOKENS, ms_dense/TOKENS, ms_moe/TOKENS, ms_lm/TOKENS);
    printf("effective throughput: %.1f GB/s\n", total_gb*1000.0/ms_tok);
    printf("hybrid arithmetic: non-routed components move to VRAM-resident GPU -> t_token ~ moe_routed share\n");
    return 0;
}
