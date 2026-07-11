// S13-M v2: full DeepSeek expert pipeline with DYNAMIC WORK-STEALING.
// Fixes v1's static 1-expert-per-thread split: on asymmetric P/E cores every
// layer ran at the speed of the slowest E-core. Here all threads cooperatively
// drain chunk queues per phase, so P-cores naturally take more work.
// Phases per layer: A) gate+up rows (chunks)  B) SiLU+requant per expert
//                   C) down rows (chunks). Spin barriers with yield.
// Build: clang -O3 -march=armv8.4-a+dotprod -o s13m2 s13m2_worksteal.c -lpthread -lm
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
#define LAYERS 58
#define TOPK   8
#define POOL   64
#define GATE_BYTES ((size_t)INTER*(HID/2))
#define DOWN_BYTES ((size_t)HID*(INTER/2))
#define EXP_BYTES  (3UL*INTER*HID/2)
#define CHUNKA 128     // rows of gate+up per task -> 16 tasks/expert -> 128 tasks/layer
#define CHUNKC 256     // rows of down per task    -> 28 tasks/expert -> 224 tasks/layer
#define NTA (TOPK*(INTER/CHUNKA))
#define NTC (TOPK*(HID/CHUNKC))

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

typedef struct { uint8_t *w_gate,*w_up,*w_down; float *s_gate,*s_up,*s_down; } expert_t;
static expert_t pool[POOL];
static int NTHREADS = 8, TOKENS = 10;
static int picks[LAYERS][TOPK];
static int8_t x_act[HID];
static float g_buf[TOPK][INTER], u_buf[TOPK][INTER];
static int8_t y_buf[TOPK][INTER];
static float yscale[TOPK];
static float out_buf[TOPK][HID];

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

static atomic_int bar_count = 0, bar_gen = 0;
static void barrier_wait(void){
    int gen = atomic_load(&bar_gen);
    if (atomic_fetch_add(&bar_count, 1) == NTHREADS - 1){
        atomic_store(&bar_count, 0); atomic_fetch_add(&bar_gen, 1);
    } else while (atomic_load_explicit(&bar_gen, memory_order_acquire) == gen)
        __asm__ volatile("yield");
}

// per-phase dynamic counters, one triple per layer per token, zeroed upfront
static atomic_int *ctrs;  // [token][layer][3]
#define CTR(tok,l,ph) ctrs[((tok)*LAYERS + (l))*3 + (ph)]

static void *worker(void *arg){
    (void)arg;
    for (int tok = 0; tok < TOKENS; tok++){
        for (int l = 0; l < LAYERS; l++){
            // phase A: gate+up rows, dynamic chunks
            int t;
            while ((t = atomic_fetch_add(&CTR(tok,l,0), 1)) < NTA){
                int e = t / (INTER/CHUNKA), c = t % (INTER/CHUNKA);
                const expert_t *ex = &pool[picks[l][e]];
                for (int r = c*CHUNKA; r < (c+1)*CHUNKA; r++){
                    g_buf[e][r] = (float)dot_row(ex->w_gate + (size_t)r*(HID/2), x_act, HID/2) * ex->s_gate[r] * 0.017f;
                    u_buf[e][r] = (float)dot_row(ex->w_up   + (size_t)r*(HID/2), x_act, HID/2) * ex->s_up[r]   * 0.017f;
                }
            }
            barrier_wait();
            // phase B: SiLU + requant, one task per expert
            while ((t = atomic_fetch_add(&CTR(tok,l,1), 1)) < TOPK){
                float amax = 1e-8f, tv[INTER];
                for (int r = 0; r < INTER; r++){
                    float s = g_buf[t][r] / (1.0f + expf(-g_buf[t][r]));
                    tv[r] = s * u_buf[t][r];
                    float a = fabsf(tv[r]); if (a > amax) amax = a;
                }
                float q = 127.0f / amax;
                for (int r = 0; r < INTER; r++) y_buf[t][r] = (int8_t)lrintf(tv[r]*q);
                yscale[t] = amax / 127.0f;
            }
            barrier_wait();
            // phase C: down rows, dynamic chunks
            while ((t = atomic_fetch_add(&CTR(tok,l,2), 1)) < NTC){
                int e = t / (HID/CHUNKC), c = t % (HID/CHUNKC);
                const expert_t *ex = &pool[picks[l][e]];
                for (int r = c*CHUNKC; r < (c+1)*CHUNKC; r++)
                    out_buf[e][r] = (float)dot_row(ex->w_down + (size_t)r*(INTER/2), y_buf[e], INTER/2) * ex->s_down[r] * yscale[e];
            }
            barrier_wait();
        }
    }
    return NULL;
}

static uint64_t rng = 0x243F6A8885A308D3ULL;
static inline uint64_t xr(void){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return rng; }

int main(int argc, char **argv){
    if (argc > 1) TOKENS = atoi(argv[1]);
    if (argc > 2) NTHREADS = atoi(argv[2]);
    printf("allocating %d experts x %.1f MB = %.2f GB, %d threads...\n",
           POOL, EXP_BYTES/1e6, (double)POOL*EXP_BYTES/1e9, NTHREADS);
    for (int e = 0; e < POOL; e++){
        pool[e].w_gate = malloc(GATE_BYTES); pool[e].w_up = malloc(GATE_BYTES);
        pool[e].w_down = malloc(DOWN_BYTES);
        pool[e].s_gate = malloc(INTER*4); pool[e].s_up = malloc(INTER*4);
        pool[e].s_down = malloc(HID*4);
        uint64_t *p;
        p=(uint64_t*)pool[e].w_gate; for(size_t i=0;i<GATE_BYTES/8;i++) p[i]=xr();
        p=(uint64_t*)pool[e].w_up;   for(size_t i=0;i<GATE_BYTES/8;i++) p[i]=xr();
        p=(uint64_t*)pool[e].w_down; for(size_t i=0;i<DOWN_BYTES/8;i++) p[i]=xr();
        for(int r=0;r<INTER;r++){ pool[e].s_gate[r]=0.008f; pool[e].s_up[r]=0.008f; }
        for(int r=0;r<HID;r++)    pool[e].s_down[r]=0.008f;
    }
    for (int i = 0; i < HID; i++) x_act[i] = (int8_t)(xr()%15) - 7;
    for (int l = 0; l < LAYERS; l++) for (int k = 0; k < TOPK; k++)
        picks[l][k] = (int)(xr() % POOL);

    ctrs = calloc((size_t)TOKENS*LAYERS*3, sizeof(atomic_int));
    pthread_t th[16];
    // warmup token
    int saved = TOKENS; TOKENS = 1;
    for (long t = 0; t < NTHREADS; t++) pthread_create(&th[t], NULL, worker, (void*)t);
    for (int t = 0; t < NTHREADS; t++) pthread_join(th[t], NULL);
    TOKENS = saved;
    memset(ctrs, 0, (size_t)TOKENS*LAYERS*3*sizeof(atomic_int));

    double t0 = now_s();
    for (long t = 0; t < NTHREADS; t++) pthread_create(&th[t], NULL, worker, (void*)t);
    for (int t = 0; t < NTHREADS; t++) pthread_join(th[t], NULL);
    double dt = now_s() - t0;

    double bytes_per_tok = (double)LAYERS * TOPK * EXP_BYTES;
    double ms_tok = dt * 1000.0 / TOKENS;
    double gbs = bytes_per_tok * TOKENS / dt / 1e9;
    printf("\nS13-M v2 WORK-STEALING (full expert: gate+up+SiLU+requant+down+scales, 174 barriers/token)\n");
    printf("tokens=%d threads=%d  time=%.2fs  ->  %.1f ms/token expert-side\n", TOKENS, NTHREADS, dt, ms_tok);
    printf("effective packed throughput: %.1f GB/s  (%.2f GB/token)\n", gbs, bytes_per_tok/1e9);
    printf("671B expert-side tok/s on THIS M4: %.2f\n", 1000.0/ms_tok);
    return 0;
}
