// S12: is the K=16 codebook format CPU-native via the NEON tbl instruction?
// Measures fused decode+dot throughput: packed 4-bit indices -> vqtbl1q_s8 lookup
// -> int8 dot product (SDOT). Reports GB/s of packed weights processed.
// Build: clang -O3 -march=armv8.4-a+dotprod -o s12 s12_tbl_gemv.c -lpthread
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// one "expert matrix" slice: rows of packed 4-bit indices, per-row dot with x
// layout: contiguous packed bytes, activation vector x (int8) reused per row
typedef struct {
    const uint8_t *packed;   // packed 4-bit indices
    const int8_t  *x;        // activations, int8, length = 2 * bytes_per_row
    size_t bytes_total;      // total packed bytes to process
    size_t bytes_per_row;    // packed bytes per output row
    int64_t sink;            // prevents dead-code elimination
} job_t;

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    const uint8_t *p = j->packed;
    const int8_t *xbase = j->x;
    size_t rows = j->bytes_total / j->bytes_per_row;
    static const int8_t cbvals[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
    int8x16_t cb = vld1q_s8(cbvals);
    uint8x16_t masklo = vdupq_n_u8(0x0F);
    int64_t sink = 0;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = p + r * j->bytes_per_row;
        const int8_t *x = xbase;
        int32x4_t acc = vdupq_n_s32(0);
        for (size_t b = 0; b + 16 <= j->bytes_per_row; b += 16) {
            uint8x16_t v  = vld1q_u8(row + b);
            uint8x16_t lo = vandq_u8(v, masklo);
            uint8x16_t hi = vshrq_n_u8(v, 4);
            // THE isomorphism: 16-entry codebook == one tbl lookup
            int8x16_t wlo = vqtbl1q_s8(cb, lo);
            int8x16_t whi = vqtbl1q_s8(cb, hi);
            int8x16_t xlo = vld1q_s8(x);
            int8x16_t xhi = vld1q_s8(x + 16);
            acc = vdotq_s32(acc, wlo, xlo);
            acc = vdotq_s32(acc, whi, xhi);
            x += 32;
        }
        sink += vaddvq_s32(acc);
    }
    j->sink = sink;
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    size_t total_mb = argc > 2 ? (size_t)atoi(argv[2]) : 1024; // packed MB total
    size_t bytes_per_row = 3584;             // 7168 weights per row, like a real expert column dim
    size_t total = total_mb * 1024ULL * 1024ULL;
    total -= total % (bytes_per_row * (size_t)nthreads);

    uint8_t *packed = malloc(total);
    int8_t  *x = malloc(2 * bytes_per_row);
    srand(42);
    for (size_t i = 0; i < total; i++) packed[i] = (uint8_t)rand();
    for (size_t i = 0; i < 2 * bytes_per_row; i++) x[i] = (int8_t)(rand() % 15 - 7);

    pthread_t th[64]; job_t jobs[64];
    size_t chunk = total / nthreads;
    // warmup pass (touch memory)
    volatile uint8_t w = 0; for (size_t i = 0; i < total; i += 4096) w ^= packed[i];

    double t0 = now_s();
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (job_t){ packed + t * chunk, x, chunk, bytes_per_row, 0 };
        pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    int64_t sink = 0;
    for (int t = 0; t < nthreads; t++) { pthread_join(th[t], NULL); sink += jobs[t].sink; }
    double dt = now_s() - t0;

    double gbs = total / dt / 1e9;
    double weights_per_s = (total * 2.0) / dt;   // 2 weights per packed byte
    printf("threads=%d packed=%zu MB time=%.3fs -> %.1f GB/s packed, %.1f Gweight/s (sink %lld)\n",
           nthreads, total_mb, dt, gbs, weights_per_s / 1e9, (long long)sink);
    printf("671B projection: 10GB experts/token at 4-bit = 5GB packed -> %.2f s/token -> %.1f tok/s (this metric alone)\n",
           5.0 / gbs, gbs / 5.0);
    free(packed); free(x);
    return 0;
}
