/*
 * Basic dense matrix multiplication (triple loop, i-j-k order) in C.
 *
 * Storage: one contiguous row-major buffer of IEEE 754 binary64 values per
 * matrix; element (i, j) of an n x n matrix is at index i*n + j.
 *
 * Usage:
 *   matmul selftest
 *   matmul bench --n N --data DIR --warmup W --reps R --min-batch-time T [--dump FILE]
 *
 * Bench output (stdout, CSV, one row per warm-up or measured repetition):
 *   language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Results are written here so the compiler cannot discard batched products. */
static volatile double sink;

/* ---------- monotonic timer ---------- */
static double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ---------- kernel: the timed region is exactly this function ---------- */
/* Allocates and zero-initialises C, then runs the straightforward triple loop. */
static double *multiply(const double *A, const double *B, int n) {
    double *C = (double *)calloc((size_t)n * n, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = s;
        }
    }
    return C;
}

/* Row-major sequential sum, used to consume the result. */
static double checksum(const double *C, int n) {
    double s = 0.0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) s += C[idx];
    return s;
}

static double *read_matrix(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    size_t count = (size_t)n * n;
    double *M = (double *)malloc(count * sizeof(double));
    if (M == NULL || fread(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "cannot read %zu doubles from %s\n", count, path);
        fclose(f);
        free(M);
        return NULL;
    }
    fclose(f);
    return M;
}

static int write_matrix(const char *path, const double *M, int n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    size_t count = (size_t)n * n;
    size_t written = fwrite(M, sizeof(double), count, f);
    fclose(f);
    return written == count;
}

/* ---------- self-test with exact known answers ---------- */
static int equal_exact(const double *X, const double *Y, int n) {
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
        if (X[idx] != Y[idx]) return 0;
    return 1;
}

static int check(const char *name, int ok) {
    printf("%-28s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

static int selftest(void) {
    int ok = 1;

    /* 1. Hand-computed 3x3 product (small integers, exact in binary64). */
    double A3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double B3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double E3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double *C3 = multiply(A3, B3, 3);
    ok &= check("known 3x3 product", equal_exact(C3, E3, 3));
    free(C3);

    /* 2. Identity on both sides, 4x4. */
    double A4[16], I4[16], Z4[16];
    for (int i = 0; i < 16; i++) {
        A4[i] = (double)(i * 3 - 7);
        I4[i] = (i % 5 == 0) ? 1.0 : 0.0;
        Z4[i] = 0.0;
    }
    double *AI = multiply(A4, I4, 4);
    double *IA = multiply(I4, A4, 4);
    ok &= check("A * I == A", equal_exact(AI, A4, 4));
    ok &= check("I * A == A", equal_exact(IA, A4, 4));
    free(AI);
    free(IA);

    /* 3. Zero matrix. */
    double *AZ = multiply(A4, Z4, 4);
    ok &= check("A * 0 == 0", equal_exact(AZ, Z4, 4));
    free(AZ);

    /* 4. Non-symmetric 2x2 (checks that A*B != B*A is respected). */
    double A2[4] = {1, 2, 0, 1}, B2[4] = {1, 0, 3, 1};
    double E2[4] = {7, 2, 3, 1};
    double *C2 = multiply(A2, B2, 2);
    ok &= check("non-commuting 2x2", equal_exact(C2, E2, 2));
    free(C2);

    /* 5. 1x1. */
    double a1 = 2.5, b1 = -4.0, e1 = -10.0;
    double *C1 = multiply(&a1, &b1, 1);
    ok &= check("1x1", equal_exact(C1, &e1, 1));
    free(C1);

    printf("selftest %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}

/* ---------- benchmark ---------- */
static int bench(int n, const char *data_dir, int warmup, int reps,
                 double min_batch_time, const char *dump_path) {
    char path_a[1024], path_b[1024];
    snprintf(path_a, sizeof path_a, "%s/A_%d.bin", data_dir, n);
    snprintf(path_b, sizeof path_b, "%s/B_%d.bin", data_dir, n);
    double *A = read_matrix(path_a, n);
    double *B = read_matrix(path_b, n);
    if (A == NULL || B == NULL) return 2;

    printf("language,n,phase,rep,inner,batch_time_s,time_per_mult_s,checksum\n");
    fflush(stdout);

    double last_single = 0.0;
    for (int w = 0; w < warmup; w++) {
        double t0 = now_seconds();
        double *C = multiply(A, B, n);
        double t1 = now_seconds();
        if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
        last_single = t1 - t0;
        printf("c,%d,warmup,%d,1,%.9e,%.9e,%.17g\n", n, w, last_single, last_single,
               checksum(C, n));
        fflush(stdout);
        free(C);
    }

    /* Batch very short multiplications so each sample lasts >= min_batch_time. */
    int inner = 1;
    if (last_single > 0.0 && last_single < min_batch_time)
        inner = (int)ceil(min_batch_time / last_single);

    double *C = NULL;
    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        for (int b = 0; b < inner; b++) {
            if (C != NULL) free(C);
            C = multiply(A, B, n);
            if (C == NULL) { fprintf(stderr, "allocation failed\n"); return 3; }
            sink = C[(size_t)b % ((size_t)n * n)]; /* every result is consumed */
        }
        double t1 = now_seconds();
        double batch = t1 - t0;
        printf("c,%d,measure,%d,%d,%.9e,%.9e,%.17g\n", n, r, inner, batch, batch / inner,
               checksum(C, n));
        fflush(stdout);
    }

    if (dump_path != NULL && C != NULL && !write_matrix(dump_path, C, n)) {
        fprintf(stderr, "cannot write %s\n", dump_path);
        return 4;
    }
    free(C);
    free(A);
    free(B);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        int n = 0, warmup = 1, reps = 5;
        double min_batch_time = 0.1;
        const char *data_dir = "data";
        const char *dump = NULL;
        for (int i = 2; i + 1 < argc; i += 2) {
            if (strcmp(argv[i], "--n") == 0) n = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--data") == 0) data_dir = argv[i + 1];
            else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--reps") == 0) reps = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--min-batch-time") == 0) min_batch_time = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];
        }
        if (n <= 0) { fprintf(stderr, "--n is required\n"); return 1; }
        return bench(n, data_dir, warmup, reps, min_batch_time, dump);
    }
    fprintf(stderr, "usage: matmul selftest | bench --n N [--data DIR] [--warmup W] "
                    "[--reps R] [--min-batch-time T] [--dump FILE]\n");
    return 1;
}
