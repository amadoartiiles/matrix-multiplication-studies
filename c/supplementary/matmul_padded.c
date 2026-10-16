/*
 * Supplementary layout experiment (NOT part of the controlled language comparison).
 *
 * Same i-j-k triple loop and arithmetic as c/matmul.c, but every matrix row is
 * stored with a leading dimension ld = n + pad doubles. pad = 0 reproduces the
 * main-study layout; pad = 2 shifts consecutive rows by 16 extra bytes, which is
 * the size of a Java array header and breaks the power-of-two row stride.
 *
 * The two layouts are measured alternately (pad0, pad2, pad0, pad2, ...) in one
 * process so that slow drifts of machine state affect both equally.
 *
 *   matmul_padded --n N --data DIR --pad P --reps R
 * Output CSV: n,pad,rep,inner,time_per_mult_s,checksum
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

static volatile double sink;

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

static double *multiply(const double *A, const double *B, int n, int ld) {
    double *C = (double *)calloc((size_t)n * ld, sizeof(double));
    if (C == NULL) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * ld + k] * B[(size_t)k * ld + j];
            }
            C[(size_t)i * ld + j] = s;
        }
    }
    return C;
}

static double checksum(const double *C, int n, int ld) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) s += C[(size_t)i * ld + j];
    return s;
}

static double *read_padded(const char *path, int n, int ld) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    double *M = (double *)calloc((size_t)n * ld, sizeof(double));
    for (int i = 0; i < n; i++) {
        if (fread(M + (size_t)i * ld, sizeof(double), n, f) != (size_t)n) {
            fclose(f);
            free(M);
            return NULL;
        }
    }
    fclose(f);
    return M;
}

/* Time one sample (batched to >= 0.1 s) for layout ld; returns time per multiplication. */
static double sample(const double *A, const double *B, int n, int ld, int inner, double *chk) {
    double *C = NULL;
    double t0 = now_seconds();
    for (int b = 0; b < inner; b++) {
        free(C);
        C = multiply(A, B, n, ld);
        sink = C[0];
    }
    double t1 = now_seconds();
    *chk = checksum(C, n, ld);
    free(C);
    return (t1 - t0) / inner;
}

int main(int argc, char **argv) {
    int n = 0, pad = 2, reps = 5;
    const char *data = "data";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--n") == 0) n = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--data") == 0) data = argv[i + 1];
        else if (strcmp(argv[i], "--pad") == 0) pad = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--reps") == 0) reps = atoi(argv[i + 1]);
    }
    if (n <= 0) { fprintf(stderr, "--n is required\n"); return 1; }
    char pa[1024], pb[1024];
    snprintf(pa, sizeof pa, "%s/A_%d.bin", data, n);
    snprintf(pb, sizeof pb, "%s/B_%d.bin", data, n);
    int pads[2] = {0, pad};
    double *A[2], *B[2];
    int inner[2];
    for (int v = 0; v < 2; v++) {
        A[v] = read_padded(pa, n, n + pads[v]);
        B[v] = read_padded(pb, n, n + pads[v]);
        if (A[v] == NULL || B[v] == NULL) { fprintf(stderr, "cannot read inputs\n"); return 2; }
        double chk;
        double t = sample(A[v], B[v], n, n + pads[v], 1, &chk); /* warm-up + calibration */
        inner[v] = t < 0.1 ? (int)ceil(0.1 / t) : 1;
    }
    printf("n,pad,rep,inner,time_per_mult_s,checksum\n");
    for (int r = 0; r < reps; r++) {
        for (int v = 0; v < 2; v++) {
            double chk;
            double t = sample(A[v], B[v], n, n + pads[v], inner[v], &chk);
            printf("%d,%d,%d,%d,%.9e,%.17g\n", n, pads[v], r, inner[v], t, chk);
            fflush(stdout);
        }
    }
    return 0;
}
