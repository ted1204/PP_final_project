#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
gcc baseline_generator/baseline.c -o baseline_generator/baseline.exe
./baseline_generator/baseline 7777
*/

#define MIN_MASS 1.0      /* 質量最小 */
#define MAX_MASS 1000.0   /* 質量最大 */

#define MIN_X -10000.0      /* x 範圍 */
#define MAX_X  10000.0
#define MIN_Y -10000.0      /* y 範圍 */
#define MAX_Y  10000.0
#define MIN_Z -10000.0      /* z 範圍 */
#define MAX_Z  10000.0

#define MIN_V -100.0        /* 速度分量範圍 (vx, vy, vz) */
#define MAX_V  100.0

/* 若要改成單一速度大小 v (而非向量)，可修改輸出格式與 rand_double 的使用。 */

static double rand_double(double a, double b) {
    /* 產生 [a, b) 的 double */
    return a + (b - a) * ((double)rand() / (double)RAND_MAX);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n_particles> [output.csv]\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "n must be > 0\n");
        return 1;
    }

    const char *outfile = (argc >= 3) ? argv[2] : "baseline.csv";

    FILE *fp = fopen(outfile, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    /* 若你不想要 header，把下面兩行註解掉 */
    // fprintf(fp, "# m x y z vx vy vz\n");

    fprintf(fp, "%ld\n", (long)n);  /* 第一行寫入粒子數量 */

    /* 使用 time seed，使每次不同。若要固定重現，改成 srand(固定值) */
    srand((unsigned int)time(NULL));

    for (long i = 0; i < n; ++i) {
        double m  = rand_double(MIN_MASS, MAX_MASS);
        double x  = rand_double(MIN_X, MAX_X);
        double y  = rand_double(MIN_Y, MAX_Y);
        double z  = rand_double(MIN_Z, MAX_Z);
        double vx = rand_double(MIN_V, MAX_V);
        double vy = rand_double(MIN_V, MAX_V);
        double vz = rand_double(MIN_V, MAX_V);

        /* 每行輸出: m x y z vx vy vz （以空白或逗號分隔皆可，這裡用空白） */
        fprintf(fp, "%.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
                m, x, y, z, vx, vy, vz);
    }

    fclose(fp);
    printf("Wrote %ld particles to %s\n", n, outfile);
    return 0;
}
