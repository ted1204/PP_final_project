#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------- parameters --------------------

// 總星數（建議你改成 10000，現在先給比較小方便測）
#define N_STARS 10000

// 重力常數，用你 simulation 的單位，這裡假設 G=1
#define G 1.0

// Plummer 球星團參數
const double CLUSTER_MASS = 10.0;   // 星團總質量
const double PLUMMER_A    = 1.0;    // Plummer scale length

// 銀河系軌道參數（非常簡化：flat rotation curve）
const double MW_R0    = 8.0;   // 星團質心離銀河中心的半徑
const double MW_VCIRC = 1.0;   // 在 R0 的圓軌道速度（flat rotation，常數）

// -------------------- types --------------------

typedef struct {
    double m;
    double x, y, z;
    double vx, vy, vz;
} Body;

// -------------------- random helpers --------------------

static double rand01(void) {
    // (0,1) 之間的 double
    return ( (double)rand() + 0.5 ) / ( (double)RAND_MAX + 1.0 );
}

// Box–Muller，高斯(0,1)
static double gauss01(void) {
    double u1 = rand01();
    double u2 = rand01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// -------------------- Plummer sampling --------------------

/*
 * 抽樣 Plummer 球的半徑 r
 * 使用標準反 CDF：
 * u ~ U(0,1), r = a / sqrt(u^{-2/3} - 1)
 */
static double sample_plummer_radius(double a) {
    double u = rand01();
    double u2 = pow(u, -2.0/3.0) - 1.0;
    return a / sqrt(u2);
}

/*
 * 給一個半徑 r，抽 isotropic 方向 → (x,y,z)
 */
static void sample_isotropic_direction(double r, double *x, double *y, double *z) {
    double u = 2.0 * rand01() - 1.0;   // cos(theta) in [-1,1]
    double phi = 2.0 * M_PI * rand01();
    double s = sqrt(1.0 - u*u);

    *x = r * s * cos(phi);
    *y = r * s * sin(phi);
    *z = r * u;
}

/*
 * 近似 Plummer 球的速度分散：
 * 使用整體 virial 估計的 global velocity dispersion
 * U_plummer = -3πGM^2 / (32a)
 * 2K = -U → K = 3πGM^2 / (64a)
 * K = (1/2) M <v^2> → <v^2> = 3πG M / (32 a)
 * 1D dispersion σ^2 = <v^2>/3 ≈ πGM / (32a)
 */
static double plummer_sigma(double G_const, double M, double a) {
    double s2 = M_PI * G_const * M / (32.0 * a);
    return sqrt(s2);
}

// -------------------- init function --------------------

void init_plummer_in_mw(Body *bodies, int n)
{
    if (n != N_STARS) {
        fprintf(stderr, "Warning: n != N_STARS, using N_STARS = %d\n", N_STARS);
    }

    srand((unsigned)time(NULL));

    // 每顆星質量相同
    double m_star = CLUSTER_MASS / (double)N_STARS;

    // 星團質心位置：在銀河系 x 軸上 (R0, 0, 0)
    double Xc = MW_R0;
    double Yc = 0.0;
    double Zc = 0.0;

    // 星團質心速度：在 y 方向做圓軌道（切向）
    double Vxc = 0.0;
    double Vyc = MW_VCIRC;
    double Vzc = 0.0;

    // Plummer 球內部速度分散（近似）
    double sigma = plummer_sigma(G, CLUSTER_MASS, PLUMMER_A);

    for (int i = 0; i < N_STARS; i++) {

        bodies[i].m = m_star;

        // 抽內部位置 (相對星團中心)
        double r = sample_plummer_radius(PLUMMER_A);
        double x_rel, y_rel, z_rel;
        sample_isotropic_direction(r, &x_rel, &y_rel, &z_rel);

        // 抽內部速度（高斯，各向同性），近似 virial 平衡
        double vx_rel = sigma * gauss01();
        double vy_rel = sigma * gauss01();
        double vz_rel = sigma * gauss01();

        // 實際的星體位置 = 星團質心 + 相對位置
        bodies[i].x = Xc + x_rel;
        bodies[i].y = Yc + y_rel;
        bodies[i].z = Zc + z_rel;

        // 實際速度 = 星團質心速度 + 內部速度
        bodies[i].vx = Vxc + vx_rel;
        bodies[i].vy = Vyc + vy_rel;
        bodies[i].vz = Vzc + vz_rel;
    }
}

// -------------------- main: write binary --------------------

int main(int argc, char *argv[])
{

    FILE *fp = fopen("testcase3.bin", "wb");
    if (!fp) {
        perror("cannot open output file");
        return 1;
    }

    Body *bodies = (Body *)malloc(sizeof(Body) * N_STARS);
    if (!bodies) {
        fprintf(stderr, "malloc failed\n");
        fclose(fp);
        return 1;
    }

    init_plummer_in_mw(bodies, N_STARS);

    int n = N_STARS;
    fwrite(&n, sizeof(int), 1, fp);

    for (int i = 0; i < N_STARS; i++) {
        fwrite(&bodies[i].m,  sizeof(double), 1, fp);
        fwrite(&bodies[i].x,  sizeof(double), 1, fp);
        fwrite(&bodies[i].y,  sizeof(double), 1, fp);
        fwrite(&bodies[i].z,  sizeof(double), 1, fp);
        fwrite(&bodies[i].vx, sizeof(double), 1, fp);
        fwrite(&bodies[i].vy, sizeof(double), 1, fp);
        fwrite(&bodies[i].vz, sizeof(double), 1, fp);
    }

    fclose(fp);
    free(bodies);

    printf("Generated Plummer sphere (N=%d, M=%.3f, a=%.3f)\n", N_STARS, CLUSTER_MASS, PLUMMER_A);
    printf("Cluster COM at (%.2f, %.2f, %.2f) with V=(%.2f, %.2f, %.2f)\n",
           MW_R0, 0.0, 0.0, 0.0, MW_VCIRC, 0.0);
    printf("Output -> %s\n", "testcase3.bin");
    return 0;
}
