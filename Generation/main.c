// nbody.c
// 簡單的 N-body 重力模擬 (Velocity Verlet)
// Input file format:
// First non-empty line: integer n (number of bodies)
// Next n non-empty lines: mass px py pz vx vy vz
//
// Output:
// For each time step (including initial t=0) prints n lines:
//   x y z
// (so total STEPS blocks of n lines each)
//
// Compile: gcc -O2 -o nbody nbody.c -lm
// Run:     ./nbody input.csv [dt] [steps]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----- 可在此修改預設參數 ----- */
#define DEFAULT_G 6.67430e-11      /* 重力常數 (SI) */
#define DEFAULT_DT 0.01            /* 預設時間間隔 */
#define DEFAULT_STEPS 100          /* 預設輸出時間點數 (包含初始 t=0) */
#define SOFTENING 1e-9             /* 軟化項，避免 r=0 時除以 0 */
/* ------------------------------- */

int main(int argc, char **argv) {
    const char *filename;
    double G = DEFAULT_G;
    double dt = DEFAULT_DT;
    int STEPS = DEFAULT_STEPS;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.csv [dt] [steps]\n", argv[0]);
        return 1;
    }
    filename = argv[1];
    if (argc >= 3) dt = atof(argv[2]);
    if (argc >= 4) STEPS = atoi(argv[3]);

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    /* 讀取 n（跳過空行與註解行） */
    int n = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        // trim leading spaces
        char *p = line;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
        if (*p == '\0' || *p == '#') continue; // 空行或註解
        // 嘗試讀取整數
        if (sscanf(p, "%d", &n) == 1) break;
    }
    if (n <= 0) {
        fprintf(stderr, "Failed to read number of bodies (n) from file.\n");
        fclose(fp);
        return 1;
    }

    /* 配置陣列 */
    double *m = (double*)malloc(sizeof(double)*n);
    double *px = (double*)malloc(sizeof(double)*n);
    double *py = (double*)malloc(sizeof(double)*n);
    double *pz = (double*)malloc(sizeof(double)*n);
    double *vx = (double*)malloc(sizeof(double)*n);
    double *vy = (double*)malloc(sizeof(double)*n);
    double *vz = (double*)malloc(sizeof(double)*n);

    double *ax = (double*)malloc(sizeof(double)*n);
    double *ay = (double*)malloc(sizeof(double)*n);
    double *az = (double*)malloc(sizeof(double)*n);

    if (!m || !px || !py || !pz || !vx || !vy || !vz || !ax || !ay || !az) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* 讀取 n 個質點（允許逗號或空白作為分隔） */
    int read = 0;
    while (read < n && fgets(line, sizeof(line), fp)) {
        // trim leading
        char *p = line;
        while (*p && (*p == ' ' || *p == '\t')) p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        // 把逗號換成空白，方便 sscanf 處理
        for (char *q = p; *q; ++q) if (*q == ',') *q = ' ';
        double mm, x, y, z, vx0, vy0, vz0;
        int cnt = sscanf(p, "%lf %lf %lf %lf %lf %lf %lf",
                         &mm, &x, &y, &z, &vx0, &vy0, &vz0);
        if (cnt == 7) {
            m[read] = mm;
            px[read] = x;
            py[read] = y;
            pz[read] = z;
            vx[read] = vx0;
            vy[read] = vy0;
            vz[read] = vz0;
            read++;
        } else {
            // 嘗試解析用逗號分隔(已替換過)，或跳過
            continue;
        }
    }
    fclose(fp);

    if (read != n) {
        fprintf(stderr, "Warning: declared n=%d but only read %d bodies. Using read=%d\n", n, read, read);
        n = read;
    }

    /* 初始化加速度為 0 */
    for (int i = 0; i < n; ++i) { ax[i]=0.0; ay[i]=0.0; az[i]=0.0; }

    /* 計算初始加速度 (t = 0) */
    for (int i = 0; i < n; ++i) {
        double aix = 0.0, aiy = 0.0, aiz = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
            double r = sqrt(r2);
            double inv_r3 = 1.0 / (r2 * r); // 1/r^3
            double f = G * m[j] * inv_r3;   // acceleration contribution times m_j
            aix += f * dx;
            aiy += f * dy;
            aiz += f * dz;
        }
        // a_i = sum_j G m_j (r_j - r_i) / |r|^3
        ax[i] = aix;
        ay[i] = aiy;
        az[i] = aiz;
    }

    /* OUTPUT: we will print STEPS blocks (including initial positions at t=0).
       If you want only positions after each dt and not initial, start loop from step=1. */
    for (int step = 0; step < STEPS; ++step) {
        // Print positions for all bodies
        for (int i = 0; i < n; ++i) {
            // Output format: x y z (scientific)
            printf("%.10e %.10e %.10e\n", px[i], py[i], pz[i]);
        }
        // After printing last block, still compute next positions except after last step
        if (step == STEPS - 1) break;

        /* Velocity Verlet integration:
           x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
           compute a(t+dt) from new positions
           v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        */
        // 1) update positions using current velocities and accelerations
        for (int i = 0; i < n; ++i) {
            px[i] += vx[i]*dt + 0.5*ax[i]*dt*dt;
            py[i] += vy[i]*dt + 0.5*ay[i]*dt*dt;
            pz[i] += vz[i]*dt + 0.5*az[i]*dt*dt;
        }

        // 2) compute new accelerations a_new based on updated positions
        double *ax_new = (double*)malloc(sizeof(double)*n);
        double *ay_new = (double*)malloc(sizeof(double)*n);
        double *az_new = (double*)malloc(sizeof(double)*n);
        if (!ax_new || !ay_new || !az_new) {
            fprintf(stderr, "Memory allocation failed for ax_new\n");
            return 1;
        }
        for (int i = 0; i < n; ++i) { ax_new[i]=0.0; ay_new[i]=0.0; az_new[i]=0.0; }

        for (int i = 0; i < n; ++i) {
            double aix = 0.0, aiy = 0.0, aiz = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
                double r = sqrt(r2);
                double inv_r3 = 1.0 / (r2 * r);
                double f = G * m[j] * inv_r3;
                aix += f * dx;
                aiy += f * dy;
                aiz += f * dz;
            }
            ax_new[i] = aix;
            ay_new[i] = aiy;
            az_new[i] = aiz;
        }

        // 3) update velocities using average of old and new accelerations
        for (int i = 0; i < n; ++i) {
            vx[i] += 0.5 * (ax[i] + ax_new[i]) * dt;
            vy[i] += 0.5 * (ay[i] + ay_new[i]) * dt;
            vz[i] += 0.5 * (az[i] + az_new[i]) * dt;
        }

        // replace old accelerations with new ones
        for (int i = 0; i < n; ++i) {
            ax[i] = ax_new[i];
            ay[i] = ay_new[i];
            az[i] = az_new[i];
        }
        free(ax_new);
        free(ay_new);
        free(az_new);
    }

    /* 釋放記憶體 */
    free(m); free(px); free(py); free(pz);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);

    return 0;
}
