#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define G 1.0         // Gravitational constant (scaled)
#define SOFTENING 1e-3 // Softening factor to avoid singularities


typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double m;
} Body;

void init_bodies(Body *bodies, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        // Random positions in a cube [-1, 1]
        bodies[i].x = 2.0 * rand() / RAND_MAX - 1.0;
        bodies[i].y = 2.0 * rand() / RAND_MAX - 1.0;
        bodies[i].z = 2.0 * rand() / RAND_MAX - 1.0;

        // Initial velocities small
        bodies[i].vx = 0.2 * (rand() / (double)RAND_MAX - 0.5);
        bodies[i].vy = 0.2 * (rand() / (double)RAND_MAX - 0.5);
        bodies[i].vz = 0.2 * (rand() / (double)RAND_MAX - 0.5);


        // Mass between 0.5 and 1.5
        bodies[i].m = 0.5 + (double)rand() / RAND_MAX;
    }
}

void compute_forces(Body *bodies, int n, double *ax, double *ay, double *az) {
    // Reset accelerations
    for (int i = 0; i < n; i++) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
    }

    // Pairwise forces
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            double invDist = 1.0 / sqrt(dist2);
            double invDist3 = invDist * invDist * invDist;
            double force = G * bodies[i].m * bodies[j].m * invDist3;

            double fx = force * dx;
            double fy = force * dy;
            double fz = force * dz;

            // a = F / m
            ax[i] += fx / bodies[i].m;
            ay[i] += fy / bodies[i].m;
            az[i] += fz / bodies[i].m;

            ax[j] -= fx / bodies[j].m;
            ay[j] -= fy / bodies[j].m;
            az[j] -= fz / bodies[j].m;
        }
    }
}

void integrate_verlet(Body *bodies, int n, double *ax, double *ay, double *az, double dt) {
    for (int i = 0; i < n; i++) {

        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;
        
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

void integrate_verlet_finish(Body *bodies, int n, double *ax, double *ay, double *az, double dt) {
    for (int i = 0; i < n; i++) {
        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;
    }
}


void dump_state_binary(FILE *fp, Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        fwrite(&bodies[i].x, sizeof(double), 1, fp);
        fwrite(&bodies[i].y, sizeof(double), 1, fp);
        fwrite(&bodies[i].z, sizeof(double), 1, fp);
    }
}

// void dump_state(Body *bodies, int n, int step) {
//     // Simple text output: step index, body index, x y z
//     for (int i = 0; i < n; i++) {
//         printf("%d %d %.6f %.6f %.6f\n",
//                step, i, bodies[i].x, bodies[i].y, bodies[i].z);
//     }
// }

int main(int argc, char *argv[]) {
    clock_t start = clock();
    if (argc != 4) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    double dt = atof(argv[2]);
    int steps = atoi(argv[3]);
    if (N <= 0) {
        fprintf(stderr, "N must be positive\n");
        return 1;
    }

    FILE *fp = fopen("trajectory.bin", "wb");
    if (!fp) {
        perror("File open failed");
        exit(1);
    }


    Body *bodies = (Body *)malloc(N * sizeof(Body));
    double *ax = (double *)malloc(N * sizeof(double));
    double *ay = (double *)malloc(N * sizeof(double));
    double *az = (double *)malloc(N * sizeof(double));
    if (!bodies || !ax || !ay || !az) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_bodies(bodies, N);

    for (int step = 1; step <= steps; step++) {
        if (step % 100 == 0) printf("step %d\n", step);
        integrate_verlet(bodies, N, ax, ay, az, dt);
        compute_forces(bodies, N, ax, ay, az);
        integrate_verlet_finish(bodies, N, ax, ay, az, dt);
        dump_state_binary(fp, bodies, N);
        printf("velocity: %.2f\n", bodies[1].vx);
        
    }

    // dump_state(bodies, N, step);
    free(bodies);
    free(ax);
    free(ay);
    free(az);
    clock_t end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("time: %.3f seconds\n", elapsed);
    return 0;
}
