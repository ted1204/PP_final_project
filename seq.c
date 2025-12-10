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

int N;
int steps;
double dt;
double* ax;
double* ay;
double* az;
Body* bodies;

void Input(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return;
    }
    FILE *fin = fopen(argv[1], "rb");
    dt = atof(argv[2]);
    steps = atoi(argv[3]);
    fread(&N, sizeof(int), 1, fin);
    bodies = (Body *)malloc(N * sizeof(Body));
    ax = (double *)malloc(N * sizeof(double));
    ay = (double *)malloc(N * sizeof(double));
    az = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        fread(&bodies[i].m,  sizeof(double), 1, fin);
        fread(&bodies[i].x,  sizeof(double), 1, fin);
        fread(&bodies[i].y,  sizeof(double), 1, fin);
        fread(&bodies[i].z,  sizeof(double), 1, fin);
        fread(&bodies[i].vx, sizeof(double), 1, fin);
        fread(&bodies[i].vy, sizeof(double), 1, fin);
        fread(&bodies[i].vz, sizeof(double), 1, fin);
    }
    fclose(fin);
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

void nBody(){
    integrate_verlet(bodies, N, ax, ay, az, dt);
    compute_forces(bodies, N, ax, ay, az);
    integrate_verlet_finish(bodies, N, ax, ay, az, dt);
    return;
}

void Simulation(){
    FILE* fout = fopen("trajectory.bin", "wb");
    for(int step = 1; step <= steps; step++) {
        printf("step: %d\n", step);
        nBody();
        dump_state_binary(fout, bodies, N);
    }
    fclose(fout);
    return;
}

int main(int argc, char *argv[]) {
    clock_t start = clock();
    Input(argc, argv);
    Simulation();
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
