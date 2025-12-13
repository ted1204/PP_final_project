/*
nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -o pp_final_project/gpu pp_final_project/gpu.cu -lm -lcublas

srun -p nvidia -N1 -n1 --gres=gpu:1 \
    ./pp_final_project/gpu \
    ./pp_final_project/Generation/body_generator/body50000.bin 0.005 100 pp_final_project/trajectory.bin
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define G 1.0
#define SOFTENING 1e-3

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double m;
} Body;

int N;
int steps;
double dt;
Body* bodies;

char* out_path;

// Host arrays
double *h_x, *h_y, *h_z;
double *h_vx, *h_vy, *h_vz;
double *h_m;

// Device arrays
double *d_x, *d_y, *d_z;
double *d_vx, *d_vy, *d_vz;
double *d_m;
double *d_ax, *d_ay, *d_az;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void Input(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s input.bin dt steps output.bin\n", argv[0]);
        exit(1);
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("fopen input"); exit(1); }
    dt = atof(argv[2]);
    steps = atoi(argv[3]);
    out_path = argv[4];

    fread(&N, sizeof(int), 1, fin);
    bodies = (Body *)malloc(N * sizeof(Body));

    h_x  = (double*)malloc(N * sizeof(double));
    h_y  = (double*)malloc(N * sizeof(double));
    h_z  = (double*)malloc(N * sizeof(double));
    h_vx = (double*)malloc(N * sizeof(double));
    h_vy = (double*)malloc(N * sizeof(double));
    h_vz = (double*)malloc(N * sizeof(double));
    h_m  = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        fread(&bodies[i].m,  sizeof(double), 1, fin);
        fread(&bodies[i].x,  sizeof(double), 1, fin);
        fread(&bodies[i].y,  sizeof(double), 1, fin);
        fread(&bodies[i].z,  sizeof(double), 1, fin);
        fread(&bodies[i].vx, sizeof(double), 1, fin);
        fread(&bodies[i].vy, sizeof(double), 1, fin);
        fread(&bodies[i].vz, sizeof(double), 1, fin);

        h_m[i]  = bodies[i].m;
        h_x[i]  = bodies[i].x;
        h_y[i]  = bodies[i].y;
        h_z[i]  = bodies[i].z;
        h_vx[i] = bodies[i].vx;
        h_vy[i] = bodies[i].vy;
        h_vz[i] = bodies[i].vz;
    }
    fclose(fin);
}

// --- GPU kernels ---

// 1. First half of Verlet
__global__ void integrate_verlet_kernel(double *x, double *y, double *z,
                                        double *vx, double *vy, double *vz,
                                        double *ax, double *ay, double *az,
                                        double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vx[i] += 0.5 * ax[i] * dt;
        vy[i] += 0.5 * ay[i] * dt;
        vz[i] += 0.5 * az[i] * dt;

        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

// 2. Warp Shuffle Acceleration Kernel
// Replaces the shared memory tiling with direct register shuffling
__global__ void compute_accels_kernel_warp(const double *x, const double *y, const double *z,
                                           const double *m,
                                           double *ax, double *ay, double *az,
                                           int n) {
    // Determine the "Target" body for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load Target Body data into registers
    // NOTE: Threads with i >= n MUST still participate in the shuffle loop 
    // to pass data to active threads, so we don't return early. 
    // We just load 0s for them.
    double my_x = (i < n) ? x[i] : 0.0;
    double my_y = (i < n) ? y[i] : 0.0;
    double my_z = (i < n) ? z[i] : 0.0;
    
    double ax_i = 0.0;
    double ay_i = 0.0;
    double az_i = 0.0;

    // The Warp Size is 32
    // We loop over the entire array in "tiles" of 32
    for (int tile = 0; tile < n; tile += 32) {
        
        // 1. Load "Source" body data (Collaborative Load)
        // threadIdx.x % 32 gives the lane ID (0-31)
        int src_idx = tile + (threadIdx.x % 32);
        
        double s_x = (src_idx < n) ? x[src_idx] : 0.0;
        double s_y = (src_idx < n) ? y[src_idx] : 0.0;
        double s_z = (src_idx < n) ? z[src_idx] : 0.0;
        double s_m = (src_idx < n) ? m[src_idx] : 0.0;

        // 2. Warp Ring: Rotate data 32 times
        // Unrolling allows the compiler to pipeline instructions efficiently
        #pragma unroll 32
        for (int j = 0; j < 32; j++) {
            
            // Calculate force between 'my' body and 'current source' body
            double dx = s_x - my_x;
            double dy = s_y - my_y;
            double dz = s_z - my_z;
            
            double dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            double invDist = rsqrt(dist2);
            double invDist3 = invDist * invDist * invDist;
            double f = G * s_m * invDist3;

            ax_i += f * dx;
            ay_i += f * dy;
            az_i += f * dz;

            // SHUFFLE: Pass source body to the next thread in the warp
            // (srcLane = laneId + 1 modulo 32)
            // 0xffffffff mask ensures all threads participate
            s_x = __shfl_sync(0xffffffff, s_x, (threadIdx.x + 1) % 32);
            s_y = __shfl_sync(0xffffffff, s_y, (threadIdx.x + 1) % 32);
            s_z = __shfl_sync(0xffffffff, s_z, (threadIdx.x + 1) % 32);
            s_m = __shfl_sync(0xffffffff, s_m, (threadIdx.x + 1) % 32);
        }
    }

    // Write result only if this thread represents a valid body
    if (i < n) {
        double inv_m = 1.0 / m[i];
        ax[i] = ax_i * inv_m;
        ay[i] = ay_i * inv_m;
        az[i] = az_i * inv_m;
    }
}

// 3. Second half of Verlet
__global__ void integrate_verlet_finish_kernel(double *vx, double *vy, double *vz,
                                               const double *ax, const double *ay, const double *az,
                                               double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vx[i] += 0.5 * ax[i] * dt;
        vy[i] += 0.5 * ay[i] * dt;
        vz[i] += 0.5 * az[i] * dt;
    }
}

// --- Simulation with GPU --- 
void gpu_alloc_and_copy() {
    CUDA_CHECK(cudaMalloc((void**)&d_x,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_y,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_z,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_vx, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_vy, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_vz, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_m,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_ay, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_az, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  h_y,  N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,  h_z,  N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m,  h_m,  N * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_ax, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ay, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_az, 0, N * sizeof(double)));
}

void gpu_free() {
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_m);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
}

void dump_state_binary_host(FILE *fp, Body *bodies, double *hx, double *hy, double *hz, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].x = hx[i];
        bodies[i].y = hy[i];
        bodies[i].z = hz[i];
    }
    // Optimization: fwrite in bigger chunks if possible, but struct is 56 bytes.
    // Keeping logic same as requested.
    for (int i = 0; i < n; i++) {
        fwrite(&bodies[i].x, sizeof(double), 1, fp);
        fwrite(&bodies[i].y, sizeof(double), 1, fp);
        fwrite(&bodies[i].z, sizeof(double), 1, fp);
    }
}

void Simulation() {
    gpu_alloc_and_copy();

    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen output"); exit(1); }

    int blockSize = 128; // Must be multiple of 32 for warp shuffle
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // No sharedBytes needed anymore
    
    for (int step = 1; step <= steps; step++) {
        // 1) half-step velocity + position
        integrate_verlet_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                                         d_vx, d_vy, d_vz,
                                                         d_ax, d_ay, d_az,
                                                         dt, N);
        CUDA_CHECK(cudaGetLastError());

        // 2) compute accelerations (WARP SHUFFLE VERSION)
        compute_accels_kernel_warp<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                                            d_m,
                                                            d_ax, d_ay, d_az,
                                                            N);
        CUDA_CHECK(cudaGetLastError());

        // 3) finish velocity
        integrate_verlet_finish_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz,
                                                                d_ax, d_ay, d_az,
                                                                dt, N);
        CUDA_CHECK(cudaGetLastError());

        // Copy back
        CUDA_CHECK(cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost));

        dump_state_binary_host(fout, bodies, h_x, h_y, h_z, N);
    }

    fclose(fout);
    gpu_free();
}

int main(int argc, char *argv[]) {
    clock_t start = clock();
    Input(argc, argv);
    Simulation();

    free(bodies);
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_m);

    clock_t end = clock();
    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("time: %.3f seconds\n", elapsed);
    return 0;
}