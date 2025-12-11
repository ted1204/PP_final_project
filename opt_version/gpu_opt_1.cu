/*
nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -o pp_final_project/gpu pp_final_project/gpu.cu -lm -lcublas

srun -p nvidia -N1 -n1 --gres=gpu:1 \
    ./pp_final_project/gpu \
    ./pp_final_project/Generation/body_generator/body50000.bin 0.005 100 pp_final_project/trajectory.bin

優化重點：
- 將 pairwise force 計算 + verlet 積分以 CUDA kernel 實作

優化細項：
- 把 AoS（Array of Struct）改成 SoA（Structure of Arrays），加快 gpu 讀取記憶體的效率
- compute_forces 改成 CUDA kernel，加入 shared memory tiling
- 用 rsqrt 取代 sqrt，透過 gpu 加快平方根計算
- 把 integrate_verlet 和 integrate_verlet_finish 改成 integrate_verlet_kernel 和 integrate_verlet_finish_kernel，讓 thread 進行平行計算
- dump_state_binary_host 前，把 x y z 等資料從 gpu 拿回 cpu 進行紀錄，寫入檔案等等

結果紀錄：
- n = 50000、dt = 0.005、step = 100 -> 35.191 seconds
    
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

// Host arrays (contiguous) for easy device memcpy
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

        // copy into contiguous host arrays
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
// half-step velocity + position update (Verlet first half)
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

// compute accelerations using tiling in shared memory
__global__ void compute_accels_kernel(const double *x, const double *y, const double *z,
                                      const double *m,
                                      double *ax, double *ay, double *az,
                                      int n) {
    extern __shared__ double sdata[]; // layout: sx[blockDim.x], sy[...], sz[...], sm[...]
    double *sx = sdata;
    double *sy = sdata + blockDim.x;
    double *sz = sdata + 2*blockDim.x;
    double *sm = sdata + 3*blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }

    double ax_i = 0.0, ay_i = 0.0, az_i = 0.0;

    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    for (int t = 0; t < numTiles; t++) {
        int idx = t * blockDim.x + threadIdx.x;
        if (idx < n) {
            sx[threadIdx.x] = x[idx];
            sy[threadIdx.x] = y[idx];
            sz[threadIdx.x] = z[idx];
            sm[threadIdx.x] = m[idx];
        } else {
            sx[threadIdx.x] = 0.0;
            sy[threadIdx.x] = 0.0;
            sz[threadIdx.x] = 0.0;
            sm[threadIdx.x] = 0.0;
        }
        __syncthreads();

        int tileSizeEffective = min(blockDim.x, n - t * blockDim.x);
        for (int j = 0; j < tileSizeEffective; j++) {
            int jj = t * blockDim.x + j;
            if (jj == i) continue; // skip self
            double dx = sx[j] - xi;
            double dy = sy[j] - yi;
            double dz = sz[j] - zi;
            double dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            double invDist = rsqrt(dist2); // fast reciprocal sqrt
            double invDist3 = invDist * invDist * invDist;
            double f = G * sm[j] * invDist3;
            ax_i += f * dx;
            ay_i += f * dy;
            az_i += f * dz;
        }
        __syncthreads();
    }

    if (i < n) {
        // acceleration = force / mass_i
        ax[i] = ax_i / m[i];
        ay[i] = ay_i / m[i];
        az[i] = az_i / m[i];
    }
}

// finish half-step velocity update
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

    // initialize accelerations to zero on device
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
    // update bodies' positions for compatibility with original dump_state_binary
    for (int i = 0; i < n; i++) {
        bodies[i].x = hx[i];
        bodies[i].y = hy[i];
        bodies[i].z = hz[i];
    }
    for (int i = 0; i < n; i++) {
        fwrite(&bodies[i].x, sizeof(double), 1, fp);
        fwrite(&bodies[i].y, sizeof(double), 1, fp);
        fwrite(&bodies[i].z, sizeof(double), 1, fp);
    }
}

void Simulation() {
    // GPU setup
    gpu_alloc_and_copy();

    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen output"); exit(1); }

    // kernel launch configuration
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedBytes = 4 * blockSize * sizeof(double); // sx,sy,sz,sm

    for (int step = 1; step <= steps; step++) {
        // 1) half-step velocity + position update
        integrate_verlet_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                                         d_vx, d_vy, d_vz,
                                                         d_ax, d_ay, d_az,
                                                         dt, N);
        CUDA_CHECK(cudaGetLastError());

        // 2) compute accelerations on new positions
        compute_accels_kernel<<<gridSize, blockSize, sharedBytes>>>(d_x, d_y, d_z,
                                                                    d_m,
                                                                    d_ax, d_ay, d_az,
                                                                    N);
        CUDA_CHECK(cudaGetLastError());

        // 3) finish half-step velocity update
        integrate_verlet_finish_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz,
                                                                d_ax, d_ay, d_az,
                                                                dt, N);
        CUDA_CHECK(cudaGetLastError());

        // copy positions back to host for dumping
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
