#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define G 1.0
#define N 70
#define MIN_DIST 0.2     // minimum distance between stars
#define V0 3.0           // flat rotation velocity

typedef struct {
    double m;
    double x, y, z;
    double vx, vy, vz;
} Body;

double rand01() {
    return (double)rand() / RAND_MAX;
}

void init_galaxy_disk(Body *bodies)
{
    srand(time(NULL));

    // ---------- central massive star ----------
    bodies[0].m  = 3000000.0;    // MUCH larger for stability
    bodies[0].x  = 0.0;
    bodies[0].y  = 0.0;
    bodies[0].z  = 0.0;
    bodies[0].vx = 0.0;
    bodies[0].vy = 0.0;
    bodies[0].vz = 0.0;

    // ---------- other stars ----------
    for (int i = 1; i < N; i++) {

        double x, y, z, r, theta;

        retry_position:

        r = 2.0 + rand01() * 2.0;       // radius 2 ~ 4 (stable range)
        theta = 2 * M_PI * rand01();

        x = r * cos(theta);
        y = r * sin(theta);
        z = 0.05 * (rand01() - 0.5);   // small disk thickness

        // ensure no two stars are too close
        for (int j = 0; j < i; j++) {
            double dx = x - bodies[j].x;
            double dy = y - bodies[j].y;
            double dz = z - bodies[j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < MIN_DIST * MIN_DIST)
                goto retry_position;   // resample if too close
        }

        bodies[i].x = x;
        bodies[i].y = y;
        bodies[i].z = z;

        // random mass ~ [0.5, 1.5]
        bodies[i].m = 0.5 + rand01();

        // stable rotation using flat rotation curve
        double vx = -V0 * sin(theta);
        double vy =  V0 * cos(theta);

        // add small random velocity noise for realism
        vx += 0.05 * (rand01() - 0.5);
        vy += 0.05 * (rand01() - 0.5);

        bodies[i].vx = vx;
        bodies[i].vy = vy;
        bodies[i].vz = 0.0;
    }
}

int main(int argc, char *argv[])
{

    FILE *fp = fopen("testcase2.bin", "wb");
    if (!fp) {
        perror("cannot open output file");
        return 1;
    }

    Body bodies[N];
    init_galaxy_disk(bodies);

    int n = N;
    fwrite(&n, sizeof(int), 1, fp);

    for (int i = 0; i < N; i++) {
        fwrite(&bodies[i].m,  sizeof(double), 1, fp);
        fwrite(&bodies[i].x,  sizeof(double), 1, fp);
        fwrite(&bodies[i].y,  sizeof(double), 1, fp);
        fwrite(&bodies[i].z,  sizeof(double), 1, fp);
        fwrite(&bodies[i].vx, sizeof(double), 1, fp);
        fwrite(&bodies[i].vy, sizeof(double), 1, fp);
        fwrite(&bodies[i].vz, sizeof(double), 1, fp);
    }

    fclose(fp);
    printf("Generated stable 70-body galaxy → %s\n", "testcase2");
    return 0;
}
