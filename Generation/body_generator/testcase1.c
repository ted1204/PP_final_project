#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define G 1.0

typedef struct {
    double m;
    double x, y, z;
    double vx, vy, vz;
} Body;

void generate_three_body_orbits(Body bodies[3]) {
    // Central heavy star
    bodies[0].m  = 10.0;
    bodies[0].x  = 0.0;
    bodies[0].y  = 0.0;
    bodies[0].z  = 0.0;
    bodies[0].vx = 0.0;
    bodies[0].vy = 0.0;
    bodies[0].vz = 0.0;

    double M = bodies[0].m;
    double mu = G * M;

    // ----- Body 1: elliptical orbit -----
    double a1 = 1.0;     // semi-major axis
    double e1 = 0.4;     // eccentricity
    double rp1 = a1 * (1.0 - e1);  // periapsis radius

    bodies[1].m = 1.0;
    bodies[1].x = rp1;
    bodies[1].y = 0.0;
    bodies[1].z = 0.0;

    double v1 = sqrt(mu * (2.0/rp1 - 1.0/a1));

    bodies[1].vx = 0.0;
    bodies[1].vy = v1;
    bodies[1].vz = 0.0;

    // ----- Body 2: elliptical orbit -----
    double a2 = 1.8;
    double e2 = 0.3;
    double rp2 = a2 * (1.0 - e2);

    bodies[2].m = 0.5;
    bodies[2].x = 0.0;
    bodies[2].y = rp2;
    bodies[2].z = 0.0;

    double v2 = sqrt(mu * (2.0/rp2 - 1.0/a2));

    bodies[2].vx = -v2;
    bodies[2].vy =  0.0;
    bodies[2].vz =  0.0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s output.bin\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot open output file");
        return 1;
    }


    int n = 3;
    fwrite(&n, sizeof(int), 1, fp);

    Body bodies[3];
    generate_three_body_orbits(bodies);

    for (int i = 0; i < n; i++) {
        fwrite(&bodies[i].m,  sizeof(double), 1, fp);
        fwrite(&bodies[i].x,  sizeof(double), 1, fp);
        fwrite(&bodies[i].y,  sizeof(double), 1, fp);
        fwrite(&bodies[i].z,  sizeof(double), 1, fp);
        fwrite(&bodies[i].vx, sizeof(double), 1, fp);
        fwrite(&bodies[i].vy, sizeof(double), 1, fp);
        fwrite(&bodies[i].vz, sizeof(double), 1, fp);
    }

    fclose(fp);
    printf("Generated 3-body elliptical orbit initial conditions → %s\n", filename);

    return 0;
}
