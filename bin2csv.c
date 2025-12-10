#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s trajectory.bin output.csv N steps\n", argv[0]);
        return 1;
    }

    const char *binfile = argv[1];
    const char *csvfile = argv[2];
    int N = atoi(argv[3]);      // number of bodies
    int steps = atoi(argv[4]);  // number of timesteps

    FILE *fin = fopen(binfile, "rb");
    if (!fin) {
        perror("Error opening input binary file");
        return 1;
    }

    FILE *fout = fopen(csvfile, "w");
    if (!fout) {
        perror("Error opening output csv file");
        fclose(fin);
        return 1;
    }

    double buf[3];
    long total_rows = (long)N * (long)steps;

    for (long i = 0; i < total_rows; i++) {
        size_t nread = fread(buf, sizeof(double), 3, fin);
        if (nread != 3) {
            fprintf(stderr,
                "ERROR: Binary file truncated! Expected %ld rows but only read %ld rows.\n",
                total_rows, i);
            fclose(fin);
            fclose(fout);
            return 2;
        }
        fprintf(fout, "%.15f,%.15f,%.15f\n",
                buf[0], buf[1], buf[2]);
    }

    fclose(fin);
    fclose(fout);

    printf("Done. Wrote %ld rows to %s\n", total_rows, csvfile);
    return 0;
}
