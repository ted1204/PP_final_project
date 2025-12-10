CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61

CCFLAGS = -O2
LDFLAGS = -lm
EXES = seq bin2csv

all: $(EXES)

seq_v1: seq_v1.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)

bin2csv: bin2csv.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)

seq: seq.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)